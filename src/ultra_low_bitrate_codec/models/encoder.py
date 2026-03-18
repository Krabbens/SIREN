import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear, RMSNorm, BitConv1d, BitSnakeBeta

class RateReduction(nn.Module):
    """Temporal compression with learnable downsampling (BitNet)"""
    def __init__(self, input_dim, output_dim, factor):
        super().__init__()
        self.factor = factor
        
        self.conv = BitConv1d(
            input_dim, output_dim,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2
        )
        self.norm = RMSNorm(output_dim)
        
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T', C)
        x = self.norm(x)
        return x


class ConformerBlock(nn.Module):
    """BitNet simplified Conformer block"""
    def __init__(self, dim, num_heads=4, kernel_size=31, dropout=0.1):
        super().__init__()
        
        # Feed-forward 1 (BitNet)
        self.ff1 = nn.Sequential(
            RMSNorm(dim),
            BitLinear(dim, dim * 4),
            BitSnakeBeta(dim * 4),
            nn.Dropout(dropout),
            BitLinear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Self-attention
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module (BitNet)
        self.conv_norm = RMSNorm(dim)
        self.conv = nn.Sequential(
            BitConv1d(dim, dim * 2, 1),  # Pointwise expansion
            nn.GLU(dim=1),                # Gated Linear Unit
            # Depthwise conv - keeping standard for now as groups=dim is hard for BitNet
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim), 
            nn.BatchNorm1d(dim),
            BitSnakeBeta(dim),
            BitConv1d(dim, dim, 1),       # Pointwise
            nn.Dropout(dropout)
        )
        
        # Feed-forward 2 (BitNet)
        self.ff2 = nn.Sequential(
            RMSNorm(dim),
            BitLinear(dim, dim * 4),
            BitSnakeBeta(dim * 4),
            nn.Dropout(dropout),
            BitLinear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.final_norm = RMSNorm(dim)
        
    def forward(self, x):
        # x: (B, T, C)
        
        # FF1 (half residual)
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention
        attn_out, _ = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x))
        x = x + self.attn_dropout(attn_out)
        
        # Convolution
        conv_in = self.conv_norm(x).transpose(1, 2)  # (B, C, T)
        conv_out = self.conv(conv_in).transpose(1, 2)  # (B, T, C)
        x = x + conv_out
        
        # FF2 (half residual)
        x = x + 0.5 * self.ff2(x)
        
        return self.final_norm(x)



class InformationFactorizer(nn.Module):
    """
    Original Factorizer (V1) matching the pretrained checkpoint.
    Used for verification and legacy support.
    """
    def __init__(self, config):
        super().__init__()
        
        sem_cfg = config['model']['semantic']
        pro_cfg = config['model']['prosody']
        spk_cfg = config['model']['speaker']
        
        input_dim = 768
        
        # Speaker (Global) - Matching fixed structure
        self.speaker_encoder = nn.Sequential(
            BitLinear(input_dim, 512, bias=True),
            BitSnakeBeta(512),
            RMSNorm(512),
            BitLinear(512, spk_cfg['embedding_dim'], bias=True)
        )
        self.speaker_attn = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )
        self.speaker_to_content = BitLinear(spk_cfg['embedding_dim'], input_dim, bias=True)
        
        # Semantic (1 Block)
        self.semantic_reducer = RateReduction(
            input_dim, sem_cfg['hidden_dim'], 
            factor=sem_cfg.get('temporal_compression', 8)
        )
        # Note: Name is 'semantic_conformer' in V1, not 'semantic_encoder'
        self.semantic_conformer = ConformerBlock(sem_cfg['hidden_dim'], num_heads=4, kernel_size=15)
        
        self.semantic_proj = nn.Sequential(
            BitLinear(sem_cfg['hidden_dim'], sem_cfg['hidden_dim'], bias=True),
            BitSnakeBeta(sem_cfg['hidden_dim']),
            BitLinear(sem_cfg['hidden_dim'], sem_cfg['output_dim'], bias=True),
            RMSNorm(sem_cfg['output_dim'])
        )
        
        # Prosody (1 Block)
        self.prosody_reducer = RateReduction(
            input_dim, pro_cfg['hidden_dim'],
            factor=pro_cfg.get('temporal_compression', 16)
        )
        # Note: Name is 'prosody_conformer'
        self.prosody_conformer = ConformerBlock(pro_cfg['hidden_dim'], num_heads=4, kernel_size=31)
        
        self.prosody_proj = nn.Sequential(
            BitLinear(pro_cfg['hidden_dim'], pro_cfg['hidden_dim'], bias=True),
            BitSnakeBeta(pro_cfg['hidden_dim']),
            BitLinear(pro_cfg['hidden_dim'], pro_cfg['output_dim'], bias=True),
            RMSNorm(pro_cfg['output_dim'])
        )

    def forward(self, x):
        # 1. Speaker
        attn_weights = self.speaker_attn(x)
        speaker_feat = torch.sum(x * attn_weights, dim=1)
        speaker_emb = self.speaker_encoder(speaker_feat)
        
        # 2. Content
        speaker_expanded = self.speaker_to_content(speaker_emb)
        x_content = x - speaker_expanded.unsqueeze(1)
        
        # 3. Semantic
        semantic = self.semantic_reducer(x_content)
        semantic = self.semantic_conformer(semantic)
        semantic = self.semantic_proj(semantic)
        
        # 4. Prosody
        prosody = self.prosody_reducer(x)
        prosody = self.prosody_conformer(prosody)
        prosody = self.prosody_proj(prosody)
        
        return semantic, prosody, speaker_emb


class InformationFactorizerV2(nn.Module):
    """
    Improved factorizer with better disentanglement.
    Outputs higher-dimensional features for RVQ.
    """
    def __init__(self, config):
        super().__init__()
        
        sem_cfg = config['model']['semantic']
        pro_cfg = config['model']['prosody']
        spk_cfg = config['model']['speaker']
        
        input_dim = 768  # HuBERT dimension
        cnn_dim = config['model'].get('cnn_dim', 256)    # CNN feature dimension (e.g. 256 for Micro, 384 for Bit)
        
        # Projection for CNN features
        self.acoustic_proj = BitLinear(cnn_dim, input_dim, bias=True)
        
        # ========================================
        # SPEAKER BRANCH (Global)
        # ========================================
        
        self.speaker_encoder = nn.Sequential(
            BitLinear(input_dim, 512, bias=True),
            BitSnakeBeta(512),
            RMSNorm(512),
            BitLinear(512, spk_cfg['embedding_dim'], bias=True),
            RMSNorm(spk_cfg['embedding_dim']) # Added for stability
        )
        # Attention pooling
        self.speaker_attn = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Speaker projection for removal from content
        self.speaker_to_content = BitLinear(spk_cfg['embedding_dim'], input_dim, bias=True)
        
        # ========================================
        # SEMANTIC BRANCH - Deeper
        # ========================================
        self.semantic_reducer = RateReduction(
            input_dim, sem_cfg['hidden_dim'], 
            factor=sem_cfg.get('temporal_compression', 8)
        )
        # Deep Conformer stacks
        self.semantic_encoder = nn.Sequential(*[
            ConformerBlock(sem_cfg['hidden_dim'], num_heads=4, kernel_size=15)
            for _ in range(3) # 3 Layers deep
        ])
        
        self.semantic_proj = nn.Sequential(
            BitLinear(sem_cfg['hidden_dim'], sem_cfg['hidden_dim'], bias=True),
            BitSnakeBeta(sem_cfg['hidden_dim']),
            nn.Linear(sem_cfg['hidden_dim'], sem_cfg['output_dim'], bias=True)
            # Replaced BitLinear with nn.Linear for the final bound projection
            # to prevent discrete initialization from collapsing into a single FSQ bin.
        )
        
        # ========================================
        # PROSODY BRANCH - Deeper
        # ========================================
        self.prosody_reducer = RateReduction(
            input_dim, pro_cfg['hidden_dim'],
            factor=pro_cfg.get('temporal_compression', 16)
        )
        self.prosody_encoder = nn.Sequential(*[
            ConformerBlock(pro_cfg['hidden_dim'], num_heads=4, kernel_size=31)
            for _ in range(3) # 3 Layers deep
        ])
        
        self.prosody_proj = nn.Sequential(
            BitLinear(pro_cfg['hidden_dim'], pro_cfg['hidden_dim'], bias=True),
            BitSnakeBeta(pro_cfg['hidden_dim']),
            nn.Linear(pro_cfg['hidden_dim'], pro_cfg['output_dim'], bias=True)
            # Replaced BitLinear with nn.Linear.
        )
        
        # Manually initialize the final linear layers with a larger variance
        # to ensure initial predictions span across multiple FSQ bins.
        nn.init.normal_(self.semantic_proj[2].weight, std=0.5)
        nn.init.zeros_(self.semantic_proj[2].bias)
        nn.init.normal_(self.prosody_proj[2].weight, std=0.5)
        nn.init.zeros_(self.prosody_proj[2].bias)
        
    def forward(self, x, x_acoustic=None):
        """
        Args:
            x: HuBERT features (B, T, 768) - Semantic
            x_acoustic: CNN features (B, T, 256) - Prosody/Speaker
        Returns:
            semantic: (B, T/4, output_dim)
            prosody: (B, T/8, output_dim)
            speaker: (B, speaker_dim)
        """
        # If x_acoustic is not provided (legacy), use x (projected if needed, but x is 768)
        # Actually x is 768. If we want to use x for acoustic, we use it directly.
        if x_acoustic is None:
            # Fallback for legacy scripts
            feat_acoustic = x 
        else:
            feat_acoustic = self.acoustic_proj(x_acoustic)
            
        # 1. Extract speaker (global) from ACOUSTIC features
        attn_weights = self.speaker_attn(feat_acoustic)  # (B, T, 1)
        speaker_feat = torch.sum(feat_acoustic * attn_weights, dim=1)  # (B, 768)
        speaker_emb = self.speaker_encoder(speaker_feat)  # (B, 256)
        
        # 2. Remove speaker from content (explicit disentanglement)
        # Content comes from SEMANTIC features (Transformer output)
        speaker_expanded = self.speaker_to_content(speaker_emb)  # (B, 768)
        x_content = x - speaker_expanded.unsqueeze(1)  # Subtract speaker info
        
        # 3. Extract semantic from CONTENT (Transformer - Speaker)
        semantic = self.semantic_reducer(x_content)
        semantic = self.semantic_encoder(semantic)
        semantic = self.semantic_proj(semantic)
        
        # 4. Extract prosody from ACOUSTIC features (CNN)
        # We don't subtract speaker from prosody? Ideally we should, but prosody is entangled with speaker.
        # Let's keep it raw from acoustic.
        prosody = self.prosody_reducer(feat_acoustic)
        prosody = self.prosody_encoder(prosody)
        prosody = self.prosody_proj(prosody)
        
        return semantic, prosody, speaker_emb
