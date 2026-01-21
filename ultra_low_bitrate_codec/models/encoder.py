"""
Improved Information Factorizer V2
Key improvements:
1. Larger hidden dimensions
2. Better temporal modeling with Conformer-style blocks
3. Explicit speaker removal from content
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RateReduction(nn.Module):
    """Temporal compression with learnable downsampling"""
    def __init__(self, input_dim, output_dim, factor):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv1d(
            input_dim, output_dim,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T', C)
        x = self.norm(x)
        return x


class ConformerBlock(nn.Module):
    """Simplified Conformer block for temporal modeling"""
    def __init__(self, dim, num_heads=4, kernel_size=31, dropout=0.1):
        super().__init__()
        
        # Feed-forward 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Self-attention
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),  # Pointwise expansion
            nn.GLU(dim=1),                # Gated Linear Unit
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),  # Depthwise
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),       # Pointwise
            nn.Dropout(dropout)
        )
        
        # Feed-forward 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(dim)
        
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
        
        # ========================================
        # SPEAKER BRANCH (Global)
        # ========================================
        self.speaker_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, spk_cfg['embedding_dim'])
        )
        # Attention pooling
        self.speaker_attn = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Speaker projection for removal from content
        self.speaker_to_content = nn.Linear(spk_cfg['embedding_dim'], input_dim)
        
        # ========================================
        # SEMANTIC BRANCH
        # ========================================
        self.semantic_reducer = RateReduction(
            input_dim, sem_cfg['hidden_dim'], 
            factor=sem_cfg.get('temporal_compression', 8)  # Default 8x
        )
        self.semantic_conformer = ConformerBlock(
            sem_cfg['hidden_dim'], num_heads=4, kernel_size=15
        )
        self.semantic_proj = nn.Sequential(
            nn.Linear(sem_cfg['hidden_dim'], sem_cfg['hidden_dim']),
            nn.SiLU(),
            nn.Linear(sem_cfg['hidden_dim'], sem_cfg['output_dim']),
            nn.LayerNorm(sem_cfg['output_dim'])
        )
        
        # ========================================
        # PROSODY BRANCH
        # ========================================
        self.prosody_reducer = RateReduction(
            input_dim, pro_cfg['hidden_dim'],
            factor=pro_cfg.get('temporal_compression', 16)  # Default 16x
        )
        self.prosody_conformer = ConformerBlock(
            pro_cfg['hidden_dim'], num_heads=4, kernel_size=31
        )
        self.prosody_proj = nn.Sequential(
            nn.Linear(pro_cfg['hidden_dim'], pro_cfg['hidden_dim']),
            nn.SiLU(),
            nn.Linear(pro_cfg['hidden_dim'], pro_cfg['output_dim']),
            nn.LayerNorm(pro_cfg['output_dim'])
        )
        
    def forward(self, x):
        """
        Args:
            x: HuBERT features (B, T, 768)
        Returns:
            semantic: (B, T/4, output_dim)
            prosody: (B, T/8, output_dim)
            speaker: (B, speaker_dim)
        """
        # 1. Extract speaker (global)
        attn_weights = self.speaker_attn(x)  # (B, T, 1)
        speaker_feat = torch.sum(x * attn_weights, dim=1)  # (B, 768)
        speaker_emb = self.speaker_encoder(speaker_feat)  # (B, 256)
        
        # 2. Remove speaker from content (explicit disentanglement)
        speaker_expanded = self.speaker_to_content(speaker_emb)  # (B, 768)
        x_content = x - speaker_expanded.unsqueeze(1)  # Subtract speaker info
        
        # 3. Extract semantic (4x compression)
        semantic = self.semantic_reducer(x_content)
        semantic = self.semantic_conformer(semantic)
        semantic = self.semantic_proj(semantic)
        
        # 4. Extract prosody (8x compression)
        prosody = self.prosody_reducer(x)  # Use full features for prosody
        prosody = self.prosody_conformer(prosody)
        prosody = self.prosody_proj(prosody)
        
        return semantic, prosody, speaker_emb
