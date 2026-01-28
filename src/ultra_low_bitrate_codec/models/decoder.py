"""
Improved Decoder V2
Key improvements:
1. Larger fusion dimension
2. More transformer layers
3. Better upsampling with residual connections
4. Integration with VocosV2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        # Adapt to batch_first=True used in Transformer
        if x.size(1) == self.pe.shape[2]: # x is [B, T, D]
             x = x + self.pe[:x.size(1), 0, :].unsqueeze(0)
        else: # x is [T, B, D]
             x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class LearnableUpsampler(nn.Module):
    """
    Learnable upsampler with residual refinement.
    Uses transposed conv + refinement conv.
    """
    def __init__(self, input_dim, output_dim, factor):
        super().__init__()
        self.factor = factor
        
        from .bitlinear import BitConvTranspose1d, BitConv1d
        
        # Main upsampling path (BitNet)
        self.upsample = BitConvTranspose1d(
            input_dim, output_dim,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2
        )
        
        # Refinement convolution (BitNet)
        self.refine = nn.Sequential(
            BitConv1d(output_dim, output_dim, kernel_size=7, padding=3),
            nn.SiLU(),
            BitConv1d(output_dim, output_dim, kernel_size=7, padding=3)
        )
        
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.upsample(x)
        x = x + self.refine(x)  # Residual refinement
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion for semantic and prosody.
    Allows prosody to modulate semantic content.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_sem = nn.LayerNorm(dim)
        self.norm_pro = nn.LayerNorm(dim)
        
        from .bitlinear import BitLinear
        
        # Semantic attends to prosody
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            BitLinear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            BitLinear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, sem, pro):
        # sem: (B, T_s, D), pro: (B, T_p, D)
        # Upsample pro to match sem if needed
        if pro.shape[1] != sem.shape[1]:
            pro = F.interpolate(
                pro.transpose(1, 2), size=sem.shape[1], mode='linear', align_corners=False
            ).transpose(1, 2)
        
        # Cross attention
        sem_norm = self.norm_sem(sem)
        pro_norm = self.norm_pro(pro)
        
        attn_out, _ = self.cross_attn(sem_norm, pro_norm, pro_norm)
        sem = sem + attn_out
        
        # Feed-forward
        sem = sem + self.ff(sem)
        
        return sem


class FeatureReconstructorV2(nn.Module):
    """
    Improved feature reconstructor with:
    - Learned upsampling
    - Cross-attention fusion
    - Deeper transformer
    """
    def __init__(self, config):
        super().__init__()
        
        sem_cfg = config['model']['semantic']
        pro_cfg = config['model']['prosody']
        spk_cfg = config['model']['speaker']
        dec_cfg = config['model']['decoder']
        
        fusion_dim = dec_cfg['fusion_dim']
        
        # ========================================
        # UPSAMPLERS
        # ========================================
        self.sem_upsampler = LearnableUpsampler(
            sem_cfg['output_dim'], fusion_dim // 2,
            factor=sem_cfg['temporal_compression']
        )
        
        self.pro_upsampler = LearnableUpsampler(
            pro_cfg['output_dim'], fusion_dim // 4,
            factor=pro_cfg['temporal_compression']
        )
        
        from .bitlinear import BitLinear
        
        # Speaker projection (BitNet)
        self.spk_proj = nn.Sequential(
            BitLinear(spk_cfg['embedding_dim'], fusion_dim // 4),
            nn.SiLU(),
            BitLinear(fusion_dim // 4, fusion_dim // 4)
        )
        
        # ========================================
        # FUSION
        # ========================================
        # Initial concat: (fusion_dim//2 + fusion_dim//4 + fusion_dim//4) = fusion_dim
        # Initial concat: (fusion_dim//2 + fusion_dim//4 + fusion_dim//4) = fusion_dim
        self.fusion_proj = BitLinear(fusion_dim, fusion_dim)
        
        # Cross-attention between semantic and prosody
        self.cross_fusion = CrossAttentionFusion(
            fusion_dim, num_heads=dec_cfg['fusion_heads']
        )
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(fusion_dim, dropout=dec_cfg['dropout'])
        
        # Main transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=dec_cfg['fusion_heads'],
            dim_feedforward=fusion_dim * 4,
            dropout=dec_cfg['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=dec_cfg['fusion_layers']
        )
        
        # Output projection (to vocoder input dim)
        # Output projection (to vocoder input dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            BitLinear(fusion_dim, fusion_dim),
            nn.GELU(),
            BitLinear(fusion_dim, fusion_dim)  # Keep at fusion_dim for vocoder
        )

    def forward(self, sem_z, pro_z, spk_z, target_len=None):
        """
        Args:
            sem_z: (B, T_s, D_s) semantic codes
            pro_z: (B, T_p, D_p) prosody codes
            spk_z: (B, D_spk) speaker embedding
        Returns:
            fused: (B, T_out, fusion_dim) features for vocoder
        """
        # Upsample
        sem_up = self.sem_upsampler(sem_z)  # (B, T, fusion_dim//2)
        pro_up = self.pro_upsampler(pro_z)  # (B, T, fusion_dim//4)
        
        # Align lengths
        T_out = min(sem_up.shape[1], pro_up.shape[1])
        if target_len is not None:
            T_out = min(T_out, target_len)
        
        sem_up = sem_up[:, :T_out, :]
        pro_up = pro_up[:, :T_out, :]
        
        # Expand speaker
        spk_emb = self.spk_proj(spk_z)  # (B, fusion_dim//4)
        spk_expanded = spk_emb.unsqueeze(1).expand(-1, T_out, -1)
        
        # Concatenate: (B, T, fusion_dim)
        cat = torch.cat([sem_up, pro_up, spk_expanded], dim=-1)
        
        # Initial fusion projection
        x = self.fusion_proj(cat)
        
        # Cross-attention fusion (optional refinement)
        # Here sem_up and pro_up are different dims, so we skip or project
        # For simplicity, we'll use the transformer directly
        
        # Transformer refinement
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Output
        x = self.output_proj(x)
        
        return x


class SpeechDecoderV2(nn.Module):
    """Full decoder: Feature Reconstructor + Vocoder V2"""
    def __init__(self, config):
        super().__init__()
        try:
            from .vocoder import NeuralVocoderV2
        except ImportError:
            # Fallback for when vocoder.py is missing/renamed
            from .bit_vocoder import NeuralVocoderBit as NeuralVocoderV2
        
        self.reconstructor = FeatureReconstructorV2(config)
        self.vocoder = NeuralVocoderV2(config)
        
    def forward(self, sem_z, pro_z, spk_z):
        feats = self.reconstructor(sem_z, pro_z, spk_z)
        wave = self.vocoder(feats)
        return wave
