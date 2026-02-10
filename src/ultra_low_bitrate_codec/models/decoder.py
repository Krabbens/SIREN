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

from .bitlinear import BitLinear, RMSNorm, BitSnakeBeta

class BitTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # BitNet style FF with BitSnakeBeta
        self.ff = nn.Sequential(
            BitLinear(d_model, dim_feedforward),
            BitSnakeBeta(dim_feedforward),
            nn.Dropout(dropout),
            BitLinear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-norm architecture
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + src2
        
        src2 = self.norm2(src)
        src = src + self.ff(src2)
        return src

class BitTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        # Extract params from the prototype layer (likely a standard TransformerEncoderLayer if passed from config)
        if hasattr(layer, 'self_attn'):
             d_model = layer.self_attn.embed_dim
             nhead = layer.self_attn.num_heads
             dim_feedforward = layer.linear1.out_features if hasattr(layer, 'linear1') else d_model * 4
             dropout = layer.dropout.p if hasattr(layer, 'dropout') else 0.1
        else:
             # Fallback if it's already a config or something else
             d_model = 512
             nhead = 8
             dim_feedforward = 2048
             dropout = 0.1

        self.layers = nn.ModuleList([
            BitTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output



class LearnableUpsampler(nn.Module):
    """
    Learnable upsampler with residual refinement.
    Uses transposed conv + refinement conv.
    """
    def __init__(self, input_dim, output_dim, factor):
        super().__init__()
        self.factor = factor
        
        from .bitlinear import BitConvTranspose1d, BitConv1d, RMSNorm
        
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
            BitSnakeBeta(output_dim),
            BitConv1d(output_dim, output_dim, kernel_size=7, padding=3)
        )
        
        self.norm = RMSNorm(output_dim)
        
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
    def __init__(self, sem_dim, pro_dim, num_heads=8, dropout=0.1):
        super().__init__()
        print(f"DEBUG CROSS_FUSION INIT: sem={sem_dim}, pro={pro_dim}")
        self.norm_sem = RMSNorm(sem_dim)
        self.norm_pro = RMSNorm(pro_dim)
        
        # Semantic attends to prosody
        self.cross_attn = nn.MultiheadAttention(
            sem_dim, num_heads, dropout=dropout, batch_first=True,
            kdim=pro_dim, vdim=pro_dim
        )
        
        self.ff = nn.Sequential(
            RMSNorm(sem_dim),
            BitLinear(sem_dim, sem_dim * 4),
            BitSnakeBeta(sem_dim * 4),
            nn.Dropout(dropout),
            BitLinear(sem_dim * 4, sem_dim),
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
        
        # Branch Dimensions (Hybrid)
        print(f"DEBUG DECODER CONFIG: {dec_cfg.keys()}")
        sem_dim = dec_cfg.get('sem_dim', fusion_dim)
        pro_dim = dec_cfg.get('pro_dim', fusion_dim)
        spk_dim = dec_cfg.get('spk_dim', fusion_dim)
        print(f"DEBUG DIMS: sem={sem_dim}, pro={pro_dim}, spk={spk_dim}, fusion={fusion_dim}")
        
        # ========================================
        # UPSAMPLERS
        # ========================================
        # ========================================
        # UPSAMPLERS (Target Spcific Dims)
        # ========================================
        self.sem_upsampler = LearnableUpsampler(
            sem_cfg['output_dim'], sem_dim,
            factor=sem_cfg['temporal_compression']
        )
        
        self.pro_upsampler = LearnableUpsampler(
            pro_cfg['output_dim'], pro_dim,
            factor=pro_cfg['temporal_compression']
        )
        
        from .bitlinear import BitLinear
        
        # Speaker projection -> spk_dim
        self.spk_proj = nn.Sequential(
            BitLinear(spk_cfg['embedding_dim'], spk_dim),
            BitSnakeBeta(spk_dim),
            BitLinear(spk_dim, spk_dim)
        )
        
        # ========================================
        # FUSION
        # ========================================
        # Concat Fusion: sem + pro + spk -> fusion_dim
        # Projection from concat to fusion base
        concat_dim = sem_dim + pro_dim + spk_dim
        self.fusion_proj = BitLinear(concat_dim, fusion_dim)
        
        # Cross-attention refinement between semantic and prosody
        self.cross_fusion = CrossAttentionFusion(
            sem_dim, pro_dim,
            num_heads=dec_cfg['fusion_heads']
        )
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, fusion_dim))
        self.dropout = nn.Dropout(dec_cfg['dropout'])
        
        # Main transformer (BitNet)
        prototype_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=dec_cfg['fusion_heads'],
            dim_feedforward=fusion_dim * 4,
            dropout=dec_cfg['dropout'],
            batch_first=True
        )
        self.transformer = BitTransformerEncoder(
            prototype_layer, 
            num_layers=dec_cfg['fusion_layers']
        )
        
        # Output projection (BitNet)
        self.output_proj = nn.Sequential(
            RMSNorm(fusion_dim),
            BitLinear(fusion_dim, fusion_dim),
            BitSnakeBeta(fusion_dim),
            BitLinear(fusion_dim, fusion_dim)
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
        sem_up = self.sem_upsampler(sem_z)  # (B, T, sem_dim)
        pro_up = self.pro_upsampler(pro_z)  # (B, T, pro_dim)
        
        # Align lengths
        T_out = min(sem_up.shape[1], pro_up.shape[1])
        if target_len is not None:
            T_out = min(T_out, target_len)
        
        sem_up = sem_up[:, :T_out, :]
        pro_up = pro_up[:, :T_out, :]
        
        # Cross-attention fusion (refinement) - sem attends to pro
        # Note: In hybrid mode, cross_fusion handles different dims
        sem_up = self.cross_fusion(sem_up, pro_up)
        
        # Expand speaker to match spk_proj output dim (not fusion dim)
        spk_emb = self.spk_proj(spk_z)  # (B, spk_dim)
        spk_expanded = spk_emb.unsqueeze(1).expand(-1, T_out, -1)
        
        # Concatenation Fusion
        # sem(256) + pro(128) + spk(128) -> 512
        x = torch.cat([sem_up, pro_up, spk_expanded], dim=-1)
        
        # Initial fusion projection
        x = self.fusion_proj(x)
        
        # Transformer refinement
        x = x + self.pos_encoder[:, :T_out, :]
        x = self.dropout(x)
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
