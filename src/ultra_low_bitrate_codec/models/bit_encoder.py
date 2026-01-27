"""
BitEncoder: 1.58-bit Ternary Audio Encoder

BitNet version of MicroEncoder using ternary weights {-1, 0, 1}.
Achieves extreme model compression (~20x smaller than FP32).

Architecture:
1. BitConv1d Frontend: Ternary convolutions
2. BitConformer: Conformer blocks with BitLinear
3. Output Projection: FP32 for final regression (standard practice)
"""

import torch
import torch.nn as nn
import math
from .bitlinear import BitLinear, BitConv1d
from .post_net import SnakeBeta, SnakePhase
from .micro_encoder import CNNFrontend, LightweightConformerBlock

# We need a custom frontend and conformer block using Bit layers

class BitConvBlock(nn.Module):
    """Single ternary conv block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = BitConv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, groups=groups, bias=False
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = SnakePhase(out_channels)
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BitCNNFrontend(nn.Module):
    """
    Ternary CNN frontend.
    """
    def __init__(self, output_dim=256):
        super().__init__()
        
        self.layers = nn.Sequential(
            BitConvBlock(1, 32, kernel_size=10, stride=5),      # 5x
            BitConvBlock(32, 64, kernel_size=8, stride=4),      # 4x
            BitConvBlock(64, 128, kernel_size=8, stride=4),     # 4x
            BitConvBlock(128, output_dim, kernel_size=8, stride=4),  # 4x
        )
        # Total stride: 320
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.layers(x)
        return x.transpose(1, 2)


class BitConformerBlock(nn.Module):
    """
    Conformer block using BitLinear and BitConv1d.
    """
    def __init__(self, dim, num_heads=4, kernel_size=15, ffn_mult=2, dropout=0.1):
        super().__init__()
        
        # Feed-forward 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            BitLinear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            BitLinear(dim * ffn_mult, dim),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        
        # Ternary Convolution module
        self.conv_norm = nn.LayerNorm(dim)
        # Sequence: Pointwise -> Snake -> Depthwise -> Norm -> Snake -> Pointwise
        self.conv_pw1 = BitConv1d(dim, dim, 1)
        self.conv_act1 = SnakePhase(dim)
        
        # Depthwise
        self.conv_dw = BitConv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.conv_gn = nn.GroupNorm(1, dim)
        self.conv_act2 = SnakePhase(dim)
        
        self.conv_pw2 = BitConv1d(dim, dim, 1)
        self.conv_drop = nn.Dropout(dropout)
        
        # Feed-forward 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            BitLinear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            BitLinear(dim * ffn_mult, dim),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # BitNet Note: We keep attention and norms in full precision usually
        # FF1
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.attn_dropout(attn_out)
        
        # Convolution
        # Input x: (B, T, C)
        conv_in = self.conv_norm(x).transpose(1, 2)  # (B, C, T)
        
        c = self.conv_pw1(conv_in)
        c = self.conv_act1(c)
        c = self.conv_dw(c)
        c = self.conv_gn(c)
        c = self.conv_act2(c)
        c = self.conv_pw2(c)
        c = self.conv_drop(c)
        
        conv_out = c.transpose(1, 2)  # (B, T, C)
        x = x + conv_out
        
        # FF2
        x = x + 0.5 * self.ff2(x)
        
        return self.final_norm(x)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class BitEncoder(nn.Module):
    """
    Ternary MicroEncoder.
    
    Target size: ~0.7MB (ternary)
    """
    def __init__(
        self,
        frontend_dim: int = 256,
        conformer_dim: int = 192,
        conformer_layers: int = 4,
        conformer_heads: int = 4,
        conformer_kernel: int = 15,
        output_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.frontend = BitCNNFrontend(output_dim=frontend_dim)
        
        # Input projection (BitLinear)
        self.input_proj = BitLinear(frontend_dim, conformer_dim)
        
        # Fixed sinusoidal embedding to save Parameters
        self.pos_embed = SinusoidalPositionalEmbedding(conformer_dim)
        
        self.conformer_blocks = nn.ModuleList([
            BitConformerBlock(
                conformer_dim, 
                num_heads=conformer_heads,
                kernel_size=conformer_kernel,
                dropout=dropout
            )
            for _ in range(conformer_layers)
        ])
        
        # Output projection (Keep FP32/standard for high precision target matching)
        # Using BitLinear for output head can degrade regression performance significantly
        # But for size, let's try BitLinear -> FP32 LayerNorm -> FP32 Linear (final)
        self.output_proj = nn.Sequential(
            BitLinear(conformer_dim, conformer_dim * 2),
            nn.GELU(),
            # Final layer could be BitLinear too, but keeping it standard for safety first
            BitLinear(conformer_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        # Frontend
        x = self.frontend(x)
        x = self.input_proj(x)
        
        # Pos Emb (Fixed)
        x = x + self.pos_embed(x)
        
        # Conformer
        for block in self.conformer_blocks:
            x = block(x)
            
        # Output
        return self.output_proj(x)
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {'total': total}
    
    def estimate_size(self):
        total = self.count_parameters()['total']
        # Rough estimation: Most are ternary, but norms/embeddings are FP32
        # Let's assume 90% ternary
        fp32_mb = total * 4 / 1024**2
        ternary_mb = (total * 0.9 * 1.58 / 8 + total * 0.1 * 4) / 1024**2
        return {'fp32_mb': fp32_mb, 'ternary_mb': ternary_mb}


if __name__ == "__main__":
    print("Testing BitEncoder...")
    model = BitEncoder()
    x = torch.randn(2, 48000)
    y = model(x)
    
    counts = model.count_parameters()
    sizes = model.estimate_size()
    
    print(f"Output shape: {y.shape}")
    print(f"Params: {counts['total']:,}")
    print(f"Est. Size: {sizes['ternary_mb']:.2f} MB")
