"""
MicroEncoder: Lightweight End-to-End Audio Encoder for Ultra-Low Bitrate Codec

Architecture:
- Conv1D Frontend (downsampling 320x): 7 strided convolutions
- BitNet Transformer: 4 layers, hidden=256
- Output: 768-dim features (compatible with existing Factorizer)

Total: ~2.5M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .bitlinear import BitLinear, RMSNorm
from .infinity_bottleneck import InfinityBottleneck


class ConvBlock(nn.Module):
    """Single conv block with normalization and activation"""
    def __init__(self, in_ch, out_ch, kernel_size, stride, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, groups=groups)
        self.norm = nn.GroupNorm(1, out_ch)  # LayerNorm equivalent for 1D
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvFrontend(nn.Module):
    """
    Convolutional frontend for raw audio.
    Downsamples 16kHz audio to ~50Hz frame rate (320x downsampling).
    
    Similar to wav2vec2/HuBERT frontend but smaller.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        # Progressive downsampling: 16000 / 320 = 50 Hz
        # Strides: 5 * 4 * 4 * 2 * 2 = 320
        self.layers = nn.Sequential(
            # Layer 1: 1 -> 32, stride 5 (3200 -> 640)
            ConvBlock(1, 32, kernel_size=10, stride=5),
            # Layer 2: 32 -> 64, stride 4 (640 -> 160)
            ConvBlock(32, 64, kernel_size=8, stride=4),
            # Layer 3: 64 -> 128, stride 4 (160 -> 40)
            ConvBlock(64, 128, kernel_size=8, stride=4),
            # Layer 4: 128 -> 256, stride 2 (40 -> 20)
            ConvBlock(128, 256, kernel_size=4, stride=2),
            # Layer 5: 256 -> 256, stride 2 (20 -> 10)
            ConvBlock(256, 256, kernel_size=4, stride=2),
            # Adjust to get ~50Hz: need one more layer or different strides
            # Actually 5*4*4*2*2 = 320, which gives us 16000/320 = 50Hz ✓
        )
        
        # Final projection
        self.proj = nn.Conv1d(256, out_dim, kernel_size=1)
        self.norm = RMSNorm(out_dim)
        
    def forward(self, x):
        # x: (B, T) raw audio
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        
        x = self.layers(x)  # (B, 256, T')
        x = self.proj(x)    # (B, out_dim, T')
        x = x.transpose(1, 2)  # (B, T', out_dim)
        x = self.norm(x)
        return x


class BitTransformerLayer(nn.Module):
    """Single Transformer layer with BitLinear"""
    def __init__(self, dim, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = RMSNorm(dim)
        self.ff = nn.Sequential(
            BitLinear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            BitLinear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # FFN
        x = x + self.ff(self.norm2(x))
        return x


class MicroEncoder(nn.Module):
    """
    Minimal end-to-end audio encoder.
    
    Input: Raw 16kHz audio (B, T)
    Output: Features (B, T', 768) at ~50Hz
    
    Architecture:
    1. Conv Frontend: Raw audio -> 256-dim features at 50Hz
    2. BitNet Transformer: 4 layers of self-attention
    3. Projection: 256 -> 768 (compatible with Factorizer)
    
    Total: ~2.5M parameters
    """
    def __init__(
        self, 
        hidden_dim=256,
        output_dim=768,
        num_layers=4,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 1. Conv Frontend
        self.frontend = ConvFrontend(out_dim=hidden_dim)
        
        # 2. Positional Encoding
        self.pos_enc = SinusoidalPosEnc(hidden_dim)
        
        # 3. Transformer Layers
        self.layers = nn.ModuleList([
            BitTransformerLayer(hidden_dim, num_heads, ff_mult=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Final Norm + Projection
        self.final_norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Infinity Bottleneck (Disabled by default)
        self.infinity_bottleneck = None
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def enable_infinity_bottleneck(self, downsample_factor=4, num_layers=2):
        """Standard method to enable compression via Infinity Attention"""
        print(f"  [MicroEncoder] Enabling Infinity Bottleneck (factor={downsample_factor}, layers={num_layers})")
        self.infinity_bottleneck = InfinityBottleneck(
            dim=self.hidden_dim, 
            num_layers=num_layers,
            downsample_factor=downsample_factor
        ).to(next(self.parameters()).device)
    
    def forward(self, x):
        """
        Args:
            x: (B, T) raw audio at 16kHz
        Returns:
            features: (B, T', 768) at ~50Hz
        """
        # Frontend
        x = self.frontend(x)  # (B, T', hidden_dim)
        
        # Add positional encoding
        x = self.pos_enc(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Infinity Bottleneck (Compression)
        if self.infinity_bottleneck is not None:
            x = self.infinity_bottleneck(x)
        
        # Output projection
        x = self.final_norm(x)
        x = self.output_proj(x)  # (B, T', 768)
        
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class SinusoidalPosEnc(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:x.size(1)]


# ============================================================================
# Smaller variant for extreme edge deployment
# ============================================================================

class MicroEncoderTiny(nn.Module):
    """
    Even smaller encoder for extreme edge cases.
    ~1M parameters.
    """
    def __init__(self, hidden_dim=128, output_dim=768, num_layers=2):
        super().__init__()
        
        # Simpler frontend
        self.frontend = nn.Sequential(
            ConvBlock(1, 32, kernel_size=10, stride=5),
            ConvBlock(32, 64, kernel_size=8, stride=4),
            ConvBlock(64, 128, kernel_size=8, stride=4),
            ConvBlock(128, hidden_dim, kernel_size=4, stride=4),  # 5*4*4*4 = 320
        )
        
        self.pos_enc = SinusoidalPosEnc(hidden_dim)
        
        self.layers = nn.ModuleList([
            BitTransformerLayer(hidden_dim, num_heads=2, ff_mult=2)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.frontend(x)
        x = x.transpose(1, 2)  # (B, T', D)
        x = self.pos_enc(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        x = self.proj(x)
        return x


if __name__ == "__main__":
    # Test
    model = MicroEncoder()
    print(f"MicroEncoder params: {model.get_num_params():,}")
    
    x = torch.randn(2, 16000)  # 1 second of audio
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    # Tiny version
    tiny = MicroEncoderTiny()
    print(f"MicroEncoderTiny params: {sum(p.numel() for p in tiny.parameters()):,}")
    out_tiny = tiny(x)
    print(f"Input: {x.shape} -> Output (Tiny): {out_tiny.shape}")
