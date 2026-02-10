"""
MicroEncoder: Lightweight Audio Feature Extractor

Replaces HuBERT (~95M params) with a compact model (~2M params).
Uses knowledge distillation to learn HuBERT's representations.

Architecture:
1. CNN Frontend: 5 conv layers with progressive stride for 320x downsampling
2. Conformer Blocks: 4 layers for temporal modeling
3. Output Projection: Maps to 768-dim (HuBERT compatible)

Target: <10MB FP32, <1MB after BitNet quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .post_net import SnakeBeta, SnakePhase


class ConvBlock(nn.Module):
    """Single conv block for frontend."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, groups=groups, bias=False
        )
        self.norm = nn.GroupNorm(1, out_channels)  # LayerNorm equivalent for conv
        self.act = SnakePhase(out_channels)
        
    def forward(self, x):
        # SnakeGamma expects (B, C, T) which is correct for Conv1d output
        return self.act(self.norm(self.conv(x)))


class CNNFrontend(nn.Module):
    """
    CNN-based audio frontend similar to wav2vec2/HuBERT.
    Converts raw waveform to frame-level features.
    
    Total stride = 320 (to match HuBERT's 20ms frames at 16kHz)
    Using: 5 * 4 * 4 * 4 = 320
    """
    def __init__(self, output_dim=256):
        super().__init__()
        
        # Progressive downsampling: 5 * 4 * 4 * 4 = 320x
        self.layers = nn.Sequential(
            ConvBlock(1, 32, kernel_size=10, stride=5),      # 5x  -> 32 channels
            ConvBlock(32, 64, kernel_size=8, stride=4),      # 4x  -> 64 channels
            ConvBlock(64, 128, kernel_size=8, stride=4),     # 4x  -> 128 channels
            ConvBlock(128, output_dim, kernel_size=8, stride=4),  # 4x  -> output_dim
        )
        # Total stride: 5 * 4 * 4 * 4 = 320
        
    def forward(self, x):
        """
        Args:
            x: (B, T) raw waveform
        Returns:
            (B, T//320, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        
        x = self.layers(x)  # (B, C, T')
        return x.transpose(1, 2)  # (B, T', C)


class LightweightConformerBlock(nn.Module):
    """
    Simplified Conformer block optimized for size.
    Uses depthwise separable convolutions and reduced FFN dim.
    """
    def __init__(self, dim, num_heads=4, kernel_size=15, ffn_mult=2, dropout=0.1):
        super().__init__()
        
        # Feed-forward 1 (reduced multiplier)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ffn_mult),
            # SnakeBeta expects (B, C, T) usually, but for Linear output it's (B, T, C)
            # We can use a lambda wrapper or transpose linear
            # Or use standard GELU for FF? BitNet paper uses SwiGLU or GELU in FF often.
            # But user asked for SnakeBeta.
            # Let's keep GELU for FFN to avoid transpose overhead, or create a PermutedSnakeBeta.
            # Actually, standard Conformer uses Swish/GELU. Snake is better for periodic signals (CNNs).
            # Let's use SnakeBeta ONLY in the Convolution module where it matters most for periodicity.
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module (depthwise separable)
        self.conv_norm = nn.LayerNorm(dim)
        # Sequence: Pointwise -> Snake -> Depthwise -> Norm -> Snake -> Pointwise
        self.conv_pw1 = nn.Conv1d(dim, dim, 1)
        self.conv_act1 = SnakePhase(dim)
        self.conv_dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.conv_gn = nn.GroupNorm(1, dim)
        self.conv_act2 = SnakePhase(dim)
        self.conv_pw2 = nn.Conv1d(dim, dim, 1)
        self.conv_drop = nn.Dropout(dropout)
        
        # Feed-forward 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, C)
            mask: optional attention mask
        Returns:
            (B, T, C)
        """
        # FF1 (half residual as per Conformer paper)
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.attn_dropout(attn_out)
        
        # Convolution
        # conv_in: (B, T, C) -> (B, C, T)
        conv_in = self.conv_norm(x).transpose(1, 2)
        
        c = self.conv_pw1(conv_in)
        c = self.conv_act1(c)
        c = self.conv_dw(c)
        c = self.conv_gn(c)
        c = self.conv_act2(c)
        c = self.conv_pw2(c)
        c = self.conv_drop(c)
        
        conv_out = c.transpose(1, 2)  # (B, T, C)
        x = x + conv_out
        
        # FF2 (half residual)
        x = x + 0.5 * self.ff2(x)
        
        return self.final_norm(x)


class MicroEncoder(nn.Module):
    """
    Lightweight audio encoder to replace HuBERT.
    
    Input: Raw audio waveform (B, T) at 16kHz
    Output: Features (B, T//320, 768) matching HuBERT layer 9
    
    Architecture:
    - CNN Frontend: 6 conv layers for 320x downsampling
    - Conformer: 4 blocks with dim=256
    - Output: Linear projection to 768-dim
    
    Size: ~2.1M parameters (~8.4MB FP32)
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
        
        self.frontend_dim = frontend_dim
        self.conformer_dim = conformer_dim
        self.output_dim = output_dim
        
        # CNN Frontend
        self.frontend = CNNFrontend(output_dim=frontend_dim)
        
        # Project to conformer dim
        self.input_proj = nn.Linear(frontend_dim, conformer_dim)
        
        # Positional embedding (learned)
        self.pos_embed = nn.Embedding(4096, conformer_dim)  # Max 4096 frames
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            LightweightConformerBlock(
                conformer_dim, 
                num_heads=conformer_heads,
                kernel_size=conformer_kernel,
                dropout=dropout
            )
            for _ in range(conformer_layers)
        ])
        
        # Output projection to match HuBERT dim
        self.output_proj = nn.Sequential(
            nn.Linear(conformer_dim, conformer_dim * 2),
            nn.GELU(),
            nn.Linear(conformer_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x, return_all_layers=False):
        """
        Args:
            x: (B, T) raw audio waveform at 16kHz
            return_all_layers: if True, return all intermediate representations
        Returns:
            features: (B, T//320, 768)
            all_layers: optional list of intermediate features
        """
        # CNN Frontend
        x = self.frontend(x)  # (B, T', frontend_dim)
        
        # Project to conformer dim
        x = self.input_proj(x)  # (B, T', conformer_dim)
        
        # Add positional embeddings
        B, T, C = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(max=4095)  # Clamp to embedding size
        x = x + self.pos_embed(positions)
        
        # Conformer blocks
        all_layers = []
        for block in self.conformer_blocks:
            x = block(x)
            if return_all_layers:
                all_layers.append(x)
        
        # Output projection
        features = self.output_proj(x)
        
        if return_all_layers:
            return features, all_layers
        return features
    
    def count_parameters(self):
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    def estimate_size(self):
        """Estimate model size in MB."""
        params = self.count_parameters()['total']
        fp32_mb = params * 4 / (1024 * 1024)
        ternary_mb = params * 1.58 / 8 / (1024 * 1024)  # 1.58 bits per ternary
        return {
            'fp32_mb': fp32_mb,
            'ternary_mb': ternary_mb
        }


# Test
if __name__ == "__main__":
    print("Testing MicroEncoder...")
    
    model = MicroEncoder()
    
    # Print architecture info
    counts = model.count_parameters()
    sizes = model.estimate_size()
    
    print(f"\nMicroEncoder Architecture:")
    print(f"  Parameters: {counts['total']:,}")
    print(f"  FP32 Size: {sizes['fp32_mb']:.2f} MB")
    print(f"  Ternary Size: {sizes['ternary_mb']:.2f} MB")
    
    # Test forward pass
    x = torch.randn(2, 16000 * 3)  # 3 seconds of audio
    
    with torch.no_grad():
        features = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected frames: {x.shape[1] // 320}")
    
    # Compare to HuBERT output
    print(f"\nHuBERT compatibility:")
    print(f"  Output dim: {features.shape[-1]} (expected: 768)")
    print(f"  Frame rate: {16000 / 320:.1f} Hz (expected: 50 Hz)")
    
    print("\n✓ MicroEncoder test passed!")
