"""
BitNet Vocoder with SnakeBeta Activations

Ultra-compact vocoder (~1 MB) using:
- Ternary weights {-1, 0, +1} from BitNet 1.58b
- SnakeBeta periodic activations for audio quality
- iSTFT-based synthesis

Target: RPi Zero 2W deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .bitlinear import BitConv1d, BitLinear, RMSNorm
from .post_net import SnakeBeta


class BitConvNeXtBlock(nn.Module):
    """
    ConvNeXt block with BitNet quantization and SnakeBeta activation.
    
    Architecture:
        Depthwise Conv (BitConv1d) → RMSNorm → SnakeBeta
        → Pointwise expand (BitLinear) → SnakeBeta
        → Pointwise project (BitLinear) → LayerScale → Residual
    """
    def __init__(self, dim: int, expand_ratio: int = 4, dilation: int = 1, 
                 layer_scale_init: float = 1e-6):
        super().__init__()
        intermediate_dim = dim * expand_ratio
        padding = (7 + (dilation - 1) * 6) // 2
        
        # Depthwise conv (groups=dim, so not really quantizable efficiently)
        # Using standard conv for depthwise, BitLinear for pointwise
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding,
                                groups=dim, dilation=dilation)
        
        # RMSNorm instead of LayerNorm
        self.norm = RMSNorm(dim)
        
        # Pointwise convolutions as BitLinear
        self.pwconv1 = BitLinear(dim, intermediate_dim)
        self.act1 = SnakeBeta(intermediate_dim)
        
        self.pwconv2 = BitLinear(intermediate_dim, dim)
        
        # Layer scale
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) \
            if layer_scale_init > 0 else None
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T)
        """
        residual = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute for linear layers: (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)
        x = self.norm(x)
        
        # Pointwise expand with SnakeBeta
        x = self.pwconv1(x)
        x = x.transpose(1, 2)  # (B, C, T) for SnakeBeta
        x = self.act1(x)
        x = x.transpose(1, 2)  # (B, T, C) for BitLinear
        
        # Pointwise project
        x = self.pwconv2(x)
        
        # Layer scale
        if self.gamma is not None:
            x = self.gamma * x
        
        # Permute back: (B, T, C) → (B, C, T)
        x = x.transpose(1, 2)
        
        return residual + x


class BitResidualBlock(nn.Module):
    """
    Residual block with BitConv1d and SnakeBeta.
    Captures multi-scale harmonic structure.
    """
    def __init__(self, channels: int, kernel_size: int = 3, 
                 dilations: tuple = (1, 3, 5)):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        for d in dilations:
            padding = d * (kernel_size - 1) // 2
            self.blocks.append(nn.Sequential(
                BitConv1d(channels, channels, kernel_size, padding=padding, dilation=d),
                SnakeBeta(channels),
                BitConv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2),
            ))
    
    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class BitInstantaneousFrequencyHead(nn.Module):
    """
    Phase prediction via instantaneous frequency with BitLinear.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = BitLinear(input_dim, output_dim)
        self.cumsum_weight = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        # x: (B, T, C)
        inst_freq = self.proj(x)
        inst_freq = torch.tanh(inst_freq) * np.pi
        
        # Integrate to get phase
        phase = torch.cumsum(inst_freq * self.cumsum_weight, dim=1)
        phase = torch.remainder(phase + np.pi, 2 * np.pi) - np.pi
        
        return phase


class BitVocoder(nn.Module):
    """
    Ultra-compact BitNet Vocoder.
    
    Architecture (dim=256, layers=4):
        - Input projection: BitConv1d
        - 4x BitConvNeXtBlock with SnakeBeta
        - 1x BitResidualBlock for harmonics
        - Magnitude + Phase heads (BitLinear)
        - iSTFT synthesis
    
    Size: ~1 MB (ternary weights)
    Target: Real-time on RPi Zero 2W
    """
    def __init__(self, 
                 input_dim: int = 256,
                 dim: int = 256,
                 n_fft: int = 1024,
                 hop_length: int = 320,
                 num_layers: int = 4,
                 num_res_blocks: int = 1):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.input_dim = input_dim
        self.dim = dim
        
        # Input projection
        self.conv_in = BitConv1d(input_dim, dim, kernel_size=7, padding=3)
        self.act_in = SnakeBeta(dim)
        
        # ConvNeXt backbone
        self.backbone = nn.ModuleList()
        dilations = [1, 2, 4, 8][:num_layers]
        for d in dilations:
            self.backbone.append(BitConvNeXtBlock(dim, expand_ratio=4, dilation=d))
        
        # Residual blocks for harmonic coherence
        self.res_blocks = nn.ModuleList([
            BitResidualBlock(dim, kernel_size=3, dilations=(1, 3, 5))
            for _ in range(num_res_blocks)
        ])
        
        # Skip connection
        self.skip_proj = BitConv1d(input_dim, dim, kernel_size=1)
        
        # Final norm
        self.norm = RMSNorm(dim)
        
        # Output heads
        self.out_dim = n_fft // 2 + 1
        
        # Magnitude head
        self.mag_head = nn.Sequential(
            BitLinear(dim, dim),
            SnakeBeta(dim),
            BitLinear(dim, self.out_dim)
        )
        
        # Phase head (IF-based)
        self.phase_head = BitInstantaneousFrequencyHead(dim, self.out_dim)
        
        # Window for iSTFT
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C) or (B, C, T)
        Returns:
            audio: (B, T_audio)
        """
        # Handle input format
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[2] == self.input_dim:
            x = x.transpose(1, 2)  # (B, C, T)
        
        # Skip connection
        skip = self.skip_proj(x)
        
        # Input projection
        x = self.conv_in(x)
        x = self.act_in(x)
        
        # Backbone
        for blk in self.backbone:
            x = blk(x)
        
        # Add skip
        min_len = min(x.shape[2], skip.shape[2])
        x = x[..., :min_len] + skip[..., :min_len]
        
        # Residual blocks
        for res in self.res_blocks:
            x = res(x)
        
        # Final processing: (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)
        x = self.norm(x)
        
        # Magnitude (log-scale)
        log_mag = self.mag_head[0](x)
        log_mag = log_mag.transpose(1, 2)
        log_mag = self.mag_head[1](log_mag)
        log_mag = log_mag.transpose(1, 2)
        log_mag = self.mag_head[2](log_mag)
        mag = torch.exp(torch.clamp(log_mag, min=-10, max=10))
        
        # Phase
        phase = self.phase_head(x)
        
        # Construct complex spectrum
        mag = mag.float()
        phase = phase.float()
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        spec = torch.complex(real, imag)
        
        # iSTFT
        spec = spec.transpose(1, 2)  # (B, F, T)
        
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True
        )
        
        return audio
    
    def count_parameters(self):
        """Count total and quantizable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Quantizable params (BitLinear/BitConv1d weights)
        quantizable = 0
        for name, module in self.named_modules():
            if isinstance(module, (BitLinear, BitConv1d)):
                quantizable += module.weight.numel()
        
        return {
            'total': total,
            'trainable': trainable,
            'quantizable': quantizable,
            'non_quantizable': total - quantizable
        }
    
    def estimate_size(self):
        """Estimate model size in different formats."""
        counts = self.count_parameters()
        
        # Ternary: 1.58 bits per weight
        # Non-quantizable (norms, SnakeBeta): FP32
        ternary_bits = counts['quantizable'] * 1.58
        fp32_bits = counts['non_quantizable'] * 32
        
        total_bits = ternary_bits + fp32_bits
        size_mb = total_bits / 8 / 1024 / 1024
        
        return {
            'fp32_mb': counts['total'] * 4 / 1024 / 1024,
            'ternary_mb': size_mb,
            'compression_ratio': (counts['total'] * 4 / 1024 / 1024) / size_mb
        }


class NeuralVocoderBit(nn.Module):
    """Wrapper matching NeuralVocoderV2 interface."""
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['model']['decoder'].get('fusion_dim', 256)
        
        bit_config = config['model'].get('bit_vocoder', {})
        
        self.model = BitVocoder(
            input_dim=self.input_dim,
            dim=bit_config.get('dim', 256),
            n_fft=1024,
            hop_length=320,
            num_layers=bit_config.get('num_layers', 4),
            num_res_blocks=bit_config.get('num_res_blocks', 1)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return self.model(x)


# Test
if __name__ == "__main__":
    print("Testing BitVocoder...")
    
    # Create model
    vocoder = BitVocoder(
        input_dim=256,
        dim=256,
        num_layers=4,
        num_res_blocks=1
    )
    
    # Test forward pass
    x = torch.randn(2, 256, 100)  # (B, C, T)
    audio = vocoder(x)
    print(f"Input: {x.shape}")
    print(f"Output audio: {audio.shape}")
    
    # Count parameters
    counts = vocoder.count_parameters()
    print(f"\nParameters:")
    print(f"  Total: {counts['total']:,}")
    print(f"  Quantizable: {counts['quantizable']:,}")
    print(f"  Non-quantizable: {counts['non_quantizable']:,}")
    
    # Estimate size
    sizes = vocoder.estimate_size()
    print(f"\nModel Size:")
    print(f"  FP32: {sizes['fp32_mb']:.2f} MB")
    print(f"  Ternary: {sizes['ternary_mb']:.2f} MB")
    print(f"  Compression: {sizes['compression_ratio']:.1f}x")
    
    print("\n✓ BitVocoder test passed!")
