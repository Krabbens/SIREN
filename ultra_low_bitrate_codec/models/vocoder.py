"""
Vocos-style Vocoder V2 - Improved architecture
Key improvements:
1. More layers (8 instead of 6)
2. Multi-scale ConvNeXt blocks with varying dilation
3. Improved phase prediction using Instantaneous Frequency
4. Skip connections for better gradient flow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvNeXtBlock(nn.Module):
    """Enhanced ConvNeXt block with optional dilation"""
    def __init__(self, dim, intermediate_dim, dilation=1, layer_scale_init_value=1e-6):
        super().__init__()
        padding = (7 + (dilation - 1) * 6) // 2  # Maintain receptive field
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, 
                                groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions for spectral coherence"""
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size, 
                             padding=d * (kernel_size - 1) // 2, dilation=d),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2),
                )
            )
    
    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x


class InstantaneousFrequencyHead(nn.Module):
    """
    Predict phase using instantaneous frequency (more stable than raw phase).
    IF = d(phase)/dt, which is smoother and easier to learn.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.cumsum_weight = nn.Parameter(torch.ones(1) * 0.1)  # Learnable integration rate
        
    def forward(self, x):
        # x: (B, T, C)
        inst_freq = self.proj(x)  # (B, T, F)
        # Wrap to [-pi, pi] range for IF
        inst_freq = torch.tanh(inst_freq) * np.pi
        
        # Integrate IF to get phase (cumulative sum along time)
        # phase[t] = sum(IF[0:t])
        phase = torch.cumsum(inst_freq * self.cumsum_weight, dim=1)
        
        # Wrap phase to [-pi, pi]
        phase = torch.remainder(phase + np.pi, 2 * np.pi) - np.pi
        
        return phase


class VocosGeneratorV2(nn.Module):
    """
    Improved Vocos-style generator.
    Predicts magnitude and phase via iSTFT.
    """
    def __init__(self, input_dim=256, dim=512, n_fft=1024, hop_length=256, 
                 num_convnext_layers=8, num_res_blocks=3):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.input_dim = input_dim
        
        # Initial projection
        self.pad = nn.ReflectionPad1d(3)
        self.conv_in = nn.Conv1d(input_dim, dim, kernel_size=7, padding=0)
        
        # Multi-scale ConvNeXt backbone with varying dilations
        self.backbone = nn.ModuleList()
        dilations = [1, 1, 2, 2, 4, 4, 8, 8][:num_convnext_layers]
        for d in dilations:
            self.backbone.append(ConvNeXtBlock(dim, dim * 4, dilation=d))
        
        # Residual blocks for harmonic structure ("vertical memory")
        self.res_blocks = nn.ModuleList([
            ResidualBlock(dim, kernel_size=3, dilations=(1, 3, 5))
            for _ in range(num_res_blocks)
        ])
        
        # Skip connection from input
        self.skip_proj = nn.Conv1d(input_dim, dim, kernel_size=1)
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Output heads
        self.out_dim = n_fft // 2 + 1
        
        # Magnitude prediction (log-scale for stability)
        self.mag_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.out_dim)
        )
        
        # Phase prediction using Instantaneous Frequency
        self.phase_head = InstantaneousFrequencyHead(dim, self.out_dim)
        
        # Learnable window
        self.register_buffer('window', torch.hann_window(n_fft))
        
    def forward(self, x):
        """
        x: (B, T, C) or (B, C, T)
        Returns: audio (B, T_audio)
        """
        # Handle input format
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[2] == self.input_dim:
            x = x.transpose(1, 2)  # (B, C, T)
        
        # Skip connection
        skip = self.skip_proj(x)
        
        # Initial conv
        x = self.pad(x)
        x = self.conv_in(x)
        
        # ConvNeXt backbone
        for blk in self.backbone:
            x = blk(x)
        
        # Add skip
        min_len = min(x.shape[2], skip.shape[2])
        x = x[..., :min_len] + skip[..., :min_len]
        
        # Residual blocks for harmonic coherence
        for res in self.res_blocks:
            x = res(x)
        
        # Normalize and predict
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        
        # Magnitude (log-scale prediction)
        log_mag = self.mag_head(x)
        mag = torch.exp(torch.clamp(log_mag, min=-10, max=10))  # Clamp for stability
        
        # Phase (via instantaneous frequency)
        phase = self.phase_head(x)
        
        # Construct complex spectrum (force float32 for stability)
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


class NeuralVocoderV2(nn.Module):
    """Wrapper for VocosGeneratorV2"""
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['model']['decoder'].get('fusion_dim', 256)
        
        # Vocos config - matching 50Hz feature rate
        # 16000 / 320 = 50 Hz
        self.model = VocosGeneratorV2(
            input_dim=self.input_dim,
            dim=512,               # Increased from 384
            n_fft=1024,
            hop_length=320,        # 50 Hz input rate
            num_convnext_layers=8, # Increased from 6
            num_res_blocks=3       # NEW: Residual blocks
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        audio = self.model(x)
        return audio
