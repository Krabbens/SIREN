import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitConv1d, BitConvTranspose1d, RMSNorm

class ResBlockBitNet(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        # BitNet is used in the core residual blocks
        self.conv1 = BitConv1d(channels, channels, kernel_size, 
                              stride=1, padding=(kernel_size * dilation - dilation) // 2, 
                              dilation=dilation)
        self.conv2 = BitConv1d(channels, channels, kernel_size, 
                              stride=1, padding=kernel_size // 2, 
                              dilation=1)
        self.norm = RMSNorm(channels)

    def forward(self, x):
        residual = x
        # RMSNorm expects channel last: (B, T, C)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        x = self.conv1(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual

class MelVocoderBitNet(nn.Module):
    """
    Hybrid BitNet Vocoder:
    - Standard Conv1d at Input/Output for numerical stability.
    - BitNet layers in the upsampling and residual blocks for efficiency.
    """
    def __init__(self, n_mels=80, upsample_factors=[5, 4, 4, 4], channels=512):
        super().__init__()
        # 1. Input: Standard Conv1d (High precision for sensitive Mel features)
        self.input_conv = nn.Conv1d(n_mels, channels, kernel_size=7, padding=3)
        
        self.upsamplers = nn.ModuleList()
        curr_channels = channels
        
        for f in upsample_factors:
            # 2. Upsampling: BitNet Transposed Conv
            self.upsamplers.append(nn.Sequential(
                BitConvTranspose1d(curr_channels, curr_channels // 2, 
                                   kernel_size=f*2, stride=f, padding=f//2),
                ResBlockBitNet(curr_channels // 2, kernel_size=3, dilation=1),
                ResBlockBitNet(curr_channels // 2, kernel_size=3, dilation=3)
            ))
            curr_channels = curr_channels // 2
            
        # 3. Output: Standard Conv1d (High precision for waveform reconstruction)
        self.output_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(curr_channels, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, mel):
        # mel: (B, T, 80)
        x = mel.transpose(1, 2) # (B, 80, T)
        x = self.input_conv(x)
        
        for up in self.upsamplers:
            x = up(x)
            
        x = self.output_conv(x)
        return x.squeeze(1) # (B, T_audio)
