"""
Feature Adapter to bridge MicroEncoder (768-dim) and BitVocoder (512-dim).
Uses SnakePhase activations to preserve phase information during projection.
Includes Conv upsampler with SnakeBeta for temporal refinement.
"""

import torch
import torch.nn as nn
from .post_net import SnakePhase, SnakeBeta

class FeatureAdapter(nn.Module):
    def __init__(self, in_dim=768, out_dim=512, hidden_dim=512, upsample_factor=2):
        super().__init__()
        
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.act = SnakePhase(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
        # Upsampler with ConvTranspose + SnakeBeta
        self.upsample_factor = upsample_factor
        if upsample_factor > 1:
            self.upsampler = nn.Sequential(
                # ConvTranspose for upsampling
                nn.ConvTranspose1d(
                    out_dim, out_dim, 
                    kernel_size=upsample_factor * 2,
                    stride=upsample_factor,
                    padding=upsample_factor // 2
                ),
                SnakeBeta(out_dim),
                # Refinement conv
                nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
                SnakeBeta(out_dim),
            )
        else:
            self.upsampler = None
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C) or (B, C, T)
        Returns:
            out: (B, T*upsample_factor, out_dim) or (B, out_dim, T*upsample_factor)
        """
        # We work in (B, T, C) for Linear, but (B, C, T) for Snake/Conv
        
        # Check input format
        input_was_channel_last = (x.shape[-1] == 768) # Assuming 768 is C
        if not input_was_channel_last:
            # (B, C, T) -> (B, T, C)
            x = x.transpose(1, 2)
            
        # 1. Linear in (B, T, C)
        x = self.in_proj(x)
        
        # 2. Snake in (B, C, T)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = x.transpose(1, 2)
        
        # 3. Linear out (B, T, C)
        x = self.out_proj(x)
        
        # 4. Upsampler in (B, C, T)
        if self.upsampler is not None:
            x = x.transpose(1, 2)  # (B, C, T)
            x = self.upsampler(x)
            x = x.transpose(1, 2)  # (B, T, C)
        
        # Restore format if needed
        if not input_was_channel_last:
            x = x.transpose(1, 2) # (B, C, T)
            
        return x

