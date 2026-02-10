#!/usr/bin/env python3
"""
MicroHuBERT: Ultra-Lightweight Speech Representation Model

Designed to replace DistilHuBERT (23MB) with a <5MB model.
- Architecture: 4-layer Transformer, 256 hidden dim
- Input: Raw Audio
- Output: 768-dim embeddings (projected from 256)
- Distillation: Trained to match HuBERT Layer 9

Size Comparison (FP32):
- HuBERT Base: 360 MB
- DistilHuBERT: 90 MB
- MicroHuBERT: ~14 MB (FP32) -> ~3.5 MB (INT8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroConvFeatureExtractor(nn.Module):
    """
    Tiny convolutional frontend.
    Matches HuBERT 320x downsampling with fewer parameters using Depthwise Separable Convs.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        
        def dw_conv(in_ch, out_ch, k, s):
            return nn.Sequential(
                # Depthwise
                nn.Conv1d(in_ch, in_ch, k, stride=s, groups=in_ch, bias=False),
                # Pointwise
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(1, out_ch), # LayerNorm equivalent for Conv
                nn.GELU()
            )

        self.convs = nn.Sequential(
            # Layer 1: 5x
            nn.Conv1d(1, 128, kernel_size=10, stride=5, bias=False),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            
            # Layers 2-5: 2x each (Total 16x from here -> 80x total)
            dw_conv(128, 192, 3, 2),
            dw_conv(192, 192, 3, 2),
            dw_conv(192, 192, 3, 2),
            dw_conv(192, 256, 3, 2),
            
            # Layers 6-7: 2x each (Total 4x from here -> 320x total)
            dw_conv(256, 256, 2, 2),
            dw_conv(256, out_dim, 2, 2),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.convs(x)

class MicroHuBERT(nn.Module):
    def __init__(
        self, 
        hidden_dim=256, 
        output_dim=768, 
        num_layers=4, 
        num_heads=4,
        ff_mult=3
    ):
        super().__init__()
        
        self.encoder = MicroConvFeatureExtractor(out_dim=hidden_dim)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, 1000, hidden_dim)) # Simple learned PE
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim) # Project to 768 for HuBERT compat
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, T_audio)
        
        # 1. Conv Features
        x = self.encoder(x) # (B, 256, T_frames)
        x = x.transpose(1, 2) # (B, T_frames, 256)
        
        # 2. Add Positional Embeddings
        seq_len = x.size(1)
        if seq_len > self.pos_emb.size(1):
             # Simple extrapolation or truncation
             pe = F.interpolate(self.pos_emb.transpose(1,2), size=seq_len).transpose(1,2)
        else:
             pe = self.pos_emb[:, :seq_len, :]
             
        x = x + pe
        
        # 3. Transformer
        x = self.transformer(x)
        x = self.final_norm(x)
        
        # 4. Project to HuBERT dim
        x = self.out_proj(x) # (B, T, 768)
        
        return x

if __name__ == "__main__":
    # San Force Check
    model = MicroHuBERT()
    print(f"MicroHuBERT Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    x = torch.randn(1, 16000)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    
    # Check size in MB
    param_size = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
    print(f"FP32 Size: {param_size:.2f} MB")
    print(f"INT8 Size: {param_size/4:.2f} MB")
