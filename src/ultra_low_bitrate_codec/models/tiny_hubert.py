import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitLinear, RMSNorm

class BitAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # QKV projections using BitLinear
        self.qkv = BitLinear(dim, dim * 3)
        self.out_proj = BitLinear(dim, dim)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class BitTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=8, ff_dim=1024):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = BitAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = nn.Sequential(
            BitLinear(dim, ff_dim),
            nn.GELU(),
            BitLinear(ff_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class TinyHubert(nn.Module):
    """
    An improved, deeper BitNet-based HuBERT emulator for edge devices.
    12 layers, 384 hidden dim, fully BitNet attention.
    """
    def __init__(self, out_dim=768, hidden_dim=384, num_layers=12):
        super().__init__()
        
        # 1. Conv Front-end (7-layer conv extractor similar to original HuBERT)
        # Matches 320x downsampling
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
        )
        # Total stride: 5 * 2 * 2 * 2 * 2 * 2 * 1 = 160. 
        # Wait, I need 320. Original HuBERT: (5,2,2,2,2,2,2) = 320.
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=10, stride=5, padding=3), # 5
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1), # 10
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1), # 20
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1), # 40
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1), # 80
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1), # 160
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, stride=2, padding=1), # 320
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )
        
        # 2. BitNet Transformer Blocks
        self.blocks = nn.ModuleList([
            BitTransformerLayer(hidden_dim, num_heads=12, ff_dim=hidden_dim*3)
            for _ in range(num_layers)
        ])
        
        # 3. Output Projection to match HuBERT (768)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.final_norm = RMSNorm(out_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.frontend(x)
        x = x.transpose(1, 2)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.out_proj(x)
        x = self.final_norm(x)
        return x
