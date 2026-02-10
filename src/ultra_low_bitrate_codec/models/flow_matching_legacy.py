"""
Legacy ConditionalFlowMatching model that matches the checkpoints_flow_matching architecture.

This model was used for training checkpoints in checkpoints/checkpoints_flow_matching/
Key differences from current flow_matching.py:
- Uses Conv1d instead of Linear for projections
- hidden_dim = 256 (not 512)
- Output is Complex STFT (1026 = 513 real + 513 imag), not Mel80
- time_mlp is [256, 256 -> 256] instead of [hidden, hidden*4 -> hidden]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .bitlinear import BitLinear, RMSNorm

class BitConv1d(nn.Module):
    """Conv1d with BitLinear-style normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.norm = RMSNorm(in_channels)  # Norm over input channels
        self.padding = padding
        self.kernel_size = kernel_size
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        # x: (B, C_in, T)
        # Apply RMSNorm on channel dim
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, C, T)
        return F.conv1d(x, self.weight, padding=self.padding)


class BitFeedForwardLegacy(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            BitLinear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            BitLinear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class AdaLNLegacy(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = BitLinear(cond_dim, dim * 2)
        nn.init.zeros_(self.proj.weight)
        if hasattr(self.proj, 'bias') and self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, c):
        if c.dim() == 2:
            c = c.unsqueeze(1)
        
        emb = self.proj(c)
        scale, shift = torch.chunk(emb, 2, dim=-1)
        
        x = self.norm(x)
        return x * (1 + scale) + shift


class BitAttentionLegacy(nn.Module):
    """Multi-head attention with BitLinear qkv and proj"""
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = BitLinear(dim, dim * 3)
        self.proj = BitLinear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


class BitTransformerBlockLegacy(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, cond_dim, dropout=0.0):
        super().__init__()
        self.adaLN1 = AdaLNLegacy(dim, cond_dim)
        self.attn = BitAttentionLegacy(dim, num_heads, dropout)
        self.adaLN2 = AdaLNLegacy(dim, cond_dim)
        self.mlp = BitFeedForwardLegacy(dim, hidden_dim, dropout)
        
    def forward(self, x, c):
        xn = self.adaLN1(x, c)
        x = x + self.attn(xn)
        
        xn = self.adaLN2(x, c)
        x = x + self.mlp(xn)
        return x


class SinusoidalPosEmbedLegacy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalFlowMatchingLegacy(nn.Module):
    """
    Legacy Flow Matching model matching checkpoints_flow_matching architecture.
    
    Architecture:
    - hidden_dim = 256
    - Uses Conv1d for input_proj, cond_proj, final_proj  
    - Input/Output: 1026 channels (Complex STFT: 513 real + 513 imag)
    - Conditioning: 512 -> 256 via cond_proj
    - 6 transformer blocks with hidden_dim*4 FFN
    """
    def __init__(self, hidden_dim=256, cond_dim=512, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.out_dim = 1026  # Complex STFT
        self.cond_dim = cond_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmbedLegacy(hidden_dim),
            BitLinear(hidden_dim, hidden_dim),
            nn.GELU(),
            BitLinear(hidden_dim, hidden_dim)
        )
        
        # Input projection: (B, 1026, T) -> (B, 256, T)
        self.input_proj = BitConv1d(1026, hidden_dim, kernel_size=3)
        
        # Conditioning projection: (B, 512, T) -> (B, 256, T)
        self.cond_proj = BitConv1d(cond_dim, hidden_dim, kernel_size=3)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            BitTransformerBlockLegacy(
                dim=hidden_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim * 4,
                cond_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final norm and projection
        self.final_norm = RMSNorm(hidden_dim)
        self.final_proj = BitConv1d(hidden_dim, 1026, kernel_size=3)
        
    def forward(self, x_t, t, cond):
        """
        Args:
            x_t: (B, T, 1026) Noisy Complex STFT
            t: (B,) Time step [0, 1]
            cond: (B, T, 512) Conditioning features
        Returns:
            v_pred: (B, T, 1026) Predicted velocity
        """
        # x_t: (B, T, 1026) -> (B, 1026, T)
        x = x_t.transpose(1, 2)
        cond_conv = cond.transpose(1, 2)  # (B, 512, T)
        
        # Project input and conditioning
        x = self.input_proj(x)  # (B, 256, T)
        c = self.cond_proj(cond_conv)  # (B, 256, T)
        
        # Time embedding
        t_emb = self.time_mlp(t)  # (B, 256)
        
        # Switch to (B, T, D) for transformer
        x = x.transpose(1, 2)
        c = c.transpose(1, 2)
        c = c + t_emb.unsqueeze(1)  # Add time embedding
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_norm(x)
        
        # Project to output
        x = x.transpose(1, 2)  # (B, 256, T)
        v_pred = self.final_proj(x)  # (B, 1026, T)
        v_pred = v_pred.transpose(1, 2)  # (B, T, 1026)
        
        return v_pred
    
    @torch.no_grad()
    def solve_ode(self, cond, steps=10, solver='euler', cfg_scale=1.0):
        """Generate Complex STFT from conditioning"""
        B, T, _ = cond.shape
        device = cond.device
        
        x = torch.randn(B, T, self.out_dim, device=device)
        t_span = torch.linspace(0, 1, steps + 1, device=device)
        
        def get_velocity(x, t, c):
            if cfg_scale > 1.0:
                v_c = self(x, t, c)
                v_u = self(x, t, torch.zeros_like(c))
                return v_u + cfg_scale * (v_c - v_u)
            return self(x, t, c)
        
        for i in range(steps):
            t_curr = t_span[i]
            dt = t_span[i+1] - t_curr
            t_batch = torch.ones(B, device=device) * t_curr
            
            if solver == 'midpoint':
                v = get_velocity(x, t_batch, cond)
                x_mid = x + v * (dt / 2)
                t_mid = t_batch + (dt / 2)
                v_mid = get_velocity(x_mid, t_mid, cond)
                x = x + v_mid * dt
            else:
                v_pred = get_velocity(x, t_batch, cond)
                x = x + v_pred * dt
                
        return x
