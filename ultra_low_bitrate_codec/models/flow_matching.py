import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ultra_low_bitrate_codec.models.bitlinear import BitLinear, BitConv1d, RMSNorm

class SinusoidalPosEmb(nn.Module):
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

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization for DiT.
    Regresses scale and shift parameters from the time embedding.
    """
    def __init__(self, dim, time_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        # SENSITIVE: Scale/Shift prediction needs high precision (FP32 or Int8, not 1.58b)
        self.proj = nn.Linear(time_dim, 2 * dim)
        
        # Zero initialization for the projection (standard DiT practice)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, t_emb):
        # x: (B, T, C)
        # t_emb: (B, time_dim)
        
        # Regress shift and scale
        # (B, time_dim) -> (B, 2*C)
        emb = self.proj(t_emb)
        
        # Reshape for broadcasting over T: (B, 1, 2*C)
        emb = emb.unsqueeze(1)
        
        scale, shift = torch.chunk(emb, 2, dim=-1)
        
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x

class BitFeedForward(nn.Module):
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

class BitAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = BitLinear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = BitLinear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BitDiTBlock(nn.Module):
    """
    Diffusion Transformer Block using BitLinear layers.
    """
    def __init__(self, dim, num_heads, time_dim, mlp_ratio=4.0):
        super().__init__()
        self.adaLN1 = AdaLN(dim, time_dim)
        self.attn = BitAttention(dim, num_heads=num_heads)
        self.adaLN2 = AdaLN(dim, time_dim)
        self.mlp = BitFeedForward(dim, int(dim * mlp_ratio))

    def forward(self, x, t_emb):
        # x: (B, T, C)
        # t_emb: (B, time_dim)
        
        x = x + self.attn(self.adaLN1(x, t_emb))
        x = x + self.mlp(self.adaLN2(x, t_emb))
        return x

class FlowMatchingHead(nn.Module):
    def __init__(
        self, 
        in_channels=1026,    # Complex STFT (513 Real + 513 Imag)
        cond_channels=512,   # BitNet feature dimension
        hidden_dim=256,
        depth=6,
        heads=8,
        time_dim=256
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Time Embedding (High Precision)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Projections (High Precision for continuous inputs)
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.cond_proj = nn.Conv1d(cond_channels, hidden_dim, kernel_size=3, padding=1)
        
        # Transformer Backbone (BitNet Core)
        self.blocks = nn.ModuleList([
            BitDiTBlock(hidden_dim, heads, time_dim)
            for _ in range(depth)
        ])
        
        # Output Head (High Precision for continuous outputs)
        self.final_norm = RMSNorm(hidden_dim)
        self.final_proj = nn.Conv1d(hidden_dim, in_channels, kernel_size=3, padding=1)
        
        # Initialize final projection to zero (standard Flow/Diffusion practice)
        nn.init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            nn.init.zeros_(self.final_proj.bias)

    def forward(self, x, t, cond):
        """
        Args:
            x: Noisy input (B, MelDim, T)
            t: Time steps (B,)
            cond: Conditioning features (B, CondDim, T) - must measure same T as x
        Returns:
            v: Vector field prediction (B, MelDim, T)
        """
        # Embed time
        t_emb = self.time_mlp(t) # (B, time_dim)
        
        # Project inputs to hidden dim
        # (B, MelDim, T) -> (B, HiddenDim, T)
        h_x = self.input_proj(x)
        h_cond = self.cond_proj(cond)
        
        # Combine input and condition
        h = h_x + h_cond
        
        # Transpose for Transformer: (B, HiddenDim, T) -> (B, T, HiddenDim)
        h = h.transpose(1, 2)
        
        # Apply DiT Blocks
        for block in self.blocks:
            h = block(h, t_emb)
            
        # Output projection
        h = self.final_norm(h)
        h = h.transpose(1, 2) # (B, HiddenDim, T) -> (B, T, HiddenDim) -> (B, HiddenDim, T)
        
        out = self.final_proj(h) # (B, MelDim, T)
        
        return out
