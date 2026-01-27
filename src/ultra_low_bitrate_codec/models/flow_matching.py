import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .bitlinear import BitLinear, RMSNorm

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

class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        # Predict scale and shift from conditioning
        # Use simple Linear here for stability as it's a critical path
        self.linear = nn.Linear(cond_dim, dim * 2)
        # Initialize shift to 0 and scale to 1 (conceptually)
        # In AdaLN-Zero we init to 0 so the block starts as identity
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        # x: (B, T, D), c: (B, T, D) or (B, D)
        if c.dim() == 2:
            c = c.unsqueeze(1)
        
        emb = self.linear(c) # (B, T, 2*D)
        scale, shift = torch.chunk(emb, 2, dim=-1)
        
        x = self.norm(x)
        # Apply modulation: (1 + scale) * x + shift
        return x * (1 + scale) + shift

class BitTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, cond_dim, dropout=0.0):
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = AdaLN(dim, cond_dim)
        self.ffn = BitFeedForward(dim, hidden_dim, dropout)
        
    def forward(self, x, c):
        # x: (B, T, D), c: (B, T, D)
        
        # Attention with AdaLN
        xn = self.norm1(x, c)
        attn_out, _ = self.attn(xn, xn, xn)
        x = x + attn_out
        
        # FFN with AdaLN
        xn = self.norm2(x, c)
        x = x + self.ffn(xn)
        return x

class SinusoidalPosEmbed(nn.Module):
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

class ConditionalFlowMatching(nn.Module):
    """
    Improved BitNet Flow Matching DiT.
    - High-dimensional hidden space (default 512).
    - AdaLN conditioning.
    - Hybrid Precision: Float for Head/Tail/AdaLN, BitNet for heavy FFN.
    """
    def __init__(self, config):
        super().__init__()
        
        # Dimensions
        # fusion_dim IS the output/target dimension (e.g. 80 for Mel)
        self.out_dim = config['model']['decoder']['fusion_dim'] 
        
        # Internal hidden dimension (e.g. 512)
        # If not specified in decoder config, default to 512 (BitNet standard)
        self.hidden_dim = config['model']['decoder'].get('hidden_dim', 512)
        
        self.num_layers = config['model'].get('flow_matching_layers', 8)
        self.num_heads = config['model']['decoder']['fusion_heads']
        self.dropout = config['model']['decoder']['dropout']
        
        # Time Embedding (Float for precision)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmbed(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        
        # Input Projection: Mel (out_dim) -> hidden_dim
        # Use standard Linear for the entry layer to preserve input precision
        self.input_proj = nn.Linear(self.out_dim, self.hidden_dim) 
        
        # Transformer Backbone (BitNet DiT)
        self.blocks = nn.ModuleList([
            BitTransformerBlock(
                dim=self.hidden_dim, 
                num_heads=self.num_heads, 
                hidden_dim=self.hidden_dim * 4, 
                cond_dim=self.hidden_dim, # cond is pre-fused to hidden_dim
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Output Projection: hidden_dim -> Mel (out_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.out_dim)
        
        # Final Norm
        self.final_norm = RMSNorm(self.hidden_dim)

    def forward(self, x_t, t, cond, mask=None):
        """
        Args:
            x_t: (B, T, out_dim) Noisy input
            t: (B,) Time step [0, 1]
            cond: (B, T, hidden_dim) Conditioning features
        """
        # Time embedding
        t_emb = self.time_mlp(t).unsqueeze(1) # (B, 1, D)
        
        # Project input to hidden dim
        x = self.input_proj(x_t)
        
        # Merge conditioning with time embedding
        # This becomes the 'c' for AdaLN
        c = cond + t_emb
        
        # Pass through DiT Blocks with AdaLN
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_norm(x)
        
        # Predict velocity (x_1 - x_0)
        v_pred = self.output_proj(x)
        
        return v_pred

    @torch.no_grad()
    def solve_ode(self, cond, steps=10, solver='euler', cfg_scale=1.0):
        # Reuse existing solve_ode logic but ensure shapes are consistent
        B, T, _ = cond.shape
        device = cond.device
        
        # Target dim is out_dim (e.g. 80)
        x = torch.randn(B, T, self.out_dim, device=device)
        t_span = torch.linspace(0, 1, steps + 1, device=device)
        
        def get_velocity(x, t, c):
            if cfg_scale > 1.0:
                v_c = self(x, t, c)
                v_u = self(x, t, torch.zeros_like(c))
                return v_u + cfg_scale * (v_c - v_u)
            else:
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
            else: # Euler
                v_pred = get_velocity(x, t_batch, cond)
                x = x + v_pred * dt
                
        return x
