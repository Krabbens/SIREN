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

class ManualFlashAttention(nn.Module):
    """
    Binary-compatible FlashAttention using F.scaled_dot_product_attention.
    Matches nn.MultiheadAttention state_dict keys for easy loading.
    """
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        
        # EXACT names to match nn.MultiheadAttention state_dict
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        B, T, C = x.shape
        # Project and split to Q, K, V
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias) # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # SDPA (FlashAttention kernel)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Combine heads
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out

class BitTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, cond_dim, dropout=0.0):
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim)
        # self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn = ManualFlashAttention(dim, num_heads, dropout)
        self.norm2 = AdaLN(dim, cond_dim)
        self.ffn = BitFeedForward(dim, hidden_dim, dropout)
        
    def forward(self, x, c):
        # x: (B, T, D), c: (B, T, D)
        
        # Attention with AdaLN
        xn = self.norm1(x, c)
        # attn_out, _ = self.attn(xn, xn, xn)
        attn_out = self.attn(xn)
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
        
        # Positional Embedding for x_t (improve temporal awareness)
        self.pos_embed = SinusoidalPosEmbed(self.hidden_dim)
        
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
        
        # Add positional encoding to x (noisified mel)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device).float()
        x = x + self.pos_embed(pos).unsqueeze(0)
        
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

    def compute_loss(self, x_1, cond, l1_weight=0.0):
        """
        Compute Flow Matching Loss.
        Args:
            x_1: (B, T, out_dim) Target data (Mel spectrogram)
            cond: (B, T, hidden_dim) Conditioning
            l1_weight: Weight for L1 loss on velocity (sharpening)
        Returns:
            loss: Scalar loss
        """
        B, T, _ = x_1.shape
        device = x_1.device
        
        # Sample x_0 (noise)
        x_0 = torch.randn_like(x_1)
        
        # Sample t
        t = torch.rand(B, device=device)
        
        # Linear interpolation (OT path)
        # x_t = (1 - t) * x_0 + t * x_1
        # reshape t for broadcasting: (B, 1, 1)
        t_b = t.view(B, 1, 1)
        x_t = (1 - t_b) * x_0 + t_b * x_1
        
        # Target velocity u_t = x_1 - x_0
        u_t = x_1 - x_0
        
        # Predict velocity
        v_pred = self(x_t, t, cond)
        
        # Loss
        # MSE is standard for OT-CFM
        loss = F.mse_loss(v_pred, u_t)
        
        # L1 Loss (Sharpening)
        if l1_weight > 0:
            loss = loss + l1_weight * F.l1_loss(v_pred, u_t)
            
        return loss

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
            elif solver == 'rk4':
                # Runge-Kutta 4
                k1 = get_velocity(x, t_batch, cond)
                
                t_half = t_batch + dt / 2
                x_k1 = x + k1 * (dt / 2)
                k2 = get_velocity(x_k1, t_half, cond)
                
                x_k2 = x + k2 * (dt / 2)
                k3 = get_velocity(x_k2, t_half, cond)
                
                t_next = t_batch + dt
                x_k3 = x + k3 * dt
                k4 = get_velocity(x_k3, t_next, cond)
                
                x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            elif solver == 'heun':
                # Heun's Method (Predictor-Corrector)
                # Predict (Euler)
                v1 = get_velocity(x, t_batch, cond)
                x_pred = x + v1 * dt
                
                # Correct
                t_next = t_batch + dt
                v2 = get_velocity(x_pred, t_next, cond)
                x = x + (v1 + v2) * (dt / 2)
            else: # Euler
                v_pred = get_velocity(x, t_batch, cond)
                x = x + v_pred * dt
                
        return x
