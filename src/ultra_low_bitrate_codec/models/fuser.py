import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConditionFuser(nn.Module):
    """
    Fuses semantic, prosody, and speaker embeddings for Flow Matching conditioning.
    Includes temporal interpolation and Gaussian smoothing.
    """
    def __init__(self, sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512):
        super().__init__()
        self.proj = nn.Linear(sem_dim + pro_dim + spk_dim, out_dim)
        # Learnable smoothing to prevent temporal artifacts from upsampling
        self.smooth = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        
        with torch.no_grad():
            sigma = 1.5
            k = 5
            x = torch.arange(k).float() - k//2
            y = torch.arange(k).float() - k//2
            xv, yv = torch.meshgrid(x, y, indexing='ij')
            gauss = torch.exp(-0.5 * (xv**2 + yv**2) / sigma**2)
            gauss = gauss / gauss.sum()
            self.smooth.weight.data.copy_(gauss.view(1, 1, k, k))

    def forward(self, s, p, spk, target_len):
        """
        Args:
            s: Semantic embeddings (B, T_s, D_s)
            p: Prosody embeddings (B, T_p, D_p)
            spk: Speaker embedding (B, D_spk)
            target_len: Target Mel spectrogram length
        Returns:
            fused: (B, target_len, out_dim)
        """
        # Interpolate content to target length
        s = s.transpose(1, 2)
        p = p.transpose(1, 2)
        
        s = F.interpolate(s, size=target_len, mode='linear', align_corners=False)
        p = F.interpolate(p, size=target_len, mode='linear', align_corners=False)
        
        s = s.transpose(1, 2)
        p = p.transpose(1, 2)
        
        # Expand speaker to every frame
        spk = spk.unsqueeze(1).expand(-1, target_len, -1)
        
        # Concatenate and Project
        cat = torch.cat([s, p, spk], dim=-1)
        x = self.proj(cat)  # (B, T, out_dim)

        # Apply smoothing (DISABLED: Checkpoint does not have this layer)
        # x = x.unsqueeze(1)
        # x = self.smooth(x)
        # x = x.squeeze(1)

        return x


# ============================================================================
# ConditionFuserV2: Edge-Friendly with Better Conditioning
# ============================================================================

class SinusoidalPosEnc(nn.Module):
    """Sinusoidal positional encoding - zero learnable parameters."""
    def __init__(self, dim, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:x.size(1)]


class LearnedUpsample(nn.Module):
    """
    Learned upsampling using transposed convolution.
    More edge-friendly than cross-attention.
    """
    def __init__(self, in_dim, out_dim, upsample_factor=4):
        super().__init__()
        # Use ConvTranspose1d for learned upsampling
        # kernel_size = 2 * upsample_factor gives smooth results
        self.conv = nn.ConvTranspose1d(
            in_dim, out_dim, 
            kernel_size=upsample_factor * 2,
            stride=upsample_factor,
            padding=upsample_factor // 2
        )
        # Small smoothing conv to reduce checkerboard artifacts
        self.smooth = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim)
        
    def forward(self, x, target_len):
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.smooth(x)
        # Adjust to exact target length
        if x.size(2) != target_len:
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x.transpose(1, 2)  # (B, T, D)


class LightweightCrossAttention(nn.Module):
    """
    Single-head cross-attention for edge devices.
    Query: target frames, Key/Value: conditioning
    ~32K params for dim=256
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        
        # Use standard Linear (can be swapped to BitLinear later)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, context):
        """
        Args:
            query: (B, T_q, D) - target frame queries
            context: (B, T_c, D) - conditioning context
        Returns:
            out: (B, T_q, D)
        """
        B, T_q, D = query.shape
        
        q = self.q_proj(query)  # (B, T_q, D)
        kv = self.kv_proj(context)  # (B, T_c, 2*D)
        k, v = kv.chunk(2, dim=-1)  # (B, T_c, D) each
        
        # Attention
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, T_q, T_c)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.bmm(attn, v)  # (B, T_q, D)
        out = self.out_proj(out)
        
        return out


class ConditionFuserV2(nn.Module):
    """
    Improved Condition Fuser for Flow Matching (Edge-Friendly).
    
    Improvements over V1:
    1. Learned upsampling (ConvTranspose1d) instead of F.interpolate
    2. Lightweight cross-attention for temporal alignment
    3. Sinusoidal positional encodings
    
    Designed for edge deployment:
    - Single-head attention (~32K params)
    - Efficient convolutions
    - No heavy transformers
    
    Total extra params: ~100K (vs 140K+ for multi-head attention)
    """
    def __init__(self, sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, 
                 sem_upsample=4, pro_upsample=8):
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = out_dim // 2  # 256 for efficiency
        
        # Initial projections to hidden dim
        self.sem_proj = nn.Linear(sem_dim, self.hidden_dim)
        self.pro_proj = nn.Linear(pro_dim, self.hidden_dim)
        self.spk_proj = nn.Linear(spk_dim, self.hidden_dim)
        
        # Learned upsampling for semantic and prosody
        self.sem_upsample = LearnedUpsample(self.hidden_dim, self.hidden_dim, sem_upsample)
        self.pro_upsample = LearnedUpsample(self.hidden_dim, self.hidden_dim, pro_upsample)
        
        # Positional encoding
        self.pos_enc = SinusoidalPosEnc(self.hidden_dim)
        
        # Cross-attention: fused features attend to speaker context
        self.cross_attn = LightweightCrossAttention(self.hidden_dim)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, out_dim),  # cat(semantic+prosody, speaker_attended)
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Temporal smoothing (1D Gaussian)
        self.smooth = nn.Conv1d(out_dim, out_dim, kernel_size=5, padding=2, groups=out_dim, bias=False)
        self._init_gaussian_kernel()
        
    def _init_gaussian_kernel(self):
        with torch.no_grad():
            sigma = 1.0
            k = 5
            x = torch.arange(k).float() - k // 2
            gauss = torch.exp(-0.5 * x**2 / sigma**2)
            gauss = gauss / gauss.sum()
            # Expand for grouped conv
            self.smooth.weight.data.copy_(gauss.view(1, 1, k).expand(self.out_dim, 1, k))
        
    def forward(self, s, p, spk, target_len):
        """
        Args:
            s: Semantic embeddings (B, T_s, D_s)
            p: Prosody embeddings (B, T_p, D_p)
            spk: Speaker embedding (B, D_spk)
            target_len: Target Mel spectrogram length
        Returns:
            fused: (B, target_len, out_dim)
        """
        B = s.size(0)
        
        # Project to hidden dim
        s = self.sem_proj(s)  # (B, T_s, hidden)
        p = self.pro_proj(p)  # (B, T_p, hidden)
        spk_h = self.spk_proj(spk)  # (B, hidden)
        
        # Learned upsampling
        s = self.sem_upsample(s, target_len)  # (B, target_len, hidden)
        p = self.pro_upsample(p, target_len)  # (B, target_len, hidden)
        
        # Combine semantic + prosody
        content = s + p  # (B, target_len, hidden)
        
        # Add positional encoding
        content = self.pos_enc(content)
        
        # Expand speaker as context for cross-attention (1 frame)
        spk_ctx = spk_h.unsqueeze(1)  # (B, 1, hidden)
        
        # Cross-attention: content attends to speaker
        spk_info = self.cross_attn(content, spk_ctx)  # (B, target_len, hidden)
        
        # Concatenate and project
        fused = torch.cat([content, spk_info], dim=-1)  # (B, T, hidden*2)
        fused = self.final_proj(fused)  # (B, T, out_dim)
        
        # Temporal smoothing
        fused = fused.transpose(1, 2)  # (B, out_dim, T)
        fused = self.smooth(fused)
        fused = fused.transpose(1, 2)  # (B, T, out_dim)
        
        return fused
