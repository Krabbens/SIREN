import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Apply smoothing
        x = x.unsqueeze(1)
        x = self.smooth(x)
        x = x.squeeze(1)

        return x
