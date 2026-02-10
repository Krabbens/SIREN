"""
TinyDiffusion Enhancer - Ultra-compact diffusion model for mel-spectrogram refinement.

Features:
- Ternary weights (BitConv1d) for <1MB model size
- SnakeBeta activations for audio-domain inductive bias
- Few-step diffusion (4-8 steps) for fast inference
- Band-aware processing (low/mid/high frequency branches)

Usage:
    enhancer = TinyDiffusionEnhancer()
    refined_mel = enhancer.sample(noisy_mel, num_steps=4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .bitlinear import BitConv1d, RMSNorm
from .post_net import SnakeBeta


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class BitResBlock(nn.Module):
    """Residual block with BitConv1d and SnakeBeta."""
    def __init__(self, channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = BitConv1d(channels, channels, kernel_size=3, padding=1)
        self.act1 = SnakeBeta(channels)
        self.conv2 = BitConv1d(channels, channels, kernel_size=3, padding=1)
        self.act2 = SnakeBeta(channels)
        
        if time_emb_dim:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, channels),
                nn.SiLU(),
            )
        else:
            self.time_mlp = None
            
    def forward(self, x, time_emb=None):
        h = self.conv1(x)
        h = self.act1(h)
        
        if self.time_mlp is not None and time_emb is not None:
            h = h + self.time_mlp(time_emb)[:, :, None]
            
        h = self.conv2(h)
        h = self.act2(h)
        
        return x + h


class BandSplit(nn.Module):
    """Split mel-spectrogram into frequency bands."""
    def __init__(self, n_mels=80, n_bands=3):
        super().__init__()
        self.n_bands = n_bands
        # Calculate splits: 80 / 3 = 26, 27, 27
        base = n_mels // n_bands
        remainder = n_mels % n_bands
        self.splits = [base + (1 if i < remainder else 0) for i in range(n_bands)]
        
    def forward(self, x):
        # x: (B, C, T) where C = n_mels
        bands = torch.split(x, self.splits, dim=1)
        return bands
    
    def merge(self, bands):
        return torch.cat(bands, dim=1)


class TinyUNet(nn.Module):
    """
    Tiny U-Net for mel-spectrogram denoising.
    Uses band-aware processing with BitConv1d.
    """
    def __init__(self, in_channels=80, hidden_dim=64, time_emb_dim=64, n_bands=3):
        super().__init__()
        
        self.n_bands = n_bands
        self.band_split = BandSplit(in_channels, n_bands)
        
        # Get actual band sizes
        self.band_sizes = self.band_split.splits
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        # Per-band processing (each band has different size)
        hidden_per_band = hidden_dim // n_bands
        self.band_encoders = nn.ModuleList([
            nn.Conv1d(self.band_sizes[i], hidden_per_band, kernel_size=3, padding=1)
            for i in range(n_bands)
        ])
        
        # Shared processing
        self.encoder = nn.Sequential(
            BitConv1d(hidden_per_band * n_bands, hidden_dim, kernel_size=3, padding=1),
            SnakeBeta(hidden_dim),
        )
        
        # Bottleneck with time conditioning
        self.bottleneck = nn.ModuleList([
            BitResBlock(hidden_dim, time_emb_dim),
            BitResBlock(hidden_dim, time_emb_dim),
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            BitConv1d(hidden_dim, hidden_per_band * n_bands, kernel_size=3, padding=1),
            SnakeBeta(hidden_per_band * n_bands),
        )
        
        # Output projection back to mel bands
        self.band_decoders = nn.ModuleList([
            nn.Conv1d(hidden_per_band, self.band_sizes[i], kernel_size=3, padding=1)
            for i in range(n_bands)
        ])
        
        # Final refinement
        self.final = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t):
        """
        Args:
            x: Noisy mel (B, n_mels, T)
            t: Timestep (B,)
        Returns:
            Predicted noise (B, n_mels, T)
        """
        # Time embedding
        time_emb = self.time_mlp(t)
        
        # Split into bands
        bands = self.band_split(x)
        
        # Encode each band
        band_features = []
        for i, (band, enc) in enumerate(zip(bands, self.band_encoders)):
            band_features.append(enc(band))
        
        # Merge and process
        h = torch.cat(band_features, dim=1)
        h = self.encoder(h)
        
        # Bottleneck with time conditioning
        for block in self.bottleneck:
            h = block(h, time_emb)
        
        # Decode
        h = self.decoder(h)
        
        # Split back to bands
        band_out = torch.split(h, h.shape[1] // self.n_bands, dim=1)
        
        # Decode each band
        decoded_bands = []
        for i, (band_h, dec) in enumerate(zip(band_out, self.band_decoders)):
            decoded_bands.append(dec(band_h))
        
        # Merge bands
        out = self.band_split.merge(decoded_bands)
        
        # Final refinement
        out = self.final(out)
        
        return out


class TinyDiffusionEnhancer(nn.Module):
    """
    Main TinyDiffusion Enhancer model.
    Trained to denoise mel-spectrograms for quality enhancement.
    """
    def __init__(self, n_mels=80, hidden_dim=64, n_steps=1000):
        super().__init__()
        self.n_mels = n_mels
        self.n_steps = n_steps
        
        # Noise prediction network
        self.unet = TinyUNet(
            in_channels=n_mels,
            hidden_dim=hidden_dim,
            time_emb_dim=64,
            n_bands=3
        )
        
        # Diffusion schedule (cosine)
        self.register_buffer('betas', self._cosine_beta_schedule(n_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.02)
    
    def q_sample(self, x_start, t, noise=None):
        """Add noise to data (forward diffusion)."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt()[:, None, None]
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[t]).sqrt()[:, None, None]
        
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
    
    def predict_noise(self, x_noisy, t):
        """Predict noise from noisy input."""
        return self.unet(x_noisy, t.float())
    
    def forward(self, x_clean, return_noise=False):
        """Training forward pass."""
        B = x_clean.shape[0]
        device = x_clean.device
        
        # Random timesteps
        t = torch.randint(0, self.n_steps, (B,), device=device)
        
        # Add noise
        noise = torch.randn_like(x_clean)
        x_noisy = self.q_sample(x_clean, t, noise)
        
        # Predict noise
        noise_pred = self.predict_noise(x_noisy, t)
        
        if return_noise:
            return noise_pred, noise
        return noise_pred
    
    @torch.no_grad()
    def sample(self, x_corrupted, num_steps=4):
        """
        Enhancement via iterative residual denoising.
        
        The model is trained to predict noise added to clean mels.
        For enhancement, we treat x_corrupted as "clean + artifacts" and
        iteratively predict/remove the "artifact noise".
        
        Args:
            x_corrupted: Degraded mel spectrogram (B, n_mels, T)
            num_steps: Number of refinement steps (default: 4)
        Returns:
            Refined mel spectrogram (B, n_mels, T)
        """
        device = x_corrupted.device
        B = x_corrupted.shape[0]
        
        x = x_corrupted.clone()
        
        # Use decreasing timesteps for gradual refinement
        # Start with medium noise level (not max) since input isn't pure noise
        timesteps = torch.linspace(500, 50, num_steps).long()
        
        for step_t in timesteps:
            t = step_t.expand(B).to(device)
            
            # Predict "noise" at this level
            noise_pred = self.predict_noise(x, t)
            
            # Get alpha for this timestep
            alpha = self.alphas_cumprod[t][:, None, None]
            
            # Simple residual update: subtract scaled noise prediction
            # Lower alpha = more noise expected, so scale removal accordingly
            noise_scale = (1 - alpha).sqrt() * 0.3  # Conservative scaling
            x = x - noise_scale * noise_pred
        
        # Blend with original to preserve structure (70% enhanced, 30% original)
        x = 0.7 * x + 0.3 * x_corrupted
        
        return x
    
    @torch.no_grad()
    def enhance_simple(self, x_corrupted, strength=0.5):
        """
        Single-step enhancement (faster, simpler).
        
        Args:
            x_corrupted: Degraded mel spectrogram (B, n_mels, T)
            strength: Enhancement strength 0-1 (default: 0.5)
        Returns:
            Refined mel spectrogram (B, n_mels, T)
        """
        device = x_corrupted.device
        B = x_corrupted.shape[0]
        
        # Use mid-level timestep
        t = torch.full((B,), 250, device=device, dtype=torch.long)
        
        # Predict noise/artifacts
        noise_pred = self.predict_noise(x_corrupted, t)
        
        # Subtract predicted noise scaled by strength
        x_enhanced = x_corrupted - strength * noise_pred
        
        return x_enhanced
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        quantizable = sum(p.numel() for m in self.modules() if isinstance(m, BitConv1d) for p in m.parameters())
        return {'total': total, 'quantizable': quantizable}


# Test
if __name__ == "__main__":
    print("Testing TinyDiffusionEnhancer...")
    
    model = TinyDiffusionEnhancer(n_mels=80, hidden_dim=64)
    
    # Test forward (training)
    mel = torch.randn(2, 80, 100)
    noise_pred, noise = model(mel, return_noise=True)
    print(f"Training: mel={mel.shape} -> noise_pred={noise_pred.shape}")
    
    # Test sampling (inference)
    corrupted = torch.randn(2, 80, 100)
    refined = model.sample(corrupted, num_steps=4)
    print(f"Inference: corrupted={corrupted.shape} -> refined={refined.shape}")
    
    # Count params
    counts = model.count_parameters()
    print(f"\nParameters: {counts['total']:,} (quantizable: {counts['quantizable']:,})")
    
    print("\n✓ TinyDiffusionEnhancer test passed!")
