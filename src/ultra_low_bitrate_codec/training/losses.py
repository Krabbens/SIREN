"""
SIREN v2 Loss Functions
Cleaned up, performance-focused implementations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CORE SPECTRAL LOSSES
# =============================================================================

class STFTLoss(nn.Module):
    """Single-resolution STFT loss (spectral convergence + log magnitude)."""
    
    __constants__ = ['fft_size', 'hop_size', 'win_length']
    
    def __init__(self, fft_size: int, hop_size: int, win_length: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Generated audio (B, T)
            y: Target audio (B, T)
        Returns:
            (spectral_convergence_loss, log_magnitude_loss)
        """
        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, 
                           window=self.window, return_complex=True)
        y_stft = torch.stft(y, self.fft_size, self.hop_size, self.win_length, 
                           window=self.window, return_complex=True)
        
        x_mag = x_stft.abs().clamp(min=1e-7)
        y_mag = y_stft.abs().clamp(min=1e-7)
        
        # Spectral convergence: normalized Frobenius norm
        sc_loss = (y_mag - x_mag).norm(p='fro') / y_mag.norm(p='fro')
        
        # Log magnitude L1
        mag_loss = F.l1_loss(x_mag.log(), y_mag.log())
        
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for time-frequency trade-off."""
    
    def __init__(self,
                 fft_sizes: tuple = (512, 1024, 2048),
                 hop_sizes: tuple = (50, 120, 240),
                 win_lengths: tuple = (240, 600, 1200)):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(fs, hs, wl) 
            for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        self._num_losses = len(self.losses)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sc_loss = torch.tensor(0.0, device=x.device)
        mag_loss = torch.tensor(0.0, device=x.device)
        
        for loss_fn in self.losses:
            sc, mag = loss_fn(x, y)
            sc_loss = sc_loss + sc
            mag_loss = mag_loss + mag
        
        return sc_loss / self._num_losses, mag_loss / self._num_losses


# =============================================================================
# MEL LOSSES
# =============================================================================

class BandwiseMelLoss(nn.Module):
    """
    Band-wise Mel loss with per-band weighting.
    Prevents model from sacrificing high frequencies.
    
    Performance: Single Mel computation, vectorized band slicing.
    """
    
    def __init__(self, 
                 n_mels: int = 80,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 band_weights: tuple = (1.0, 1.5, 1.5, 2.0),
                 device: str = 'cuda'):
        super().__init__()
        import torchaudio
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False
        ).to(device)
        
        # Precompute band boundaries
        n_bands = len(band_weights)
        band_size = n_mels // n_bands
        self.register_buffer('band_starts', torch.arange(0, n_mels, band_size)[:n_bands])
        self.register_buffer('band_ends', torch.cat([
            torch.arange(band_size, n_mels, band_size)[:n_bands-1],
            torch.tensor([n_mels])
        ]))
        self.register_buffer('band_weights', torch.tensor(band_weights, dtype=torch.float32))
        self._weight_sum = sum(band_weights)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Generated audio (B, T)
            y: Target audio (B, T)
        """
        # Align lengths
        min_len = min(x.shape[-1], y.shape[-1])
        x, y = x[..., :min_len], y[..., :min_len]
        
        # Compute Mel (B, n_mels, T_frames)
        mel_x = self.mel_transform(x).clamp(min=1e-5).log()
        mel_y = self.mel_transform(y).clamp(min=1e-5).log()
        
        # Vectorized band loss computation
        loss = torch.tensor(0.0, device=x.device)
        for i, weight in enumerate(self.band_weights):
            start, end = int(self.band_starts[i]), int(self.band_ends[i])
            band_loss = F.l1_loss(mel_x[:, start:end], mel_y[:, start:end])
            loss = loss + band_loss * weight
            
        return loss / self._weight_sum


# =============================================================================
# TIME-DOMAIN LOSSES
# =============================================================================

class WaveformL1Loss(nn.Module):
    """
    Direct L1 loss on waveform.
    Implicitly enforces phase alignment (phase errors = waveform errors).
    Extremely fast - no FFT computation.
    """
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        min_len = min(x.shape[-1], y.shape[-1])
        return F.l1_loss(x[..., :min_len], y[..., :min_len])


class SpectralFluxLoss(nn.Module):
    """
    Spectral flux loss - penalizes temporal discontinuities.
    Measures rate of change in spectrum over time.
    
    Performance: Reuses STFT computation, vectorized diff.
    """
    
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        min_len = min(x.shape[-1], y.shape[-1])
        x, y = x[..., :min_len], y[..., :min_len]
        
        # STFT magnitudes
        x_spec = torch.stft(x, self.n_fft, self.hop_length, 
                           window=self.window, return_complex=True).abs()
        y_spec = torch.stft(y, self.n_fft, self.hop_length, 
                           window=self.window, return_complex=True).abs()
        
        # Temporal difference (flux)
        x_flux = torch.diff(x_spec, dim=-1)
        y_flux = torch.diff(y_spec, dim=-1)
        
        return F.l1_loss(x_flux, y_flux)


# =============================================================================
# LEGACY LOSSES (kept for compatibility, use sparingly)
# =============================================================================

class PhaseAwareLoss(nn.Module):
    """Group delay matching for temporal coherence."""
    
    def __init__(self, fft_size: int = 1024, hop_size: int = 256):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.register_buffer("window", torch.hann_window(fft_size))
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        min_len = min(x.shape[-1], y.shape[-1])
        x, y = x[..., :min_len].squeeze(1) if x.dim() == 3 else x[..., :min_len], \
               y[..., :min_len].squeeze(1) if y.dim() == 3 else y[..., :min_len]
        
        # STFT phase
        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.fft_size,
                           window=self.window, return_complex=True)
        y_stft = torch.stft(y, self.fft_size, self.hop_size, self.fft_size,
                           window=self.window, return_complex=True)
        
        x_phase = x_stft.angle()
        y_phase = y_stft.angle()
        
        # Group delay = negative frequency derivative of phase
        x_gd = -torch.diff(x_phase, dim=1)
        y_gd = -torch.diff(y_phase, dim=1)
        
        # Wrap to [-pi, pi]
        x_gd = x_gd - 2 * torch.pi * (x_gd / (2 * torch.pi)).round()
        y_gd = y_gd - 2 * torch.pi * (y_gd / (2 * torch.pi)).round()
        
        min_t = min(x_gd.shape[-1], y_gd.shape[-1])
        return F.l1_loss(x_gd[..., :min_t], y_gd[..., :min_t])


# =============================================================================
# REGULARIZATION LOSSES
# =============================================================================

class SnakeBetaDiversityLoss(nn.Module):
    """
    Regularization loss to prevent SnakeBeta alpha parameters from clustering.
    
    Problem: When alpha values cluster (e.g., 67% in [1.04, 1.11]), SnakeBeta
    activations add harmonics at nearly identical frequencies, causing 
    horizontal banding artifacts in spectrograms.
    
    Solution: Encourage log-uniform spread of alpha values across the model.
    This ensures diverse frequency responses from sin²(αx) terms.
    
    Loss = -std(log(α)) + (mean(log(α)) - target_mean)² + entropy_penalty
    """
    
    def __init__(self, target_log_std: float = 1.5, target_log_mean: float = 1.0,
                 entropy_weight: float = 0.1):
        super().__init__()
        self.target_log_std = target_log_std
        self.target_log_mean = target_log_mean
        self.entropy_weight = entropy_weight
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute alpha diversity loss for all SnakeBeta modules in model.
        
        Args:
            model: Model containing SnakeBeta layers
        Returns:
            Scalar loss encouraging alpha diversity
        """
        from ..models.post_net import SnakeBeta, SnakePhase
        
        alphas = []
        for m in model.modules():
            if isinstance(m, (SnakeBeta, SnakePhase)):
                alphas.append(m.alpha.flatten())
        
        if not alphas:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        all_alphas = torch.cat(alphas)
        log_alphas = torch.log(all_alphas.abs() + 1e-9)
        
        # 1. Encourage high log-variance (spread)
        std_loss = F.relu(self.target_log_std - log_alphas.std())
        
        # 2. Center the distribution around target mean
        mean_loss = (log_alphas.mean() - self.target_log_mean) ** 2
        
        # 3. Anti-clustering: penalize if too many values are close together
        # Using histogram-based entropy approximation
        if len(all_alphas) > 10:
            # Soft histogram via Gaussian KDE approximation
            diffs = all_alphas.unsqueeze(0) - all_alphas.unsqueeze(1)
            sigma = all_alphas.std() / 3 + 1e-6
            kde = torch.exp(-0.5 * (diffs / sigma) ** 2).mean(dim=1)
            entropy = -torch.mean(torch.log(kde + 1e-9))
            entropy_loss = F.relu(2.0 - entropy)  # Encourage entropy > 2
        else:
            entropy_loss = torch.tensor(0.0, device=all_alphas.device)
        
        total = std_loss + 0.5 * mean_loss + self.entropy_weight * entropy_loss
        return total
