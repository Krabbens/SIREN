"""
Spectral Sharpener Module
Post-vocoder processing to recover high-frequency details.
Operates in STFT domain for efficient processing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block for spectral processing"""
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * 4, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
        
    def forward(self, x):
        # x: (B, C, T)
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)  # (B, C, T)
        return residual + x


class SpectralSharpener(nn.Module):
    """
    Learnable post-processing module that enhances high-frequency details.
    Operates on STFT magnitude and learns to predict the residual.
    """
    def __init__(self, n_fft=1024, hop_length=256, num_layers=4, hidden_dim=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_bins = n_fft // 2 + 1
        
        # Register window as buffer
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # Project STFT bins to hidden dim
        self.input_proj = nn.Conv1d(self.num_bins, hidden_dim, 1)
        
        # ConvNeXt blocks for processing
        self.layers = nn.ModuleList([
            ConvNeXtBlock(hidden_dim, kernel_size=7) for _ in range(num_layers)
        ])
        
        # Output projection (predicts magnitude residual)
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, self.num_bins, 1),
            nn.Tanh()  # Residual bounded to [-1, 1]
        )
        
        # Learnable scaling for residual
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, audio):
        """
        Args:
            audio: (B, T) or (B, 1, T) waveform
        Returns:
            enhanced: (B, T) or (B, 1, T) enhanced waveform
        """
        squeeze = False
        if audio.dim() == 3:
            squeeze = True
            audio = audio.squeeze(1)
        
        B, T = audio.shape
        
        # STFT
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, self.n_fft,
            window=self.window, return_complex=True
        )  # (B, F, T')
        
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Log magnitude for better dynamic range
        log_mag = torch.log(mag.clamp(min=1e-7))
        
        # Process magnitude
        x = self.input_proj(log_mag)  # (B, hidden_dim, T')
        
        for layer in self.layers:
            x = layer(x)
        
        # Predict residual
        residual = self.output_proj(x) * self.residual_scale  # (B, F, T')
        
        # Apply residual to magnitude
        enhanced_log_mag = log_mag + residual
        enhanced_mag = torch.exp(enhanced_log_mag)
        
        # Reconstruct with original phase
        enhanced_stft = enhanced_mag * torch.exp(1j * phase)
        
        # iSTFT
        enhanced = torch.istft(
            enhanced_stft, self.n_fft, self.hop_length, self.n_fft,
            window=self.window, length=T
        )
        
        if squeeze:
            enhanced = enhanced.unsqueeze(1)
        
        return enhanced


class SpectralSharpenerLight(nn.Module):
    """
    Lightweight version for edge deployment.
    Uses simpler convolutions instead of ConvNeXt.
    """
    def __init__(self, n_fft=512, hop_length=128, hidden_dim=64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_bins = n_fft // 2 + 1
        
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # Simple 3-layer CNN
        self.net = nn.Sequential(
            nn.Conv1d(self.num_bins, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, self.num_bins, 3, padding=1),
            nn.Tanh()
        )
        
        self.scale = nn.Parameter(torch.tensor(0.05))
        
    def forward(self, audio):
        squeeze = False
        if audio.dim() == 3:
            squeeze = True
            audio = audio.squeeze(1)
        
        B, T = audio.shape
        
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, self.n_fft,
            window=self.window, return_complex=True
        )
        
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        log_mag = torch.log(mag.clamp(min=1e-7))
        
        residual = self.net(log_mag) * self.scale
        enhanced_mag = torch.exp(log_mag + residual)
        
        enhanced_stft = enhanced_mag * torch.exp(1j * phase)
        enhanced = torch.istft(
            enhanced_stft, self.n_fft, self.hop_length, self.n_fft,
            window=self.window, length=T
        )
        
        if squeeze:
            enhanced = enhanced.unsqueeze(1)
        
        return enhanced
