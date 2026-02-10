import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        # Optimization: Reduced periods from [2, 3, 5, 7, 11] to [2, 3, 5] for speed
        periods = [2, 3, 5]
        
        discs = [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorR(nn.Module):
    def __init__(self, resolution, use_spectral_norm=False):
        super(DiscriminatorR, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        
        self.resolution = resolution
        # n_fft, hop_length, win_length as per resolution tuple? 
        # Usually resolution is just n_fft, others derived or fixed.
        # But for simplicity let's assume 'resolution' is a config dict or tuple.
        # Implementing standard HiFiGAN scales:
        # 1: (1024, 120, 600)
        # 2: (2048, 240, 1200)
        # 3: (512, 50, 240)
        
        # We can just hardcode the STFT parameters in the wrapper and pass spectrograms?
        # No, MultiScaleDiscriminator usually operates on raw audio with different downsampling?
        # Wait, HiFiGAN has Multi-Period and Multi-SCALE.
        # EnCodec/Vocos use Multi-Resolution (STFT-based).
        # MRD is better for spectral artifacts.
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiResolutionDiscriminator(nn.Module):
    def __init__(self):
        super(MultiResolutionDiscriminator, self).__init__()
        self.resolutions = [
            (1024, 120, 600),
            (2048, 240, 1200),
            (512, 50, 240),
        ]
        
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(res) for res in self.resolutions]
        )
        
        # Pre-cache windows as buffers (avoids allocation on every forward pass)
        for i, (n_fft, hop_length, win_length) in enumerate(self.resolutions):
            self.register_buffer(f'window_{i}', torch.hann_window(win_length))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(self.discriminators):
            n_fft, hop_length, win_length = self.resolutions[i]
            window = getattr(self, f'window_{i}')
            
            # STFT (using cached window)
            y_stft = torch.stft(y.squeeze(1), n_fft, hop_length, win_length, return_complex=True, window=window)
            y_hat_stft = torch.stft(y_hat.squeeze(1), n_fft, hop_length, win_length, return_complex=True, window=window)
            
            y_mag = torch.abs(y_stft).unsqueeze(1)
            y_hat_mag = torch.abs(y_hat_stft).unsqueeze(1) # (B, 1, F, T)
            
            y_d_r, fmap_r = d(y_mag)
            y_d_g, fmap_g = d(y_hat_mag)
            
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class HiFiGANDiscriminator(nn.Module):
    def __init__(self):
        super(HiFiGANDiscriminator, self).__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator()
    
    def forward(self, y, y_hat):
        # y, y_hat: (B, 1, T)
        
        mpd_res = self.mpd(y, y_hat)
        mrd_res = self.mrd(y, y_hat)
        
        return mpd_res, mrd_res


# =============================================================================
# BITNET GAN (BitDiscriminator)
# =============================================================================
from .bitlinear import BitConv2d, BitSnakeBeta

class BitSnakeBeta2d(nn.Module):
    """Wrapper for BitSnakeBeta to handle 4D tensors (B, C, H, W)."""
    def __init__(self, channels):
        super().__init__()
        self.snake = BitSnakeBeta(channels)
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W) # Flatten spatial
        x = self.snake(x)
        x = x.view(B, C, H, W)
        return x

class BitDiscriminatorP(nn.Module):
    """BitNet version of Period Discriminator."""
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        
        # No weight_norm needed for BitNet, it has internal RMSNorm
        self.convs = nn.ModuleList([
            BitConv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            BitConv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            BitConv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            BitConv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            BitConv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
        ])
        
        self.activations = nn.ModuleList([
            BitSnakeBeta2d(32),
            BitSnakeBeta2d(128),
            BitSnakeBeta2d(512),
            BitSnakeBeta2d(1024),
            BitSnakeBeta2d(1024),
        ])
        
        self.conv_post = BitConv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for cv, act in zip(self.convs, self.activations):
            x = cv(x)
            x = act(x)
            fmap.append(x)
            
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class BitMultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        periods = [2, 3, 5]
        self.discriminators = nn.ModuleList([BitDiscriminatorP(i) for i in periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r); y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class BitDiscriminatorR(nn.Module):
    """BitNet version of Resolution Discriminator."""
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution
        
        self.convs = nn.ModuleList([
            BitConv2d(1, 32, (3, 9), padding=(1, 4)),
            BitConv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            BitConv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            BitConv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            BitConv2d(32, 32, (3, 3), padding=(1, 1)),
        ])
        
        self.activations = nn.ModuleList([
            BitSnakeBeta2d(32),
            BitSnakeBeta2d(32),
            BitSnakeBeta2d(32),
            BitSnakeBeta2d(32),
            BitSnakeBeta2d(32),
        ])
        
        self.conv_post = BitConv2d(32, 1, (3, 3), padding=(1, 1))

    def forward(self, x):
        fmap = []
        for cv, act in zip(self.convs, self.activations):
            x = cv(x)
            x = act(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class BitMultiResolutionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.resolutions = [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]
        self.discriminators = nn.ModuleList([BitDiscriminatorR(res) for res in self.resolutions])
        for i, (_, _, win_length) in enumerate(self.resolutions):
            self.register_buffer(f'window_{i}', torch.hann_window(win_length))

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            n_fft, hop_length, win_length = self.resolutions[i]
            window = getattr(self, f'window_{i}')
            y_stft = torch.stft(y.squeeze(1), n_fft, hop_length, win_length, return_complex=True, window=window).abs().unsqueeze(1)
            y_hat_stft = torch.stft(y_hat.squeeze(1), n_fft, hop_length, win_length, return_complex=True, window=window).abs().unsqueeze(1)
            y_d_r, fmap_r = d(y_stft)
            y_d_g, fmap_g = d(y_hat_stft)
            y_d_rs.append(y_d_r); y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class BitHiFiGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = BitMultiPeriodDiscriminator()
        self.mrd = BitMultiResolutionDiscriminator()
    
    def forward(self, y, y_hat):
        mpd_res = self.mpd(y, y_hat)
        mrd_res = self.mrd(y, y_hat)
        return mpd_res, mrd_res


class BitSpectrogramDiscriminator(nn.Module):
    """
    BitNet GAN Discriminator for Spectrograms (2D).
    Skips STFT and Periodicity checks; operates directly on Mel Spectrograms.
    """
    def __init__(self):
        super().__init__()
        # Multi-Scale Discriminator on Spectrograms
        # Scale 1.0, 0.5, 0.25
        self.scales = [1, 0.5, 0.25]
        self.discriminators = nn.ModuleList([
            BitDiscriminatorR(resolution=None) for _ in self.scales
        ])
        
    def forward(self, y, y_hat):
        # y, y_hat: (B, 1, F, T) Mel Spectrograms
        
        y_d_rs, y_d_gs = [], []
        fmap_rs, fmap_gs = [], []
        
        for i, (d, scale) in enumerate(zip(self.discriminators, self.scales)):
            if scale == 1:
                y_in = y
                y_hat_in = y_hat
            else:
                y_in = F.interpolate(y, scale_factor=scale, mode='bilinear', align_corners=False)
                y_hat_in = F.interpolate(y_hat, scale_factor=scale, mode='bilinear', align_corners=False)
                
            y_d_r, fmap_r = d(y_in)
            y_d_g, fmap_g = d(y_hat_in)
            
            y_d_rs.append(y_d_r); y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r); fmap_gs.append(fmap_g)
            
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
