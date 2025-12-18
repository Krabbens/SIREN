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
        periods = [2, 3, 5, 7, 11]
        
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

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(self.discriminators):
            n_fft, hop_length, win_length = self.resolutions[i]
            
            # STFT
            y_stft = torch.stft(y.squeeze(1), n_fft, hop_length, win_length, return_complex=True, window=torch.hann_window(win_length).to(y.device))
            y_hat_stft = torch.stft(y_hat.squeeze(1), n_fft, hop_length, win_length, return_complex=True, window=torch.hann_window(win_length).to(y_hat.device))
            
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
