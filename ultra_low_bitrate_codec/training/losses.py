import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionSTFTLoss(nn.Module):
    """
    Computes STFT loss at multiple resolutions to capture both
    time (short-term) and frequency (long-term) details.
    Standard high-fidelity loss for non-GAN vocoders.
    """
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.losses.append(STFTLoss(fs, hs, wl, window))

    def forward(self, x, y):
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.losses:
            sc, mag = f(x, y)
            sc_loss += sc
            mag_loss += mag
        
        sc_loss /= len(self.losses)
        mag_loss /= len(self.losses)
        
        return sc_loss, mag_loss

class STFTLoss(nn.Module):
    def __init__(self, fft_size, hop_size, win_length, window):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.win_length = win_length
        self.window_name = window
        self.register_buffer("window_tensor", getattr(torch, window)(win_length))

    def forward(self, x, y):
        # x, y: (B, T)
        
        # Spectral Convergence Loss
        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, return_complex=True, window=self.window_tensor)
        y_stft = torch.stft(y, self.fft_size, self.hop_size, self.win_length, return_complex=True, window=self.window_tensor)
        
        x_mag = torch.clamp(torch.abs(x_stft), min=1e-7)
        y_mag = torch.clamp(torch.abs(y_stft), min=1e-7)
        
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        
        # Log Magnitude Loss
        mag_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        
        return sc_loss, mag_loss

def feature_matching_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
            
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


class WavLMPerceptualLoss(nn.Module):
    """
    Perceptual loss using WavLM features.
    More stable than GAN, captures semantic and acoustic quality.
    """
    def __init__(self, model_name="microsoft/wavlm-base", layers=[4, 8, 12], device='cuda'):
        super().__init__()
        from transformers import WavLMModel
        
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.layers = layers
        self.model.to(device)
        
    def extract_features(self, audio):
        """Extract features from multiple layers"""
        # audio: (B, T) at 16kHz
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        with torch.no_grad():
            outputs = self.model(audio, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        features = []
        for layer_idx in self.layers:
            if layer_idx < len(hidden_states):
                features.append(hidden_states[layer_idx])
        
        return features
    
    def forward(self, audio_real, audio_fake):
        """
        Compute perceptual loss between real and fake audio.
        """
        # Align lengths
        min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_len]
        audio_fake = audio_fake[..., :min_len]
        
        if audio_real.dim() == 3:
            audio_real = audio_real.squeeze(1)
        if audio_fake.dim() == 3:
            audio_fake = audio_fake.squeeze(1)
        
        # Extract features
        feats_real = self.extract_features(audio_real)
        feats_fake = self.extract_features(audio_fake.detach())  # Detach for stability
        
        # Compute L1 loss across layers
        loss = 0
        for fr, ff in zip(feats_real, feats_fake):
            # Align temporal dimension
            min_t = min(fr.shape[1], ff.shape[1])
            loss += F.l1_loss(ff[:, :min_t], fr[:, :min_t])
        
        return loss / len(feats_real)

