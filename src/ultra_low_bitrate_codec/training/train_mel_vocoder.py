import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
from tqdm import tqdm
import argparse
import glob
import numpy as np
import sys

# Force torchaudio to use soundfile backend to avoid libtorchcodec/FFmpeg hangs
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window='hann_window'):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window

    def forward(self, x, y):
        # x, y: (B, T)
        loss = 0
        for f, h, w in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = getattr(torch, self.window)(w).to(x.device)
            sc_x = torch.stft(x, f, h, w, window, return_complex=True).abs().clamp(min=1e-7)
            sc_y = torch.stft(y, f, h, w, window, return_complex=True).abs().clamp(min=1e-7)
            
            # Spectral convergence loss (Per-sample ratio)
            # sc_x, sc_y: (B, F, T)
            sc_x_flat = sc_x.reshape(x.shape[0], -1)
            sc_y_flat = sc_y.reshape(y.shape[0], -1)
            
            norm_diff = torch.norm(sc_y_flat - sc_x_flat, p="fro", dim=1)
            norm_y = torch.norm(sc_y_flat, p="fro", dim=1)
            
            # Safe ratio to avoid explosion on silent ground truth
            sc_loss = norm_diff / (norm_y + 1e-3)
            
            # Log STFT magnitude loss
            log_mag_loss = F.l1_loss(torch.log(sc_x), torch.log(sc_y))
            loss += sc_loss.mean() + log_mag_loss
            
        return loss / len(self.fft_sizes)


class MelVocoderDataset(Dataset):
    def __init__(self, audio_dir, segment_size=16000):
        self.files = glob.glob(os.path.join(audio_dir, "**/*.wav"), recursive=True)
        self.segment_size = segment_size
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            import soundfile as sf
            wav, sr = sf.read(self.files[idx])
            wav = torch.from_numpy(wav).float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.transpose(0, 1) # (C, T)
            
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            
            if wav.shape[-1] < self.segment_size:
                wav = F.pad(wav, (0, self.segment_size - wav.shape[-1]))
            else:
                start = torch.randint(0, wav.shape[-1] - self.segment_size + 1, (1,)).item()
                wav = wav[:, start:start+self.segment_size]
            
            # Explicit normalization and debug
            max_val = wav.abs().max().item()
            if max_val > 1.0:
                wav = wav / max_val
            elif max_val == 0:
                # Avoid black hole
                wav = wav + torch.randn_like(wav) * 1e-6
                
            # Target Mel
            mel = self.mel_spec(wav) # (1, 80, T_mel)
            log_mel = torch.log10(torch.clamp(mel, min=1e-5))
            
            return wav.squeeze(0), log_mel.squeeze(0).transpose(0, 1)
        except Exception as e:
            # We don't have log_print here, just use print
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(self.segment_size), torch.zeros(51, 80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="data/audio")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def log_print(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        sys.stdout.flush()
    
    os.makedirs("checkpoints/vocoder_mel", exist_ok=True)
    log_file = open("checkpoints/vocoder_mel/train.log", "a")

    log_print(f"Using device: {device}")
    
    log_print("Initializing model...")
    model = MelVocoderBitNet().to(device)
    if args.resume and os.path.exists(args.resume):
        log_print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    log_print("Initializing dataset...")
    dataset = MelVocoderDataset(args.audio_dir)
    log_print(f"Found {len(dataset)} files.")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    log_print("--- MelVocoder Training Started ---")

    for epoch in range(args.epochs):
        model.train()
        last_wav_pred = None
        
        # Simple loop without tqdm for background logging cleanliness
        for i, (wav, mel) in enumerate(loader):
            wav, mel = wav.to(device), mel.to(device)
            # Skip silent batches to save time
            if wav.abs().max() < 1e-4:
                continue
            pred_wav = model(mel)
            
            min_len = min(wav.shape[1], pred_wav.shape[1])
            wav_target = wav[:, :min_len]
            wav_pred = pred_wav[:, :min_len]
            
            # Hybrid Loss
            loss_stft = stft_loss(wav_pred, wav_target)
            loss_wav = F.l1_loss(wav_pred, wav_target)
            
            loss = loss_stft + 50.0 * loss_wav
            
            
            if torch.isnan(loss):
                log_print("NaN detected! Emergency skip.")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Simple logging for every batch
            pred_max = wav_pred.abs().max().item()
            msg = f"Vocoder Epoch {epoch} | Batch {i+1}/{len(loader)} | STFT: {loss_stft.item():.4f} | Pred Max: {pred_max:.4f}"
            log_print(msg)
            
            last_wav_pred = wav_pred

        log_print(f"Epoch {epoch} finished.")
        torch.save(model.state_dict(), f"checkpoints/vocoder_mel/vocoder_latest.pt")
        
        if epoch % 1 == 0 and last_wav_pred is not None:
            import soundfile as sf
            os.makedirs("outputs/vocoder_samples", exist_ok=True)
            sf.write(f"outputs/vocoder_samples/epoch_{epoch}.wav", last_wav_pred[0].detach().cpu().numpy(), 16000)

if __name__ == "__main__":
    main()
