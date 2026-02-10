
#!/usr/bin/env python3
"""
Train BitVocoder directly on 80-band Mel Spectrograms (STFT-only, No GAN).
(Strategy A3: High-speed STFT Training)

Focus:
- Speed (No Discriminator overhead)
- Stability (Loss = MultiResolutionSTFT + L1)
- Large Batch Size (32+)

Architecture:
    Mel (80-band) -> BitVocoder (SnakeBeta + ConvNeXt) -> Audio
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import glob
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Models
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss, WaveformL1Loss, SpectralFluxLoss

def plot_spectrogram(y, y_hat, save_path):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80, f_min=0, f_max=8000
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()
    
    # Original
    if y.dim() == 1: y = y.unsqueeze(0)
    spec = db_transform(mel_transform(y.cpu())).squeeze().numpy()
    axs[0].imshow(spec, aspect='auto', origin='lower', vmin=-80, vmax=20)
    axs[0].set_title("Real")
    
    # Generated
    if y_hat.dim() == 1: y_hat = y_hat.unsqueeze(0)
    spec_hat = db_transform(mel_transform(y_hat.cpu())).squeeze().numpy()
    axs[1].imshow(spec_hat, aspect='auto', origin='lower', vmin=-80, vmax=20)
    axs[1].set_title("Generated (STFT only)")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class AudioDataset(Dataset):
    def __init__(self, data_dir, segment_length=16000*2): # 2 seconds
        # Recursive search for wav files
        self.files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
        # Filter small files
        self.files = [f for f in self.files if os.path.getsize(f) > 32000] # Min 1 sec
        self.segment_length = segment_length
        print(f"Found {len(self.files)} files")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        try:
            import soundfile as sf
            wav, sr = sf.read(self.files[idx])
            wav = torch.tensor(wav, dtype=torch.float32)
            
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            
            if wav.ndim > 1:
                wav = wav.mean(dim=-1)
            
            if wav.shape[0] < self.segment_length:
                pad = self.segment_length - wav.shape[0]
                wav = F.pad(wav, (0, pad))
            else:
                start = random.randint(0, wav.shape[0] - self.segment_length)
                wav = wav[start:start+self.segment_length]
            
            # Normalize
            peak = wav.abs().max()
            if peak > 0:
                wav = wav / (peak + 1e-6) * 0.95
                
            return wav
        except Exception as e:
            return torch.zeros(self.segment_length)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/audio", help="Path to audio training data")
    parser.add_argument("--output_dir", default="checkpoints/vocoder_80band_stft")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32) # Increased batch size
    parser.add_argument("--lr", type=float, default=2e-4) # Standard LR
    parser.add_argument("--resume", default=None, help="Resume checkpoint")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    
    print("Initializing BitVocoder (80-band) [NO GAN]...")
    
    # BitVocoder Config
    vocoder = BitVocoder(
        input_dim=80,      # match 80-band mel
        dim=256,           
        n_fft=1024,
        hop_length=320,
        num_layers=4,
        num_res_blocks=1
    ).to(device)
    
    # Losses
    # 1. Multi-Resolution STFT Loss
    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[50, 120, 240],
        win_lengths=[240, 600, 1200]
    ).to(device)
    
    # 2. Spectral Flux (consistency)
    flux_loss = SpectralFluxLoss().to(device)
    
    # 3. Waveform L1 (phase alignment)
    l1_loss = WaveformL1Loss().to(device)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80, f_min=0, f_max=8000
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(vocoder.parameters(), lr=args.lr, betas=(0.8, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Compile
    print("  Compiling vocoder with torch.compile(mode='default')...")
    torch.set_float32_matmul_precision('high')
    vocoder = torch.compile(vocoder, mode='default')
    
    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        
        # Handle vocoder state dict key mismatch if loading from GAN training
        state_dict = ckpt['vocoder'] if 'vocoder' in ckpt else ckpt
        # Clean prefix if needed
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        # Partial load if needed
        dummy_state = vocoder.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in dummy_state and v.shape == dummy_state[k].shape}
        vocoder.load_state_dict(filtered_state, strict=False)
        
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                start_epoch = ckpt['epoch'] + 1
                pass
            except:
                print("  Optimizer state mismatch, restarting optimizer.")
        
        print(f"Resumed from {args.resume} (Start Epoch {start_epoch})")
    
    # Dataset
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    global_step = start_epoch * len(dataloader)
    
    print(f"Starting Training! Batch Size: {args.batch_size}")
    
    MEAN = -5.0
    STD = 3.5
    
    for epoch in range(start_epoch, args.epochs):
        vocoder.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        avg_stft = 0
        avg_flux = 0
        
        for wav in pbar:
            wav = wav.to(device)
            wav = wav.unsqueeze(1) # (B, 1, T)
            
            # 1. Compute Mel
            with torch.no_grad():
                mel = mel_transform(wav.squeeze(1))
                mel = torch.log(torch.clamp(mel, min=1e-5))
                mel_norm = (mel - MEAN) / STD
                mel_in = mel_norm.transpose(1, 2) # (B, T, C)
            
            # 2. Forward
            wav_fake = vocoder(mel_in)
            
            # 3. Match lengths
            min_len = min(wav.shape[2], wav_fake.shape[1])
            wav_real_crop = wav[:, :, :min_len]
            wav_fake_crop = wav_fake[:, :min_len].unsqueeze(1)
            
            # 4. Losses
            # STFT Loss
            sc_l, mag_l = stft_loss(wav_fake_crop.squeeze(1), wav_real_crop.squeeze(1))
            loss_stft = sc_l + mag_l
            
            # Auxiliary Losses
            loss_flux = flux_loss(wav_fake_crop.squeeze(1), wav_real_crop.squeeze(1))
            loss_l1 = l1_loss(wav_fake_crop.squeeze(1), wav_real_crop.squeeze(1))
            
            # Total Loss
            # Weighted: 1.0 STFT + 1.0 Flux + 45.0 L1 ? 
            # Usually STFT is dominant.
            loss_total = loss_stft + loss_flux + 10.0 * loss_l1
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            avg_stft += loss_stft.item()
            avg_flux += loss_flux.item()
            
            if global_step % 20 == 0:
                pbar.set_postfix({'STFT': f"{loss_stft.item():.3f}", 'Flux': f"{loss_flux.item():.3f}", 'L1': f"{loss_l1.item():.3f}"})
                writer.add_scalar("loss/stft", loss_stft.item(), global_step)
                writer.add_scalar("loss/flux", loss_flux.item(), global_step)
                writer.add_scalar("loss/l1", loss_l1.item(), global_step)
                writer.add_scalar("loss/total", loss_total.item(), global_step)
            
            if global_step % 1000 == 0:
                 img_path = os.path.join(args.output_dir, "spectrograms", f"step_{global_step}.png")
                 os.makedirs(os.path.dirname(img_path), exist_ok=True)
                 plot_spectrogram(wav_real_crop[0, 0].detach(), wav_fake_crop[0, 0].detach(), img_path)
                 
                 # Save audio sample
                 import soundfile as sf
                 sf.write(os.path.join(args.output_dir, "spectrograms", f"step_{global_step}.wav"), wav_fake_crop[0,0].detach().cpu().numpy(), 16000)
            
            global_step += 1
            
        scheduler.step()
        
        # Save Checkpoint
        save_path = os.path.join(args.output_dir, f"vocoder_last.pt")
        torch.save({
            'vocoder': vocoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)
        
        if epoch % 5 == 0:
             torch.save(vocoder.state_dict(), os.path.join(args.output_dir, f"vocoder_epoch{epoch}.pt"))
    
    writer.close()

if __name__ == "__main__":
    main()
