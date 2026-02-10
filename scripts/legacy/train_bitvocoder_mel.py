#!/usr/bin/env python3
"""
Train BitVocoder directly on 80-band Mel Spectrograms.
(Strategy A2: Retrain BitVocoder on 80-band, iSTFT)

Architecture:
    Mel (80-band) -> BitVocoder (Input=80) -> Audio

Data:
    Loads random audio segments, computes GT Mel on-the-fly.
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
from ultra_low_bitrate_codec.models.discriminator import HiFiGANDiscriminator
from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss

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
    axs[1].set_title("Generated")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
            # print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(self.segment_length)

def feature_loss(fmap_r, fmap_g):
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
        dr = dr.float()
        dg = dg.float()
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
        dg = dg.float()
        l = torch.mean((1-dg)**2)
        loss += l
        gen_losses.append(l.item())
    return loss, gen_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/audio", help="Path to audio training data")
    parser.add_argument("--output_dir", default="checkpoints/vocoder_80band")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d_lr", type=float, default=2e-4) 
    parser.add_argument("--resume", default=None, help="Resume checkpoint")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    
    print("Initializing BitVocoder (80-band)...")
    
    # BitVocoder Config
    # Match Flow Matching output: 80 bands
    vocoder = BitVocoder(
        input_dim=80,      # << 80-band input
        dim=256,           # Compact dimension
        n_fft=1024,
        hop_length=320,
        num_layers=4,
        num_res_blocks=1
    ).to(device)
    
    discriminator = HiFiGANDiscriminator().to(device)
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80, f_min=0, f_max=8000
    ).to(device)
    
    # Optimizers
    opt_g = torch.optim.AdamW(vocoder.parameters(), lr=args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr, betas=(0.8, 0.99))
    
    # Compile
    print("  Compiling models with torch.compile(mode='default')...")
    torch.set_float32_matmul_precision('high')
    vocoder = torch.compile(vocoder, mode='default')
    discriminator = torch.compile(discriminator, mode='default')
    
    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        vocoder.load_state_dict(ckpt['vocoder'])
        discriminator.load_state_dict(ckpt['discriminator'])
        opt_g.load_state_dict(ckpt['opt_g'])
        opt_d.load_state_dict(ckpt['opt_d'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from {args.resume} (Epoch {start_epoch})")
    
    # Dataset
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    
    global_step = start_epoch * len(dataloader)
    
    print("Starting BitVocoder 80-band Training!")
    
    for epoch in range(start_epoch, args.epochs):
        vocoder.train()
        discriminator.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for wav in pbar:
            wav = wav.to(device)
            wav = wav.unsqueeze(1) # (B, 1, T)
            
            # 1. Compute GT Mel
            with torch.no_grad():
                # Consistent log-mel creation
                mel = mel_transform(wav.squeeze(1))
                mel = torch.log(torch.clamp(mel, min=1e-5)) # (B, 80, T_mel)
                
                # Normalize (Standard Scaling for BitNet input distribution)
                # Matches E2E Training mean/std
                MEAN = -5.0
                STD = 3.5
                mel_norm = (mel - MEAN) / STD
                
                # BitVocoder expects (B, T, C)
                mel_in = mel_norm.transpose(1, 2)
            
            # --- 2. Train Discriminator ---
            if epoch >= 1: # Warmup D slightly later
                # Forward G (Detach)
                wav_fake = vocoder(mel_in) # (B, T_fake)
                
                # Match lengths (vocoder might produce slightly different length due to iSTFT padding)
                min_len = min(wav.shape[2], wav_fake.shape[1])
                wav_real_crop = wav[:, :, :min_len]
                wav_fake_crop = wav_fake[:, :min_len].unsqueeze(1)
                
                mpd_real, mrd_real = discriminator(wav_real_crop, wav_real_crop)
                mpd_fake, mrd_fake = discriminator(wav_real_crop, wav_fake_crop.detach())
                
                loss_d_mpd, _, _ = discriminator_loss(mpd_real[0], mpd_fake[0])
                loss_d_mrd, _, _ = discriminator_loss(mrd_real[1], mrd_fake[1])
                loss_d = loss_d_mpd + loss_d_mrd
                
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
            else:
                wav_fake = vocoder(mel_in)
                min_len = min(wav.shape[2], wav_fake.shape[1])
                wav_real_crop = wav[:, :, :min_len]
                wav_fake_crop = wav_fake[:, :min_len].unsqueeze(1)
                loss_d = torch.tensor(0.0, device=device)
            
            # --- 3. Train Generator ---
            # Re-run Disc for FM loss
            mpd_res, mrd_res = discriminator(wav_real_crop, wav_fake_crop)
            
            # Adv Loss
            loss_g_mpd, _ = generator_loss(mpd_res[0])
            loss_g_mrd, _ = generator_loss(mrd_res[1])
            loss_adv = loss_g_mpd + loss_g_mrd
            
            # FM Loss
            loss_fm = feature_loss(mpd_res[2], mpd_res[3]) + feature_loss(mrd_res[2], mrd_res[3])
            
            # STFT Loss
            sc_loss, mag_loss = stft_loss(wav_fake_crop.squeeze(1), wav_real_crop.squeeze(1))
            loss_stft = (sc_loss + mag_loss)
            
            # Weighted Sum
            if epoch < 5:
                 # Initial stabilization: Focus on STFT, less on Adv/FM
                 loss_g_total = 45.0 * loss_stft + 1.0 * loss_adv + 1.0 * loss_fm
            else:
                 loss_g_total = 20.0 * loss_stft + 2.0 * loss_adv + 2.0 * loss_fm
            
            opt_g.zero_grad()
            loss_g_total.backward()
            opt_g.step()
            
            # Logs
            if global_step % 20 == 0:
                pbar.set_postfix({'D': f"{loss_d.item():.3f}", 'G': f"{loss_g_total.item():.3f}", 'STFT': f"{loss_stft.item():.3f}"})
                writer.add_scalar("loss/d_total", loss_d.item(), global_step)
                writer.add_scalar("loss/g_total", loss_g_total.item(), global_step)
                writer.add_scalar("loss/g_stft", loss_stft.item(), global_step)
            
            if global_step % 500 == 0:
                img_path = os.path.join(args.output_dir, "spectrograms", f"step_{global_step}.png")
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                plot_spectrogram(wav_real_crop[0, 0].detach(), wav_fake_crop[0, 0].detach(), img_path)
            
            global_step += 1
            
        # Save Checkpoint
        save_path = os.path.join(args.output_dir, f"bitvocoder_last.pt")
        torch.save({
            'vocoder': vocoder.state_dict(),
            'discriminator': discriminator.state_dict(),
            'opt_g': opt_g.state_dict(),
            'opt_d': opt_d.state_dict(),
            'epoch': epoch
        }, save_path)
        
        if epoch % 5 == 0:
             torch.save(vocoder.state_dict(), os.path.join(args.output_dir, f"bitvocoder_epoch{epoch}.pt"))
             
    writer.close()

if __name__ == "__main__":
    main()
