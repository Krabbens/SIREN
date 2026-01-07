#!/usr/bin/env python3
"""
Improved Training Script V2
Key improvements:
1. ResidualFSQ (RVQ)
2. VocosV2 with ResBlocks
3. Larger dimensions
4. Discriminator warmup
5. Multi-speaker support
"""
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import sys
import os
import argparse

sys.path.insert(0, '/home/sperm/diff')

from ultra_low_bitrate_codec.models.encoder_v2 import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers_v2 import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.decoder_v2 import SpeechDecoderV2
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from ultra_low_bitrate_codec.models.discriminator import HiFiGANDiscriminator
from ultra_low_bitrate_codec.data.feature_dataset import PrecomputedFeatureDataset
from ultra_low_bitrate_codec.training.losses import (
    MultiResolutionSTFTLoss, feature_matching_loss, 
    discriminator_loss, generator_loss
)

import torchaudio
import matplotlib.pyplot as plt
import numpy as np


class MelSpecComputation(nn.Module):
    """Cached mel spectrogram computation"""
    def __init__(self, n_fft=1024, num_mels=80, sampling_rate=16000, 
                 hop_size=256, win_size=1024):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate, n_fft=n_fft, n_mels=num_mels,
            hop_length=hop_size, win_length=win_size
        )
    
    def forward(self, y):
        with torch.amp.autocast('cuda', enabled=False):
            y = y.float()
            self.mel = self.mel.to(y.device)
            spec = self.mel(y)
            return torch.log(torch.clamp(spec, min=1e-5))


class ImprovedTrainer:
    def __init__(self, config_path: str, feature_dir: str, checkpoint_dir: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(f"{checkpoint_dir}/spectrograms", exist_ok=True)
        os.makedirs(f"{checkpoint_dir}/audio_samples", exist_ok=True)
        
        torch.backends.cudnn.benchmark = True
        
        # ========================================
        # MODELS V2
        # ========================================
        print("🔧 Initializing improved models...")
        
        # Encoder (Factorizer)
        self.factorizer = InformationFactorizerV2(self.config).to(self.device)
        
        # Quantizers - ResidualFSQ
        levels = self.config['model']['fsq_levels']
        num_rfsq_levels = self.config['model'].get('rfsq_num_levels', 4)
        sem_dim = self.config['model']['semantic']['output_dim']
        pro_dim = self.config['model']['prosody']['output_dim']
        
        self.sem_rfsq = ResidualFSQ(
            levels=levels, 
            num_levels=num_rfsq_levels,
            input_dim=sem_dim
        ).to(self.device)
        
        self.pro_rfsq = ResidualFSQ(
            levels=levels,
            num_levels=num_rfsq_levels,
            input_dim=pro_dim
        ).to(self.device)
        
        self.spk_pq = ProductQuantizer(
            input_dim=self.config['model']['speaker']['embedding_dim'],
            num_groups=self.config['model']['speaker']['num_groups'],
            codes_per_group=self.config['model']['speaker']['codes_per_group']
        ).to(self.device)
        
        # Decoder + Vocoder V2
        self.decoder = SpeechDecoderV2(self.config).to(self.device)
        
        # Entropy model
        self.entropy_model = EntropyModel(self.config).to(self.device)
        
        # Discriminator
        self.discriminator = HiFiGANDiscriminator().to(self.device)
        
        # ========================================
        # OPTIMIZERS
        # ========================================
        self.gen_params = (
            list(self.factorizer.parameters()) +
            list(self.sem_rfsq.parameters()) +
            list(self.pro_rfsq.parameters()) +
            list(self.spk_pq.parameters()) +
            list(self.decoder.parameters()) +
            list(self.entropy_model.parameters())
        )
        
        self.optimizer_g = optim.AdamW(
            self.gen_params, 
            lr=float(self.config['training']['learning_rate']),
            betas=(0.8, 0.99)
        )
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=float(self.config['training']['learning_rate']),
            betas=(0.8, 0.99)
        )
        
        # Cosine annealing scheduler
        max_steps = self.config['training']['max_steps']
        self.scheduler_g = CosineAnnealingLR(self.optimizer_g, T_max=max_steps, eta_min=1e-6)
        self.scheduler_d = CosineAnnealingLR(self.optimizer_d, T_max=max_steps, eta_min=1e-6)
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda')
        
        # ========================================
        # LOSSES
        # ========================================
        self.mr_stft = MultiResolutionSTFTLoss().to(self.device)
        self.mel_fn = MelSpecComputation().to(self.device)
        
        # ========================================
        # DATA
        # ========================================
        self.train_ds = PrecomputedFeatureDataset(
            feature_dir=feature_dir,
            manifest_path=self.config['data']['train_manifest'],
            max_frames=150  # ~3s audio
        )
        
        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=True
        )
        
        # Training state
        self.step = 0
        self.disc_start_step = self.config['training'].get('discriminator_start_step', 5000)
        
        print(f"✅ Models initialized")
        print(f"   📊 Dataset: {len(self.train_ds)} samples")
        print(f"   🎯 Max steps: {max_steps}")
        print(f"   ⏳ Disc warmup: {self.disc_start_step} steps")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.gen_params)
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"   📐 Generator params: {total_params/1e6:.2f}M")
        print(f"   📐 Discriminator params: {disc_params/1e6:.2f}M")
    
    def train_step(self, batch):
        features, audio = batch
        features = features.to(self.device)
        audio = audio.to(self.device)
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            print(f"⚠️ NaN/Inf in target audio! Skipping batch.")
            return None

        # ========================================
        # GENERATOR FORWARD
        # ========================================
        with torch.amp.autocast('cuda'):
            # Check inputs - moved before audio check, but check features too
            if torch.isnan(features).any() or torch.isinf(features).any():
                print(f"⚠️ NaN in input features! Skipping batch.")
                return None
            
            # Factorize
            sem, pro, spk = self.factorizer(features)
            if torch.isnan(sem).any() or torch.isnan(pro).any():
                print(f"⚠️ NaN in factorizer output!")
                return None
            
            # Quantize with ResidualFSQ
            sem_z, sem_loss, sem_idx = self.sem_rfsq(sem)
            pro_z, pro_loss, pro_idx = self.pro_rfsq(pro)
            spk_z, spk_loss, spk_idx = self.spk_pq(spk)
            
            vq_loss = sem_loss + pro_loss + spk_loss
            
            # Decode
            audio_hat = self.decoder(sem_z, pro_z, spk_z)
            
            # Force float32 for audio output to ensure stable loss calc
            audio_hat = audio_hat.float()
            
            if torch.isnan(audio_hat).any():
                print(f"⚠️ NaN in decoder output (audio_hat)!")
                return None
            
            if audio_hat.dim() == 2:
                audio_hat = audio_hat.unsqueeze(1)
            
            # Match lengths
            min_len = min(audio.shape[2], audio_hat.shape[2])
            audio = audio[..., :min_len]
            audio_hat = audio_hat[..., :min_len]
        
        # ========================================
        # DISCRIMINATOR STEP (after warmup)
        # ========================================
        loss_d = torch.tensor(0.0, device=self.device)
        
        if self.step >= self.disc_start_step:
            self.optimizer_d.zero_grad()
            
            with torch.amp.autocast('cuda'):
                mpd_res_d, mrd_res_d = self.discriminator(audio, audio_hat.detach())
                loss_d_mpd, _, _ = discriminator_loss(mpd_res_d[0], mpd_res_d[1])
                loss_d_mrd, _, _ = discriminator_loss(mrd_res_d[0], mrd_res_d[1])
                loss_d = loss_d_mpd + loss_d_mrd
            
            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.optimizer_d)
        
        # ========================================
        # GENERATOR STEP
        # ========================================
        self.optimizer_g.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # Reconstruction losses
            orig_spec = self.mel_fn(audio.squeeze(1))
            recon_spec = self.mel_fn(audio_hat.squeeze(1))
            mel_loss = F.l1_loss(recon_spec, orig_spec) * self.config['training'].get('mel_weight', 45.0)
            
            sc_loss, mag_loss = self.mr_stft(audio.squeeze(1), audio_hat.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * self.config['training'].get('stft_weight', 2.0)
            
            # GAN losses (after warmup)
            loss_gen_gan = torch.tensor(0.0, device=self.device)
            loss_fm = torch.tensor(0.0, device=self.device)
            
            if self.step >= self.disc_start_step:
                mpd_res, mrd_res = self.discriminator(audio, audio_hat)
                
                loss_g_mpd, _ = generator_loss(mpd_res[1])
                loss_g_mrd, _ = generator_loss(mrd_res[1])
                loss_gen_gan = (loss_g_mpd + loss_g_mrd) * self.config['training'].get('gan_weight', 1.0)
                
                loss_fm_mpd = feature_matching_loss(mpd_res[2], mpd_res[3])
                loss_fm_mrd = feature_matching_loss(mrd_res[2], mrd_res[3])
                loss_fm = (loss_fm_mpd + loss_fm_mrd) * self.config['training'].get('fm_weight', 2.0)
            
            # Entropy loss
            # Note: sem_idx and pro_idx now have shape (B, T, num_levels)
            # We use the first level for entropy estimation
            sem_bits, pro_bits = self.entropy_model.estimate_bits(
                sem_idx[..., 0] if sem_idx.dim() > 2 else sem_idx,
                pro_idx[..., 0] if pro_idx.dim() > 2 else pro_idx
            )
            entropy_loss = (sem_bits.mean() + pro_bits.mean()) * 0.01
            
            # Total loss
            loss_g = vq_loss + mel_loss + stft_loss + loss_gen_gan + loss_fm + entropy_loss
            
            # Bitrate calculation
            sem_bits_per_frame = self.sem_rfsq.get_total_bits_per_frame()
            pro_bits_per_frame = self.pro_rfsq.get_total_bits_per_frame()
            T_sem = sem.shape[1]
            T_pro = pro.shape[1]
            duration = min_len / 16000.0
            total_bits = T_sem * sem_bits_per_frame + T_pro * pro_bits_per_frame + 64
            bps = total_bits / (duration * features.shape[0] + 1e-6)
        
        self.scaler.scale(loss_g).backward()
        
        # Gradient Clipping
        self.scaler.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.gen_params, max_norm=1.0)
        
        # Step with check
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        
        # Check for NaNs
        if torch.isnan(loss_g) or torch.isinf(loss_g):
             print(f"⚠️ NaN detected at step {self.step}!")
             print(f"   loss_g: {loss_g.item()}")
             print(f"   mel: {mel_loss.item()}")
             print(f"   stft: {stft_loss.item()} (sc={sc_loss.item()}, mag={mag_loss.item()})")
             print(f"   vq: {vq_loss.item()}")
             print(f"   entropy: {entropy_loss.item()}")
             print(f"FATAL ERROR: Exiting to avoid infinite loop.")
             sys.exit(1)
        
        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item() if self.step >= self.disc_start_step else 0,
            "mel": mel_loss.item(),
            "stft": stft_loss.item(),
            "vq": vq_loss.item(),
            "gan": loss_gen_gan.item() if self.step >= self.disc_start_step else 0,
            "fm": loss_fm.item() if self.step >= self.disc_start_step else 0,
            "bps": bps,
            "lr": self.scheduler_g.get_last_lr()[0]
        }
    
    def save_checkpoint(self):
        """Save all model checkpoints"""
        step = self.step
        ckpt_dir = self.checkpoint_dir
        
        torch.save(self.factorizer.state_dict(), f"{ckpt_dir}/factorizer_{step}.pt")
        torch.save(self.sem_rfsq.state_dict(), f"{ckpt_dir}/sem_rfsq_{step}.pt")
        torch.save(self.pro_rfsq.state_dict(), f"{ckpt_dir}/pro_rfsq_{step}.pt")
        torch.save(self.spk_pq.state_dict(), f"{ckpt_dir}/spk_pq_{step}.pt")
        torch.save(self.decoder.state_dict(), f"{ckpt_dir}/decoder_{step}.pt")
        torch.save(self.discriminator.state_dict(), f"{ckpt_dir}/discriminator_{step}.pt")
        torch.save(self.entropy_model.state_dict(), f"{ckpt_dir}/entropy_{step}.pt")
    
    def save_visualizations(self):
        """Generate and save spectrograms"""
        self.factorizer.eval()
        self.decoder.eval()
        
        step = self.step
        
        with torch.no_grad():
            batch = next(iter(self.train_dl))
            features, audio = batch
            features = features[:1].to(self.device)
            audio = audio[:1].to(self.device)
            
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            
            # Forward
            sem, pro, spk = self.factorizer(features)
            sem_z, _, _ = self.sem_rfsq(sem)
            pro_z, _, _ = self.pro_rfsq(pro)
            spk_z, _, _ = self.spk_pq(spk)
            
            # Quantized reconstruction
            audio_hat_q = self.decoder(sem_z, pro_z, spk_z)
            
            # Oracle reconstruction (no quantization)
            audio_hat_c = self.decoder(sem, pro, spk_z)
            
            if audio_hat_q.dim() == 2:
                audio_hat_q = audio_hat_q.unsqueeze(1)
            if audio_hat_c.dim() == 2:
                audio_hat_c = audio_hat_c.unsqueeze(1)
            
            # Prepare for plotting
            orig = audio[0, 0].cpu()
            min_len = min(orig.shape[0], audio_hat_q.shape[2], audio_hat_c.shape[2])
            recon_q = audio_hat_q[0, 0, :min_len].cpu()
            recon_c = audio_hat_c[0, 0, :min_len].cpu()
            orig = orig[:min_len]
            
            # Compute spectrograms
            orig_spec = torch.log(torch.clamp(
                torch.abs(torch.stft(orig, n_fft=1024, return_complex=True)), min=1e-5
            )).numpy()
            q_spec = torch.log(torch.clamp(
                torch.abs(torch.stft(recon_q, n_fft=1024, return_complex=True)), min=1e-5
            )).numpy()
            c_spec = torch.log(torch.clamp(
                torch.abs(torch.stft(recon_c, n_fft=1024, return_complex=True)), min=1e-5
            )).numpy()
            
            # Plot
            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            
            # Waveforms
            axes[0, 0].plot(orig.numpy(), linewidth=0.3, color='blue')
            axes[0, 0].set_title(f"Original (Step {step})")
            axes[0, 0].set_ylabel("Amp")
            
            axes[1, 0].plot(recon_c.numpy(), linewidth=0.3, color='green')
            axes[1, 0].set_title("Continuous (Oracle)")
            axes[1, 0].set_ylabel("Amp")
            
            axes[2, 0].plot(recon_q.numpy(), linewidth=0.3, color='orange')
            axes[2, 0].set_title("Quantized")
            axes[2, 0].set_ylabel("Amp")
            axes[2, 0].set_xlabel("Sample")
            
            # Spectrograms
            axes[0, 1].imshow(orig_spec, origin='lower', aspect='auto', vmin=-10, vmax=5)
            axes[0, 1].set_title("Original Spectrogram")
            
            axes[1, 1].imshow(c_spec, origin='lower', aspect='auto', vmin=-10, vmax=5)
            axes[1, 1].set_title("Continuous Spectrogram")
            
            axes[2, 1].imshow(q_spec, origin='lower', aspect='auto', vmin=-10, vmax=5)
            axes[2, 1].set_title("Quantized Spectrogram")
            axes[2, 1].set_xlabel("Time")
            
            plt.suptitle(f"Ultra-Low Bitrate Codec V2 - Step {step}", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.checkpoint_dir}/spectrograms/step_{step}.png", dpi=150)
            plt.close()
            
            # Save audio
            torchaudio.save(f"{self.checkpoint_dir}/audio_samples/step_{step}_orig.wav",
                          orig.unsqueeze(0), 16000)
            torchaudio.save(f"{self.checkpoint_dir}/audio_samples/step_{step}_quant.wav",
                          recon_q.unsqueeze(0), 16000)
            torchaudio.save(f"{self.checkpoint_dir}/audio_samples/step_{step}_cont.wav",
                          recon_c.unsqueeze(0), 16000)
        
        self.factorizer.train()
        self.decoder.train()
    
    def train(self):
        max_steps = self.config['training']['max_steps']
        log_every = self.config['training'].get('log_every', 100)
        save_every = self.config['training'].get('save_every', 500)
        
        pbar = tqdm(total=max_steps, desc="Training V2")
        log_file = open(f"{self.checkpoint_dir}/training.log", "w")
        
        while self.step < max_steps:
            for batch in self.train_dl:
                metrics = self.train_step(batch)
                
                if metrics is None:
                    continue  # Skip step on NaN
                
                self.step += 1
                self.scheduler_g.step()
                self.scheduler_d.step()
                
                pbar.update(1)
                pbar.set_postfix({
                    "g": f"{metrics['loss_g']:.2f}",
                    "d": f"{metrics['loss_d']:.2f}",
                    "mel": f"{metrics['mel']:.1f}",
                    "bps": f"{metrics['bps']:.0f}"
                })
                
                if self.step % log_every == 0:
                    log_line = (f"Step {self.step}: loss_g={metrics['loss_g']:.3f}, "
                               f"mel={metrics['mel']:.3f}, stft={metrics['stft']:.3f}, "
                               f"gan={metrics['gan']:.3f}, bps={metrics['bps']:.1f}\n")
                    log_file.write(log_line)
                    log_file.flush()
                    print(f"\n{log_line.strip()}")
                
                if self.step % save_every == 0:
                    self.save_checkpoint()
                    self.save_visualizations()
                
                if self.step >= max_steps:
                    break
        
        pbar.close()
        log_file.close()
        
        # Final save
        self.save_checkpoint()
        print(f"\n✅ Training complete! Checkpoints saved to {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Improved Codec Training V2")
    parser.add_argument('--config', type=str, 
                       default='/home/sperm/diff/ultra_low_bitrate_codec/configs/improved.yaml',
                       help='Path to config file')
    parser.add_argument('--features', type=str,
                       default='/home/sperm/diff/data/features_train',
                       help='Path to precomputed features')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='/home/sperm/diff/checkpoints_v2',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    trainer = ImprovedTrainer(args.config, args.features, args.checkpoint_dir)
    trainer.train()


if __name__ == "__main__":
    main()
