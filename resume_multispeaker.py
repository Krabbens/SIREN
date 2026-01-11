#!/usr/bin/env python3
"""
Resume training from checkpoint for Multi-Speaker Codec
"""
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import yaml
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/sperm/diff')

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.discriminator import HiFiGANDiscriminator
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from ultra_low_bitrate_codec.data.feature_dataset import PrecomputedFeatureDataset
from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss, discriminator_loss, generator_loss

# Configuration
CONFIG_PATH = "/home/sperm/diff/ultra_low_bitrate_codec/configs/multispeaker.yaml"
CHECKPOINT_DIR = "/home/sperm/diff/checkpoints_multispeaker"
RESUME_STEP = 20000  # Resume from this step

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

print("="*70)
print(f"🚀 RESUMING MULTI-SPEAKER CODEC FROM STEP {RESUME_STEP}")
print("="*70)

# Models
factorizer = InformationFactorizerV2(config).to(device)
decoder = SpeechDecoderV2(config).to(device)

fsq_levels = config['model']['fsq_levels']
sem_dim = config['model']['semantic']['output_dim']
pro_dim = config['model']['prosody']['output_dim']

sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=4, input_dim=sem_dim).to(device)
pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=4, input_dim=pro_dim).to(device)
spk_pq = ProductQuantizer(
    input_dim=config['model']['speaker']['embedding_dim'],
    num_groups=config['model']['speaker']['num_groups'],
    codes_per_group=config['model']['speaker']['codes_per_group']
).to(device)
entropy_model = EntropyModel(config).to(device)
discriminator = HiFiGANDiscriminator().to(device)

# Load checkpoints
print("Loading checkpoints...")
factorizer.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/factorizer_{RESUME_STEP}.pt", map_location=device))
decoder.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/decoder_{RESUME_STEP}.pt", map_location=device))
sem_vq.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/sem_rfsq_{RESUME_STEP}.pt", map_location=device))
pro_vq.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/pro_rfsq_{RESUME_STEP}.pt", map_location=device))
spk_pq.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/spk_pq_{RESUME_STEP}.pt", map_location=device))
entropy_model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/entropy_{RESUME_STEP}.pt", map_location=device))
discriminator.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/discriminator_{RESUME_STEP}.pt", map_location=device))
print("Checkpoints loaded!")

# Optimizers
params = list(factorizer.parameters()) + list(decoder.parameters()) + \
         list(sem_vq.parameters()) + list(pro_vq.parameters()) + \
         list(spk_pq.parameters()) + list(entropy_model.parameters())

optimizer = optim.AdamW(params, lr=1e-4, betas=(0.8, 0.99)) # Reduced LR for stability
optimizer_d = optim.AdamW(discriminator.parameters(), lr=float(config['training']['learning_rate']), betas=(0.8, 0.99))

max_steps = config['training']['max_steps']
disc_start_step = 20000 # config['training'].get('discriminator_start_step', 5000) - Postponed to avoid hang/collapse

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

# Simple cosine annealing for resume (warmup already done)
remaining_steps = max_steps - RESUME_STEP
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_steps)
scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=remaining_steps)

# Data
train_ds = PrecomputedFeatureDataset(
    feature_dir=config['data']['feature_dir'],
    manifest_path=config['data']['train_manifest'],
    max_frames=500
)
val_ds = PrecomputedFeatureDataset(
    feature_dir=config['data']['feature_dir'],
    manifest_path=config['data']['val_manifest'],
    max_frames=500
)
train_dl = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True,
                      num_workers=config['training']['num_workers'], pin_memory=True,
                      persistent_workers=True, prefetch_factor=4)

# Losses
mr_stft = MultiResolutionSTFTLoss().to(device)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256, win_length=1024
).to(device)

def mel_fn(y):
    return torch.log(mel_transform(y).clamp(min=1e-5))

def validate(step):
    factorizer.eval()
    decoder.eval()
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=2)
    total_mel, count = 0, 0
    with torch.no_grad():
        for features, audio in val_dl:
            features, audio = features.to(device), audio.to(device)
            if audio.dim() == 2: audio = audio.unsqueeze(1)
            sem, pro, spk = factorizer(features)
            sem_z, _, _ = sem_vq(sem)
            pro_z, _, _ = pro_vq(pro)
            spk_z, _, _ = spk_pq(spk)
            audio_hat = decoder(sem_z, pro_z, spk_z)
            if audio_hat.dim() == 2: audio_hat = audio_hat.unsqueeze(1)
            min_len = min(audio.shape[-1], audio_hat.shape[-1])
            mel_loss = F.l1_loss(mel_fn(audio[..., :min_len]), mel_fn(audio_hat[..., :min_len]))
            total_mel += mel_loss.item()
            count += 1
            if count >= 10: break
    factorizer.train()
    decoder.train()
    return total_mel / count if count > 0 else 0

# Training
print(f"\n🎯 Resuming from step {RESUME_STEP}...")
scaler = torch.amp.GradScaler('cuda')
steps = RESUME_STEP
pbar = tqdm(total=max_steps - RESUME_STEP, desc=f"Training from {RESUME_STEP}")

log_file = open(f"{CHECKPOINT_DIR}/training.log", "a")
log_file.write(f"\n=== Resumed from step {RESUME_STEP} ===\n")

while steps < max_steps:
    for batch_idx, batch in enumerate(train_dl):
        features, audio = batch
        features, audio = features.to(device), audio.to(device)
        if audio.dim() == 2: audio = audio.unsqueeze(1)
        
        # Run Factorizer in FP32 to avoid overflow/Inf
        with torch.amp.autocast('cuda', enabled=False):
            sem, pro, spk = factorizer(features.float())
            
        with torch.amp.autocast('cuda'):
            sem_z, sem_loss, _ = sem_vq(sem)
            pro_z, pro_loss, _ = pro_vq(pro)
            spk_z, spk_loss, _ = spk_pq(spk)
            audio_hat = decoder(sem_z, pro_z, spk_z)
            if audio_hat.dim() == 2: audio_hat = audio_hat.unsqueeze(1)
            min_len = min(audio.shape[-1], audio_hat.shape[-1])
            audio, audio_hat = audio[..., :min_len], audio_hat[..., :min_len]
        
        # Discriminator
        if steps >= disc_start_step:
            optimizer_d.zero_grad()
            with torch.amp.autocast('cuda'):
                mpd_res, mrd_res = discriminator(audio, audio_hat.detach())
                loss_d_mpd, _, _ = discriminator_loss(mpd_res[0], mpd_res[1])
                loss_d_mrd, _, _ = discriminator_loss(mrd_res[0], mrd_res[1])
                loss_d = loss_d_mpd + loss_d_mrd
            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
        else:
            loss_d = torch.tensor(0.0)
        
        # Generator
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            if steps >= disc_start_step:
                mpd_res, mrd_res = discriminator(audio, audio_hat)
                loss_g_mpd, _ = generator_loss(mpd_res[1])
                loss_g_mrd, _ = generator_loss(mrd_res[1])
                loss_gan = (loss_g_mpd + loss_g_mrd) * config['training'].get('gan_weight', 1.0)
            else:
                loss_gan = torch.tensor(0.0, device=device)
            
            sc_loss, mag_loss = mr_stft(audio_hat.squeeze(1), audio.squeeze(1))
            stft_loss = sc_loss + mag_loss
            mel_loss = F.l1_loss(mel_fn(audio), mel_fn(audio_hat))
            q_loss = sem_loss + pro_loss + spk_loss
            
            bits_sem = sem_vq.get_total_bits_per_frame() * sem.shape[1]
            bits_pro = pro_vq.get_total_bits_per_frame() * pro.shape[1]
            bps = (bits_sem + bits_pro + 32) / (audio.shape[-1] / 16000)
            
            loss_g = (stft_loss * 2.0 + mel_loss * 45.0 + q_loss * 0.25 + loss_gan)
        
        # NaN check
        if torch.isnan(loss_g):
            print(f"⚠️ NaN detected at step {steps}. Skipping batch.")
            optimizer.zero_grad()
            continue

        scaler.scale(loss_g).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        scheduler_d.step()
        
        steps += 1
        pbar.update(1)
        
        # Enhanced logging
        if steps >= disc_start_step:
            d_fmt = f"{loss_d.item():.4f}"
            g_fmt = f"{loss_gan.item():.4f}"
        else:
            d_fmt = "warmup"
            g_fmt = "0.0000"
            
        metrics = {
            "g": f"{loss_g.item():.2f}", 
            "stft": f"{stft_loss.item():.2f}",
            "mel": f"{mel_loss.item():.2f}",
            "q": f"{q_loss.item():.2f}",
            "d": d_fmt,
            "gan": g_fmt,
            "bps": f"{bps:.0f}"
        }
        pbar.set_postfix(metrics)
        
        if steps % 100 == 0:
            log_file.write(f"Step {steps}: g={loss_g.item():.2f}, mel={mel_loss.item():.3f}, d={loss_d.item():.2f}\n")
            log_file.flush()
        
        if steps % 500 == 0:
            torch.save(factorizer.state_dict(), f"{CHECKPOINT_DIR}/factorizer_{steps}.pt")
            torch.save(decoder.state_dict(), f"{CHECKPOINT_DIR}/decoder_{steps}.pt")
            torch.save(entropy_model.state_dict(), f"{CHECKPOINT_DIR}/entropy_{steps}.pt")
            torch.save(discriminator.state_dict(), f"{CHECKPOINT_DIR}/discriminator_{steps}.pt")
            torch.save(spk_pq.state_dict(), f"{CHECKPOINT_DIR}/spk_pq_{steps}.pt")
            torch.save(sem_vq.state_dict(), f"{CHECKPOINT_DIR}/sem_rfsq_{steps}.pt")
            torch.save(pro_vq.state_dict(), f"{CHECKPOINT_DIR}/pro_rfsq_{steps}.pt")
            
            with torch.no_grad():
                mel_viz = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256)
                orig_mel = torch.log(mel_viz(audio[0,0].cpu().unsqueeze(0)).clamp(min=1e-5)).squeeze().numpy()
                recon_mel = torch.log(mel_viz(audio_hat[0,0].cpu().unsqueeze(0)).clamp(min=1e-5)).squeeze().numpy()
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].imshow(orig_mel, aspect='auto', origin='lower', cmap='magma')
                axes[0].set_title(f'Original (Step {steps})')
                axes[1].imshow(recon_mel, aspect='auto', origin='lower', cmap='magma')
                axes[1].set_title(f'Reconstructed (MEL: {mel_loss.item():.2f})')
                plt.savefig(f"{CHECKPOINT_DIR}/spectrograms/step_{steps}.png", dpi=100)
                plt.close()
            
            val_mel = validate(steps)
            print(f"\n  📊 Step {steps}: MEL={mel_loss.item():.2f}, Val={val_mel:.2f}, D={loss_d.item():.2f}")
            log_file.write(f"  Validation MEL: {val_mel:.3f}\n")
            log_file.flush()
        
        if steps >= max_steps: break

pbar.close()
log_file.close()
print("\n✅ Training complete!")
