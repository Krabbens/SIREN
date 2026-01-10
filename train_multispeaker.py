#!/usr/bin/env python3
"""
Train Ultra-Low Bitrate Codec on Multi-Speaker Dataset
LibriTTS (English) + MLS Polish
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
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

print("="*70)
print("🚀 TRENING MULTI-SPEAKER CODEC")
print("   LibriTTS (EN) + MLS Polish (PL)")
print("="*70)

# ============================================================================
# MODELS
# ============================================================================
print("\n📦 Initializing models...")

factorizer = InformationFactorizerV2(config).to(device)
decoder = SpeechDecoderV2(config).to(device)

# Quantizers (ResidualFSQ)
fsq_levels = config['model']['fsq_levels']
sem_dim = config['model']['semantic']['output_dim']
pro_dim = config['model']['prosody']['output_dim']

sem_vq = ResidualFSQ(
    levels=fsq_levels,
    num_levels=config['model'].get('rfsq_num_levels', 4),
    input_dim=sem_dim
).to(device)

pro_vq = ResidualFSQ(
    levels=fsq_levels,
    num_levels=config['model'].get('rfsq_num_levels', 4),
    input_dim=pro_dim
).to(device)

spk_pq = ProductQuantizer(
    input_dim=config['model']['speaker']['embedding_dim'],
    num_groups=config['model']['speaker']['num_groups'],
    codes_per_group=config['model']['speaker']['codes_per_group']
).to(device)

entropy_model = EntropyModel(config).to(device)
discriminator = HiFiGANDiscriminator().to(device)

# Count parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters())

total_params = (count_params(factorizer) + count_params(decoder) + 
                count_params(sem_vq) + count_params(pro_vq) + count_params(spk_pq))
print(f"   Total trainable params: {total_params/1e6:.2f}M")

# ============================================================================
# OPTIMIZERS
# ============================================================================
params = list(factorizer.parameters()) + list(decoder.parameters()) + \
         list(sem_vq.parameters()) + list(pro_vq.parameters()) + \
         list(spk_pq.parameters()) + list(entropy_model.parameters())

optimizer = optim.AdamW(params, lr=float(config['training']['learning_rate']), betas=(0.8, 0.99))
optimizer_d = optim.AdamW(discriminator.parameters(), lr=float(config['training']['learning_rate']), betas=(0.8, 0.99))

max_steps = config['training']['max_steps']
warmup_steps = config['training']['warmup_steps']
disc_start_step = config['training'].get('discriminator_start_step', 5000)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps)
scheduler_d = get_linear_schedule_with_warmup(optimizer_d, warmup_steps, max_steps)

# ============================================================================
# DATA
# ============================================================================
print("\n📊 Loading datasets...")

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

train_dl = DataLoader(
    train_ds,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=config['training']['num_workers'],
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

print(f"   Train: {len(train_ds)} samples")
print(f"   Val: {len(val_ds)} samples")

# ============================================================================
# LOSSES
# ============================================================================
mr_stft = MultiResolutionSTFTLoss().to(device)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256, win_length=1024
).to(device)

def mel_fn(y):
    return torch.log(mel_transform(y).clamp(min=1e-5))

os.makedirs(f"{CHECKPOINT_DIR}/spectrograms", exist_ok=True)

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================
def validate(step):
    factorizer.eval()
    decoder.eval()
    
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=2)
    
    total_mel = 0
    count = 0
    
    with torch.no_grad():
        for features, audio in val_dl:
            features = features.to(device)
            audio = audio.to(device)
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            
            sem, pro, spk = factorizer(features)
            sem_z, _, _ = sem_vq(sem)
            pro_z, _, _ = pro_vq(pro)
            spk_z, _, _ = spk_pq(spk)
            audio_hat = decoder(sem_z, pro_z, spk_z)
            
            if audio_hat.dim() == 2:
                audio_hat = audio_hat.unsqueeze(1)
            
            min_len = min(audio.shape[-1], audio_hat.shape[-1])
            mel_loss = F.l1_loss(mel_fn(audio[..., :min_len]), mel_fn(audio_hat[..., :min_len]))
            total_mel += mel_loss.item()
            count += 1
            
            if count >= 10:
                break
    
    factorizer.train()
    decoder.train()
    
    return total_mel / count if count > 0 else 0

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n🎯 Starting training...")
print(f"   Max steps: {max_steps}")
print(f"   Batch size: {config['training']['batch_size']}")
print(f"   Discriminator starts at step: {disc_start_step}")

scaler = torch.amp.GradScaler('cuda')
steps = 0
pbar = tqdm(total=max_steps, desc="Training")

log_file = open(f"{CHECKPOINT_DIR}/training.log", "w")
log_file.write(f"=== Multi-speaker training started ===\n")
log_file.write(f"Config: {CONFIG_PATH}\n")
log_file.write(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}\n\n")
log_file.flush()

while steps < max_steps:
    for batch in train_dl:
        features, audio = batch
        features = features.to(device)
        audio = audio.to(device)
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        # Forward pass
        # Forward pass
        # Force Factorizer to run in FP32 to avoid overflow/Inf (Stability Fix)
        with torch.amp.autocast('cuda', enabled=False):
            sem, pro, spk = factorizer(features.float())

        with torch.amp.autocast('cuda'):
            sem_z, sem_loss, sem_idx = sem_vq(sem)
            pro_z, pro_loss, pro_idx = pro_vq(pro)
            spk_z, spk_loss, spk_idx = spk_pq(spk)
            
            audio_hat = decoder(sem_z, pro_z, spk_z)
            if audio_hat.dim() == 2:
                audio_hat = audio_hat.unsqueeze(1)
            
            min_len = min(audio.shape[-1], audio_hat.shape[-1])
            audio = audio[..., :min_len]
            audio_hat = audio_hat[..., :min_len]
        
        # Discriminator (after warmup)
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
            # GAN losses
            if steps >= disc_start_step:
                mpd_res, mrd_res = discriminator(audio, audio_hat)
                loss_g_mpd, _ = generator_loss(mpd_res[1])
                loss_g_mrd, _ = generator_loss(mrd_res[1])
                loss_gan = (loss_g_mpd + loss_g_mrd) * config['training'].get('gan_weight', 1.0)
            else:
                loss_gan = torch.tensor(0.0, device=device)
            
            # Reconstruction losses
            sc_loss, mag_loss = mr_stft(audio_hat.squeeze(1), audio.squeeze(1))
            stft_loss = sc_loss + mag_loss
            mel_loss = F.l1_loss(mel_fn(audio), mel_fn(audio_hat))
            
            # Quantization losses
            q_loss = sem_loss + pro_loss + spk_loss
            
            # Bitrate estimation
            bits_sem = sem_vq.get_total_bits_per_frame() * sem.shape[1]
            bits_pro = pro_vq.get_total_bits_per_frame() * pro.shape[1]
            bits_spk = 32  # Fixed for speaker
            total_bits = bits_sem + bits_pro + bits_spk
            duration = audio.shape[-1] / 16000
            bps = total_bits / duration
            
            # Total loss
            loss_g = (stft_loss * config['training'].get('stft_weight', 2.0) +
                      mel_loss * config['training'].get('mel_weight', 45.0) +
                      q_loss * config['training'].get('commitment_weight', 0.25) +
                      loss_gan)
        
        scaler.scale(loss_g).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        scheduler_d.step()
        
        steps += 1
        pbar.update(1)
        
        metrics = {
            "g": f"{loss_g.item():.2f}", 
            "d": f"{loss_d.item():.2f}" if steps >= disc_start_step else "warmup",
            "mel": f"{mel_loss.item():.1f}", 
            "bps": f"{bps:.0f}"
        }
        pbar.set_postfix(metrics)
        
        # Logging
        if steps % 100 == 0:
            log_line = f"Step {steps}: loss_g={loss_g.item():.3f}, mel={mel_loss.item():.3f}, stft={stft_loss.item():.3f}, bps={bps:.1f}\n"
            log_file.write(log_line)
            log_file.flush()
        
        # Checkpoints + Validation
        if steps % 500 == 0:
            # Save checkpoints
            torch.save(factorizer.state_dict(), f"{CHECKPOINT_DIR}/factorizer_{steps}.pt")
            torch.save(decoder.state_dict(), f"{CHECKPOINT_DIR}/decoder_{steps}.pt")
            torch.save(entropy_model.state_dict(), f"{CHECKPOINT_DIR}/entropy_{steps}.pt")
            torch.save(discriminator.state_dict(), f"{CHECKPOINT_DIR}/discriminator_{steps}.pt")
            torch.save(spk_pq.state_dict(), f"{CHECKPOINT_DIR}/spk_pq_{steps}.pt")
            torch.save(sem_vq.state_dict(), f"{CHECKPOINT_DIR}/sem_rfsq_{steps}.pt")
            torch.save(pro_vq.state_dict(), f"{CHECKPOINT_DIR}/pro_rfsq_{steps}.pt")
            
            # Save spectrogram comparison
            with torch.no_grad():
                orig_viz = audio[0, 0].cpu()
                recon_viz = audio_hat[0, 0, :orig_viz.shape[0]].cpu()
                
                # Use CPU mel transform for visualization
                mel_viz = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256
                )
                orig_mel = torch.log(mel_viz(orig_viz.unsqueeze(0)).clamp(min=1e-5)).squeeze().numpy()
                recon_mel = torch.log(mel_viz(recon_viz.unsqueeze(0)).clamp(min=1e-5)).squeeze().numpy()
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].imshow(orig_mel, aspect='auto', origin='lower', cmap='magma')
                axes[0].set_title(f'Original (Step {steps})')
                axes[1].imshow(recon_mel, aspect='auto', origin='lower', cmap='magma')
                axes[1].set_title(f'Reconstructed (MEL: {mel_loss.item():.2f})')
                plt.savefig(f"{CHECKPOINT_DIR}/spectrograms/step_{steps}.png", dpi=100)
                plt.close()
            
            # Validation
            val_mel = validate(steps)
            print(f"\n  📊 Step {steps}: Train MEL={mel_loss.item():.2f}, Val MEL={val_mel:.2f}")
            log_file.write(f"  Validation MEL: {val_mel:.3f}\n")
            log_file.flush()
        
        if steps >= max_steps:
            break

pbar.close()
log_file.close()
print("\n✅ Training complete!")
print(f"   Checkpoints saved to: {CHECKPOINT_DIR}")
