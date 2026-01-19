#!/usr/bin/env python3
"""
Sub-100bps Training Script - GAN-Free
Uses WavLM perceptual loss + STFT + Mel for stable training
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
import argparse
import glob

# Remove explicit path insertion as we will run this as an installed module
# sys.path.insert(0, '/home/sperm/diff')

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from ultra_low_bitrate_codec.data.feature_dataset import PrecomputedFeatureDataset
from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss, WavLMPerceptualLoss

def main():
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/sperm/diff/ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/sperm/diff/checkpoints_sub100bps")
    parser.add_argument("--resume_step", type=int, default=-1, help="Step to resume from. -1 to auto-detect.")
    args = parser.parse_args()

    CONFIG_PATH = args.config
    CHECKPOINT_DIR = args.checkpoint_dir

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(f"{CHECKPOINT_DIR}/spectrograms", exist_ok=True)

    # Auto-detect resume step
    if args.resume_step == -1:
        ckpts = glob.glob(f"{CHECKPOINT_DIR}/factorizer_*.pt")
        if not ckpts:
            RESUME_STEP = 0
            print("No checkpoints found. Starting from scratch.")
        else:
            steps = [int(os.path.basename(c).split('_')[-1].split('.')[0]) for c in ckpts]
            RESUME_STEP = max(steps)
            print(f"Auto-detected checkpoint step: {RESUME_STEP}")
    else:
        RESUME_STEP = args.resume_step

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    print("="*70)
    print(f"🚀 SUB-100BPS TRAINING (GAN-FREE) FROM STEP {RESUME_STEP}")
    print("="*70)
    print(f"Config: {CONFIG_PATH}")
    print(f"FSQ levels: {config['model']['fsq_levels']}")
    print(f"RFSQ levels: {config['model']['rfsq_num_levels']}")
    print(f"Semantic compression: {config['model']['semantic']['temporal_compression']}x")
    print(f"Prosody compression: {config['model']['prosody']['temporal_compression']}x")
    print("="*70)

    # Models
    factorizer = InformationFactorizerV2(config).to(device)
    decoder = SpeechDecoderV2(config).to(device)

    fsq_levels = config['model']['fsq_levels']
    sem_dim = config['model']['semantic']['output_dim']
    pro_dim = config['model']['prosody']['output_dim']
    rfsq_num_levels = config['model']['rfsq_num_levels']

    sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=sem_dim).to(device)
    pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=pro_dim).to(device)
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    entropy_model = EntropyModel(config).to(device)

    # Load checkpoints if resuming
    if RESUME_STEP > 0:
        print(f"Loading checkpoints from step {RESUME_STEP}...")
        try:
            factorizer.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/factorizer_{RESUME_STEP}.pt", map_location=device))
            decoder.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/decoder_{RESUME_STEP}.pt", map_location=device))
            sem_vq.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/sem_rfsq_{RESUME_STEP}.pt", map_location=device))
            pro_vq.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/pro_rfsq_{RESUME_STEP}.pt", map_location=device))
            spk_pq.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/spk_pq_{RESUME_STEP}.pt", map_location=device))
            try:
                entropy_model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/entropy_{RESUME_STEP}.pt", map_location=device))
            except (RuntimeError, FileNotFoundError) as e:
                print(f"⚠️ Could not load entropy model: {e}. Re-initializing.")
            print("Checkpoints loaded!")
        except FileNotFoundError as e:
            print(f"Error loading checkpoints: {e}. Starting from 0.")
            RESUME_STEP = 0
    else:
        print("Starting training from scratch (Step 0).")

    # Compile models for speed
    print("Compiling models with torch.compile()...")
    factorizer = torch.compile(factorizer, mode='default')
    decoder = torch.compile(decoder, mode='default')

    # Single optimizer for all models (simpler without GAN)
    params = list(factorizer.parameters()) + list(decoder.parameters()) + \
             list(sem_vq.parameters()) + list(pro_vq.parameters()) + \
             list(spk_pq.parameters()) + list(entropy_model.parameters())

    optimizer = optim.AdamW(params, lr=float(config['training']['learning_rate']), betas=(0.8, 0.99))

    max_steps = config['training']['max_steps']
    remaining_steps = max_steps - RESUME_STEP
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_steps)

    # Data
    train_ds = PrecomputedFeatureDataset(
        features_dir=config['data']['feature_dir'],
        manifest_path=config['data']['train_manifest'],
        max_frames=500
    )
    val_ds = PrecomputedFeatureDataset(
        features_dir=config['data']['feature_dir'],
        manifest_path=config['data']['val_manifest'],
        max_frames=500
    )
    train_dl = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True,
                          num_workers=config['training']['num_workers'], pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=True,
                        num_workers=config['training']['num_workers'], pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)

    # Losses (NO GAN)
    mr_stft = MultiResolutionSTFTLoss().to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256, win_length=1024
    ).to(device)

    # WavLM Perceptual Loss
    print("Loading WavLM for perceptual loss...")
    wavlm_loss = WavLMPerceptualLoss(device=device)

    def mel_fn(y):
        return torch.log(mel_transform(y).clamp(min=1e-5))

    def validate(step):
        factorizer.eval()
        decoder.eval()
        total_mel, count = 0, 0
        with torch.no_grad():
            for batch in val_dl:
                features = batch['features'].to(device)
                audio = batch['audio'].to(device)
                if audio.dim() == 2: audio = audio.unsqueeze(1)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
    print(f"\n🎯 Starting training from step {RESUME_STEP}...")
    scaler = torch.amp.GradScaler('cuda')
    steps = RESUME_STEP
    pbar = tqdm(total=max_steps - RESUME_STEP, desc=f"Training from {RESUME_STEP}")

    log_file = open(f"{CHECKPOINT_DIR}/training.log", "a")
    log_file.write(f"\n=== Started from step {RESUME_STEP} ===\n")
    log_file.write(f"Config: {CONFIG_PATH}\n")
    log_file.write(f"FSQ: {fsq_levels}, RFSQ levels: {rfsq_num_levels}\n")

    # Loss weights
    stft_weight = config['training'].get('stft_weight', 2.0)
    mel_weight = config['training'].get('mel_weight', 45.0)
    wavlm_weight = config['training'].get('wavlm_weight', 0.1)
    entropy_weight = config['training'].get('entropy_weight', 0.01)
    commitment_weight = config['training'].get('commitment_weight', 0.25)

    while steps < max_steps:
        for batch_idx, batch in enumerate(train_dl):
            features = batch['features'].to(device)
            audio = batch['audio'].to(device)
            if audio.dim() == 2: audio = audio.unsqueeze(1)
            
            # Forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sem, pro, spk = factorizer(features)
                
                sem_z, sem_loss, sem_indices = sem_vq(sem)
                pro_z, pro_loss, pro_indices = pro_vq(pro)
                spk_z, spk_loss, spk_indices = spk_pq(spk)
                
                # Entropy estimation
                sem_idx_flat = sem_indices.view(sem_indices.size(0), -1).detach()
                pro_idx_flat = pro_indices.view(pro_indices.size(0), -1).detach()
                sem_bits_batch, pro_bits_batch = entropy_model.estimate_bits(sem_idx_flat, pro_idx_flat)
                
                # Entropy Warmup: 0 weight for first 5000 steps
                curr_entropy_weight = 0.0 if steps < 5000 else entropy_weight
                entropy_loss = (sem_bits_batch.mean() + pro_bits_batch.mean()) * curr_entropy_weight
                
                audio_hat = decoder(sem_z, pro_z, spk_z)
                if audio_hat.dim() == 2: audio_hat = audio_hat.unsqueeze(1)
                min_len = min(audio.shape[-1], audio_hat.shape[-1])
                audio, audio_hat = audio[..., :min_len], audio_hat[..., :min_len]
            
            # Losses (computed outside autocast for stability)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # STFT Loss
                sc_loss, mag_loss = mr_stft(audio_hat.squeeze(1), audio.squeeze(1))
                stft_loss = sc_loss + mag_loss
                
                # Mel Loss
                mel_loss = F.l1_loss(mel_fn(audio), mel_fn(audio_hat))
                
                # Quantization Loss
                q_loss = sem_loss + pro_loss + spk_loss
                
                # Bitrate estimation
                est_bits = sem_bits_batch.mean() + pro_bits_batch.mean() + 32  # +32 for speaker
                bps = est_bits / (audio.shape[-1] / 16000)
            
            # WavLM Perceptual Loss (in FP32 for stability)
            with torch.amp.autocast('cuda', enabled=False):
                perceptual_loss = wavlm_loss(audio.float(), audio_hat.float())
            
            # Total loss
            loss = (stft_loss * stft_weight + 
                    mel_loss * mel_weight + 
                    q_loss * commitment_weight + 
                    entropy_loss +
                    perceptual_loss * wavlm_weight)
            
            # NaN check
            if torch.isnan(loss):
                print(f"⚠️ NaN detected at step {steps}. Skipping batch.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            steps += 1
            pbar.update(1)
            
            # Logging
            metrics = {
                "loss": f"{loss.item():.2f}", 
                "stft": f"{stft_loss.item():.2f}",
                "mel": f"{mel_loss.item():.2f}",
                "wavlm": f"{perceptual_loss.item():.3f}",
                "q": f"{q_loss.item():.2f}",
                "bps": f"{bps:.0f}"
            }
            pbar.set_postfix(metrics)
            
            if steps % 100 == 0:
                log_file.write(f"Step {steps}: loss={loss.item():.2f}, mel={mel_loss.item():.3f}, bps={bps:.0f}\n")
                log_file.flush()
            
            if steps % config['training'].get('save_every', 500) == 0:
                torch.save(factorizer.state_dict(), f"{CHECKPOINT_DIR}/factorizer_{steps}.pt")
                torch.save(decoder.state_dict(), f"{CHECKPOINT_DIR}/decoder_{steps}.pt")
                torch.save(entropy_model.state_dict(), f"{CHECKPOINT_DIR}/entropy_{steps}.pt")
                torch.save(spk_pq.state_dict(), f"{CHECKPOINT_DIR}/spk_pq_{steps}.pt")
                torch.save(sem_vq.state_dict(), f"{CHECKPOINT_DIR}/sem_rfsq_{steps}.pt")
                torch.save(pro_vq.state_dict(), f"{CHECKPOINT_DIR}/pro_rfsq_{steps}.pt")
                
                # Save spectrogram visualization
                with torch.no_grad():
                    mel_viz = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256)
                    orig_mel = torch.log(mel_viz(audio[0,0].cpu().unsqueeze(0)).clamp(min=1e-5)).squeeze().numpy()
                    recon_mel = torch.log(mel_viz(audio_hat[0,0].cpu().unsqueeze(0)).clamp(min=1e-5)).squeeze().numpy()
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].imshow(orig_mel, aspect='auto', origin='lower', cmap='magma')
                    axes[0].set_title(f'Original (Step {steps})')
                    axes[1].imshow(recon_mel, aspect='auto', origin='lower', cmap='magma')
                    axes[1].set_title(f'Reconstructed (BPS: {bps:.0f})')
                    plt.savefig(f"{CHECKPOINT_DIR}/spectrograms/step_{steps}.png", dpi=100)
                    plt.close()
                
                val_mel = validate(steps)
                print(f"\n  📊 Step {steps}: MEL={mel_loss.item():.2f}, Val={val_mel:.2f}, BPS={bps:.0f}")
                log_file.write(f"  Validation MEL: {val_mel:.3f}\n")
                log_file.flush()
            
            if steps >= max_steps: break

    pbar.close()
    log_file.close()
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
