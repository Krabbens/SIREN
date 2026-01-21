#!/usr/bin/env python3
"""
SIREN v2 Training Script
Clean, performance-focused implementation with balanced loss suite.
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
import glob
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from ultra_low_bitrate_codec.data.feature_dataset import PrecomputedFeatureDataset
from ultra_low_bitrate_codec.training.losses import (
    MultiResolutionSTFTLoss, 
    BandwiseMelLoss, 
    WaveformL1Loss, 
    SpectralFluxLoss
)


def fix_state_dict_keys(state_dict: dict) -> dict:
    """Remove torch.compile prefixes from state dict keys."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--resume_step", type=int, default=-1)
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(f"{args.checkpoint_dir}/spectrograms", exist_ok=True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # =========================================================================
    # RESUME STEP DETECTION
    # =========================================================================
    if args.fresh:
        resume_step = 0
    elif args.resume_step >= 0:
        resume_step = args.resume_step
    else:
        step_dirs = glob.glob(f"{args.checkpoint_dir}/step_*")
        steps = [int(os.path.basename(d).split('_')[-1]) for d in step_dirs if d.split('_')[-1].isdigit()]
        resume_step = max(steps) if steps else 0
        if resume_step > 0:
            print(f"Auto-detected checkpoint: step {resume_step}")

    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    print("=" * 70)
    print(f"🚀 SIREN v2 TRAINING - Step {resume_step}")
    print("=" * 70)
    print(f"FSQ: {config['model']['fsq_levels']}")
    print(f"Semantic: {config['model']['semantic']['output_dim']}d @ {50 // config['model']['semantic']['temporal_compression']}Hz")
    print(f"Prosody: {config['model']['prosody']['output_dim']}d @ {50 // config['model']['prosody']['temporal_compression']}Hz")
    print("=" * 70)

    factorizer = InformationFactorizerV2(config).to(device)
    decoder = SpeechDecoderV2(config).to(device)
    
    sem_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=config['model']['semantic']['output_dim']
    ).to(device)
    
    pro_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=config['model']['prosody']['output_dim']
    ).to(device)
    
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    
    entropy_model = EntropyModel(config).to(device)

    # =========================================================================
    # LOAD CHECKPOINTS
    # =========================================================================
    if resume_step > 0:
        step_dir = f"{args.checkpoint_dir}/step_{resume_step}"
        if os.path.isdir(step_dir):
            print(f"Loading from {step_dir}...")
            factorizer.load_state_dict(fix_state_dict_keys(torch.load(f"{step_dir}/factorizer.pt", map_location=device)), strict=False)
            decoder.load_state_dict(fix_state_dict_keys(torch.load(f"{step_dir}/decoder.pt", map_location=device)), strict=False)
            sem_vq.load_state_dict(fix_state_dict_keys(torch.load(f"{step_dir}/sem_rfsq.pt", map_location=device)), strict=False)
            pro_vq.load_state_dict(fix_state_dict_keys(torch.load(f"{step_dir}/pro_rfsq.pt", map_location=device)), strict=False)
            spk_pq.load_state_dict(fix_state_dict_keys(torch.load(f"{step_dir}/spk_pq.pt", map_location=device)), strict=False)
            try:
                entropy_model.load_state_dict(fix_state_dict_keys(torch.load(f"{step_dir}/entropy.pt", map_location=device)), strict=False)
            except:
                print("⚠️ Entropy model not found, reinitializing")
            print("Checkpoints loaded!")
        else:
            print(f"⚠️ Step dir not found, starting fresh")
            resume_step = 0

    # Compile models
    # print("Compiling with torch.compile()...")
    # factorizer = torch.compile(factorizer, mode='default')
    # decoder = torch.compile(decoder, mode='default')

    # =========================================================================
    # OPTIMIZER
    # =========================================================================
    params = (
        list(factorizer.parameters()) + 
        list(decoder.parameters()) + 
        list(sem_vq.parameters()) + 
        list(pro_vq.parameters()) + 
        list(spk_pq.parameters()) + 
        list(entropy_model.parameters())
    )
    
    optimizer = optim.AdamW(params, lr=float(config['training']['learning_rate']), betas=(0.8, 0.99))
    
    max_steps = config['training']['max_steps']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps - resume_step)

    # =========================================================================
    # DATA
    # =========================================================================
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
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=config['training']['num_workers'], 
        pin_memory=True,
        persistent_workers=True
    )
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    # =========================================================================
    # LOSSES (Balanced Suite)
    # =========================================================================
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    bandwise_mel_fn = BandwiseMelLoss(device=device)
    waveform_fn = WaveformL1Loss()
    spectral_flux_fn = SpectralFluxLoss().to(device)
    
    # Mel for validation only
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256
    ).to(device)
    
    def mel_fn(x):
        return mel_transform(x).clamp(min=1e-5).log()

    # Loss weights
    w_stft = config['training'].get('stft_weight', 5.0)
    w_mel = config['training'].get('bandwise_mel_weight', 3.0)
    w_wave = config['training'].get('waveform_weight', 1.0)
    w_flux = config['training'].get('spectral_flux_weight', 0.5)
    w_commit = config['training'].get('commitment_weight', 0.25)
    w_entropy = config['training'].get('entropy_weight', 0.0001)

    # =========================================================================
    # VALIDATION
    # =========================================================================
    def validate():
        factorizer.eval()
        decoder.eval()
        total_mel = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_dl):
                if i >= 10: break
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
                total_mel += F.l1_loss(mel_fn(audio[..., :min_len]), mel_fn(audio_hat[..., :min_len])).item()
        
        factorizer.train()
        decoder.train()
        return total_mel / 10

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print(f"\n🎯 Starting from step {resume_step}...")
    scaler = torch.amp.GradScaler('cuda')
    steps = resume_step
    pbar = tqdm(total=max_steps - resume_step, desc="Training")

    log_file = open(f"{args.checkpoint_dir}/training.log", "a")
    log_file.write(f"\n=== SIREN v2 from step {resume_step} ===\n")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=f"{args.checkpoint_dir}/tensorboard")
    
    # Auto-launch TensorBoard
    import subprocess
    tb_process = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", f"{args.checkpoint_dir}/tensorboard", "--port", "6006", "--bind_all"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print(f"📊 TensorBoard started at http://localhost:6006")

    while steps < max_steps:
        for batch in train_dl:
            features = batch['features'].to(device)
            audio = batch['audio'].to(device)
            if audio.dim() == 2: audio = audio.unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sem, pro, spk = factorizer(features)
                sem_z, sem_loss, sem_idx = sem_vq(sem)
                pro_z, pro_loss, pro_idx = pro_vq(pro)
                spk_z, spk_loss, _ = spk_pq(spk)
                
                audio_hat = decoder(sem_z, pro_z, spk_z)
                if audio_hat.dim() == 2: audio_hat = audio_hat.unsqueeze(1)
                
                min_len = min(audio.shape[-1], audio_hat.shape[-1])
                audio = audio[..., :min_len]
                audio_hat = audio_hat[..., :min_len]
                
                # Entropy (ENABLED - Byte-level modeling)
                sem_bits, pro_bits = entropy_model.estimate_bits(
                    sem_idx.view(sem_idx.size(0), -1).detach(),
                    pro_idx.view(pro_idx.size(0), -1).detach()
                )
                entropy_loss = (sem_bits.mean() + pro_bits.mean()) * w_entropy
                
                # Bitrate (approximate based on FSQ levels)
                bits_per_frame = config['model']['rfsq_num_levels'] * 8 * 3  # 3 bits per level * 8 dims * n residual
                frames_per_sec = 50 / config['model']['semantic']['temporal_compression']
                bps = torch.tensor(bits_per_frame * frames_per_sec, device=device)
            
            # Losses (FP32 for stability)
            audio_sq = audio.squeeze(1)
            audio_hat_sq = audio_hat.squeeze(1)
            
            sc_loss, mag_loss = stft_loss_fn(audio_hat_sq, audio_sq)
            stft_loss = sc_loss + mag_loss
            
            mel_loss = bandwise_mel_fn(audio_hat_sq, audio_sq)
            wave_loss = waveform_fn(audio_hat_sq, audio_sq)
            flux_loss = spectral_flux_fn(audio_hat_sq, audio_sq)
            q_loss = sem_loss + pro_loss + spk_loss
            
            loss = (
                stft_loss * w_stft +
                mel_loss * w_mel +
                wave_loss * w_wave +
                flux_loss * w_flux +
                q_loss * w_commit +
                entropy_loss
            )
            
            if torch.isnan(loss):
                print(f"⚠️ NaN at step {steps}")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            steps += 1
            pbar.update(1)
            
            # Calculate Entropy BPS (approximate)
            duration_avg = audio.shape[-1] / 16000.0
            ent_bps = (sem_bits.mean() + pro_bits.mean()) / duration_avg
            
            # Logging
            metrics = {
                "loss": f"{loss.item():.2f}",
                "stft": f"{stft_loss.item():.2f}",
                "mel": f"{mel_loss.item():.2f}",
                "ent": f"{entropy_loss.item():.2f}",
                "bps": f"{ent_bps.item():.0f}"
            }
            pbar.set_postfix(metrics)
            
            if steps % 100 == 0:
                log_file.write(f"Step {steps}: {metrics}\n")
                log_file.flush()
                
                # TensorBoard logging
                writer.add_scalar('Loss/total', loss.item(), steps)
                writer.add_scalar('Loss/stft', stft_loss.item(), steps)
                writer.add_scalar('Loss/mel', mel_loss.item(), steps)
                writer.add_scalar('Loss/entropy', entropy_loss.item(), steps)
                writer.add_scalar('Loss/quantization', q_loss.item(), steps)
                writer.add_scalar('Bitrate/bps', ent_bps.item(), steps)
                writer.add_scalar('LR', scheduler.get_last_lr()[0], steps)
            
            if steps % config['training'].get('save_every', 500) == 0:
                step_dir = f"{args.checkpoint_dir}/step_{steps}"
                os.makedirs(step_dir, exist_ok=True)
                
                torch.save(factorizer.state_dict(), f"{step_dir}/factorizer.pt")
                torch.save(decoder.state_dict(), f"{step_dir}/decoder.pt")
                torch.save(sem_vq.state_dict(), f"{step_dir}/sem_rfsq.pt")
                torch.save(pro_vq.state_dict(), f"{step_dir}/pro_rfsq.pt")
                torch.save(spk_pq.state_dict(), f"{step_dir}/spk_pq.pt")
                torch.save(entropy_model.state_dict(), f"{step_dir}/entropy.pt")
                
                # Spectrogram
                with torch.no_grad():
                    mel_viz = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256)
                    orig = mel_viz(audio[0, 0].cpu()).clamp(min=1e-5).log().numpy()
                    recon = mel_viz(audio_hat[0, 0].cpu()).clamp(min=1e-5).log().numpy()
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].imshow(orig, aspect='auto', origin='lower', cmap='magma')
                    axes[0].set_title(f'Original (Step {steps})')
                    axes[1].imshow(recon, aspect='auto', origin='lower', cmap='magma')
                    axes[1].set_title(f'Reconstructed (BPS: {bps.item():.0f})')
                    plt.savefig(f"{args.checkpoint_dir}/spectrograms/step_{steps}.png", dpi=100)
                    plt.close()
                
                val_mel = validate()
                print(f"\n  📊 Step {steps}: Val MEL={val_mel:.3f}, BPS={bps.item():.0f}")
                log_file.write(f"  Validation MEL: {val_mel:.3f}\n")
                log_file.flush()
            
            if steps >= max_steps:
                break

    pbar.close()
    log_file.close()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
