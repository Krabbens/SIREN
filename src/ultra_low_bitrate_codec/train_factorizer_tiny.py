#!/usr/bin/env python3
"""
SIREN v2 Factorizer Adaptation (TinyHubert)
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
    parser.add_argument("--pretrained_checkpoint", type=str, help="Path to pretrained model (step_87000) to initialize from")
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
    print(f"🚀 SIREN v2 Factorizer Adaptation - Step {resume_step}")
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
                print("⚠️ Entropy model not found in checkpoint")
            print("Checkpoints loaded!")
        else:
            print(f"⚠️ Step dir not found, starting fresh")
            resume_step = 0
            
    # Function to safely load state dict with size mismatch handling
    def load_safe(model, state_dict, prefix=""):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if own_state[name].shape != param.shape:
                print(f"⚠️ Skipping {prefix}{name} due to size mismatch: {param.shape} vs {own_state[name].shape}")
                continue
            own_state[name].copy_(param)
            
    if resume_step == 0 and args.pretrained_checkpoint:
        print(f"🔄 Initializing from Pretrained Checkpoint: {args.pretrained_checkpoint}")
        step_dir = args.pretrained_checkpoint
        
        try:
            sd = fix_state_dict_keys(torch.load(f"{step_dir}/factorizer.pt", map_location=device))
            load_safe(factorizer, sd, "factorizer.")
            print("Loaded Factorizer (to be adapted)")
        except Exception as e:
            print(f"Could not load Factorizer: {e}")
            
        try:
            sd = fix_state_dict_keys(torch.load(f"{step_dir}/decoder.pt", map_location=device))
            load_safe(decoder, sd, "decoder.")
        except Exception as e:
            print(f"Error loading decoder: {e}")
            
        try:
            sem_vq.load_state_dict(fix_state_dict_keys(torch.load(f"{step_dir}/sem_rfsq.pt", map_location=device)), strict=False)
            pro_vq.load_state_dict(fix_state_dict_keys(torch.load(f"{step_dir}/pro_rfsq.pt", map_location=device)), strict=False)
            spk_pq.load_state_dict(fix_state_dict_keys(torch.load(f"{step_dir}/spk_pq.pt", map_location=device)), strict=False)
        except Exception as e:
            print(f"Error loading quantizers: {e}")
            
        print("Loaded Decoder & Quantizers (Fixed Target via safe load)")
        
        # Reset resume_step to 0 to start "Adaptation" phase counter
        resume_step = 0


    # =========================================================================
    # OPTIMIZER - FROZEN DECODER/QUANTIZERS
    # =========================================================================
    print("🥶 Freezing Decoder, Quantizers, and Entropy Model...")
    for m in [decoder, sem_vq, pro_vq, spk_pq, entropy_model]:
        for p in m.parameters():
            p.requires_grad = False
            
    # Verification
    print(f"Factorizer trainable: {any(p.requires_grad for p in factorizer.parameters())}")
    print(f"Decoder trainable: {any(p.requires_grad for p in decoder.parameters())}")

    # Only optimize Factorizer
    params = list(factorizer.parameters())
    
    optimizer = optim.AdamW(params, lr=float(config['training']['learning_rate']), betas=(0.8, 0.99))
    
    max_steps = config['training']['max_steps'] # e.g. 120,000
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    # =========================================================================
    # DATA
    # =========================================================================
    # HACK: Handle separate train/val feature dirs
    # If feature_dir ends with _train, we guess val is _val
    train_feature_dir = config['data']['feature_dir']
    val_feature_dir = train_feature_dir
    if train_feature_dir.endswith("_train"):
        val_feature_dir = train_feature_dir.replace("_train", "_val")
        print(f"Using inferred val feature dir: {val_feature_dir}")

    train_ds = PrecomputedFeatureDataset(
        features_dir=train_feature_dir,
        manifest_path=config['data']['train_manifest'],
        max_frames=500
    )
    val_ds = PrecomputedFeatureDataset(
        features_dir=val_feature_dir,
        manifest_path=config['data']['val_manifest'],
        max_frames=500
    )
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0), 
        pin_memory=True,
        persistent_workers=config['training'].get('persistent_workers', False) and config['training'].get('num_workers', 0) > 0
    )
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    # =========================================================================
    # LOSSES
    # =========================================================================
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    bandwise_mel_fn = BandwiseMelLoss(device=device)
    waveform_fn = WaveformL1Loss()
    spectral_flux_fn = SpectralFluxLoss().to(device)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256
    ).to(device)
    
    def mel_fn(x):
        return mel_transform(x).clamp(min=1e-5).log()

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
    print(f"\n🎯 Starting training (Adaptation)...")
    scaler = torch.amp.GradScaler('cuda')
    steps = 0 # Local steps
    global_step = resume_step if resume_step > 0 else 0
    
    # If initialized from pretrained but fresh, start from 87000 conceptually?
    # No, let's just track steps from 0 for this run, but maybe log as global_steps + steps.
    # Actually, config max_steps is 120000. It implies total steps.
    # If we want to fine tune, we should probably start `global_step` at 87000.
    if args.pretrained_checkpoint and resume_step == 0:
        try:
            global_step = int(os.path.basename(args.pretrained_checkpoint).split('_')[-1])
        except:
            global_step = 0
            
    print(f"Global Step: {global_step}")
    
    pbar = tqdm(total=max_steps - global_step, desc="Adaptation")

    log_file = open(f"{args.checkpoint_dir}/training.log", "a")
    log_file.write(f"\n=== SIREN v2 Adaptation ===\n")
    
    writer = SummaryWriter(log_dir=f"{args.checkpoint_dir}/tensorboard")
    
    while global_step < max_steps:
        for batch in train_dl:
            features = batch['features'].to(device)
            audio = batch['audio'].to(device)
            if audio.dim() == 2: audio = audio.unsqueeze(1)
            
            optimizer.zero_grad()
            
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
                
                # Entropy
                sem_bits, pro_bits = entropy_model.estimate_bits(
                    sem_idx.view(sem_idx.size(0), -1).detach(),
                    pro_idx.view(pro_idx.size(0), -1).detach()
                )
                entropy_loss = (sem_bits.mean() + pro_bits.mean()) * w_entropy
                
                # BPS
                frames_per_sec = 50 / config['model']['semantic']['temporal_compression']
                bits_per_frame = config['model']['rfsq_num_levels'] * 8 * 3
                bps = torch.tensor(bits_per_frame * frames_per_sec, device=device)
            
            # Loss
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
                print(f"⚠️ NaN at step {global_step}")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            global_step += 1
            pbar.update(1)
            
            # Estimate BPS
            duration_avg = audio.shape[-1] / 16000.0
            ent_bps = (sem_bits.mean() + pro_bits.mean()) / duration_avg
            
            metrics = {
                "loss": f"{loss.item():.2f}",
                "stft": f"{stft_loss.item():.2f}",
                "mel": f"{mel_loss.item():.2f}",
                "ent": f"{entropy_loss.item():.2f}",
                "bps": f"{ent_bps.item():.0f}"
            }
            pbar.set_postfix(metrics)
            
            if global_step % 100 == 0:
                log_file.write(f"Step {global_step}: {metrics}\n")
                log_file.flush()
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/mel', mel_loss.item(), global_step)
            
            if global_step % config['training'].get('save_every', 2000) == 0:
                step_dir = f"{args.checkpoint_dir}/step_{global_step}"
                os.makedirs(step_dir, exist_ok=True)
                
                torch.save(factorizer.state_dict(), f"{step_dir}/factorizer.pt")
                torch.save(decoder.state_dict(), f"{step_dir}/decoder.pt")
                torch.save(sem_vq.state_dict(), f"{step_dir}/sem_rfsq.pt")
                torch.save(pro_vq.state_dict(), f"{step_dir}/pro_rfsq.pt")
                torch.save(spk_pq.state_dict(), f"{step_dir}/spk_pq.pt")
                torch.save(entropy_model.state_dict(), f"{step_dir}/entropy.pt")
                
                val_mel = validate()
                print(f"\n  📊 Step {global_step}: Val MEL={val_mel:.3f}")
                
            if global_step >= max_steps:
                break

    print("\n✅ Adaptation complete!")

if __name__ == "__main__":
    main()
