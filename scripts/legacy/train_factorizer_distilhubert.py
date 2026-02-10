#!/usr/bin/env python3
"""
Train Factorizer on DistilHuBERT Features

This adapts the existing Factorizer training for DistilHuBERT features.
Key changes from HuBERT training:
- Uses precomputed DistilHuBERT features (768-dim, same as HuBERT)
- No runtime feature extraction needed
- Faster training due to precomputation

Usage:
    python train_factorizer_distilhubert.py \
        --config configs/ultra200bps_distilhubert.yaml \
        --features_dir data/features_distilhubert \
        --checkpoint_dir checkpoints/factorizer_distilhubert
"""

import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import yaml
import os
import glob
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss, BandwiseMelLoss, HingeGANLoss, FeatureMatchingLoss
from ultra_low_bitrate_codec.models.discriminator import BitSpectrogramDiscriminator


class DistilHuBERTFeatureDataset(Dataset):
    """Dataset for precomputed DistilHuBERT features."""
    
    def __init__(
        self, 
        features_dir: str,
        audio_dir: str = None,
        max_frames: int = 500,
        include_audio: bool = True
    ):
        self.features_dir = features_dir
        self.audio_dir = audio_dir
        self.max_frames = max_frames
        self.include_audio = include_audio
        
        # Load metadata
        metadata_path = os.path.join(features_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            self.files = [f['name'] for f in self.metadata['files']]
        else:
            # Fallback: scan directory
            self.files = [os.path.splitext(f)[0] for f in os.listdir(features_dir) 
                          if f.endswith('.pt')]
            self.metadata = {'model': 'distilhubert', 'feature_dim': 768}
        
        print(f"Dataset: {len(self.files)} files, features in {features_dir}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        name = self.files[idx]
        feature_path = os.path.join(self.features_dir, f"{name}.pt")
        
        try:
            data = torch.load(feature_path, map_location='cpu')
            features = data['features']  # (T, 768)
            
            # Truncate if needed
            if features.shape[0] > self.max_frames:
                start = torch.randint(0, features.shape[0] - self.max_frames, (1,)).item()
                features = features[start:start + self.max_frames]
            
            result = {'features': features, 'name': name}
            
            # Optionally load audio
            if self.include_audio and 'audio_path' in data:
                try:
                    import soundfile as sf
                    audio_path = data['audio_path']
                    if self.audio_dir:
                        audio_path = os.path.join(self.audio_dir, os.path.basename(audio_path))
                    wav, sr = sf.read(audio_path)
                    wav = torch.tensor(wav, dtype=torch.float32)
                    if wav.dim() > 1:
                        wav = wav.mean(dim=0)
                    if sr != 16000:
                        wav = torchaudio.functional.resample(wav, sr, 16000)
                    
                    # Align to features (320 samples per frame)
                    target_samples = features.shape[0] * 320
                    if wav.shape[0] > target_samples:
                        wav = wav[:target_samples]
                    elif wav.shape[0] < target_samples:
                        wav = F.pad(wav, (0, target_samples - wav.shape[0]))
                    
                    result['audio'] = wav
                except Exception as e:
                    # Return zeros if audio load fails
                    result['audio'] = torch.zeros(features.shape[0] * 320)
            
            return result
            
        except Exception as e:
            print(f"Error loading {feature_path}: {e}")
            # Return dummy data
            return {
                'features': torch.zeros(self.max_frames, 768),
                'audio': torch.zeros(self.max_frames * 320),
                'name': name
            }


def collate_fn(batch):
    """Collate with padding."""
    max_frames = max(item['features'].shape[0] for item in batch)
    max_audio = max_frames * 320
    
    features = []
    audio = []
    
    for item in batch:
        f = item['features']
        if f.shape[0] < max_frames:
            f = F.pad(f, (0, 0, 0, max_frames - f.shape[0]))
        features.append(f)
        
        if 'audio' in item:
            a = item['audio']
            if a.shape[0] < max_audio:
                a = F.pad(a, (0, max_audio - a.shape[0]))
            audio.append(a)
    
    result = {'features': torch.stack(features)}
    if audio:
        result['audio'] = torch.stack(audio)
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--audio_dir", default=None)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--pretrained", default=None, 
                        help="Path to pretrained Factorizer to initialize from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze_flow_epoch", type=int, default=5, help="Epoch to unfreeze flow model")
    parser.add_argument("--use_gan", action='store_true', help="Enable GAN training")
    parser.add_argument("--gan_start_epoch", type=int, default=10, help="Epoch to start GAN training")
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(f"{args.checkpoint_dir}/spectrograms", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Optimization: High precision matmuls
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("🚀 Tensor Core optimization enabled (high precision)")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # =========================================================================
    # Models
    # =========================================================================
    # =========================================================================
    # Models
    # =========================================================================
    print("Initializing models (Flow Matching Strategy)...")
    
    # 1. Factorizer (Trainable)
    factorizer = InformationFactorizerV2(config).to(device)
    
    # 2. Quantizers (Frozen)
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
    
    # 3. Fuser (Frozen)
    from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
    fuser = ConditionFuserV2(
        sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512,
        sem_upsample=4, pro_upsample=8
    ).to(device)
    
    # 4. Flow Model (Frozen)
    from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
    # Override config for Flow (80 bands)
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    
    print(f"  Factorizer params: {sum(p.numel() for p in factorizer.parameters())/1e6:.2f}M")
    
    # Load checkpoints
    print("Loading checkpoints...")
    
    # Factorizer Checkpoint (Base initialization)
    # Factorizer Checkpoint (Base initialization)
    if args.pretrained:
         if os.path.exists(args.pretrained):
             base_path = args.pretrained
             if os.path.isfile(base_path):
                  base_path = os.path.dirname(base_path)
             
             print(f"  Loading Factorizer base from {base_path}")
             def load_ckpt(path, model):
                 if os.path.exists(path):
                     print(f"    - Loading {os.path.basename(path)}")
                     ckpt = torch.load(path, map_location=device)
                     if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                         ckpt = ckpt['model_state_dict']
                     ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
                     model.load_state_dict(ckpt, strict=False)
             
             load_ckpt(f"{base_path}/factorizer.pt", factorizer)
             load_ckpt(f"{base_path}/sem_rfsq.pt", sem_vq)
             load_ckpt(f"{base_path}/pro_rfsq.pt", pro_vq)
             load_ckpt(f"{base_path}/spk_pq.pt", spk_pq)
             
             # Also try to load Fuser/Flow from the prompt path if they exist (for Resume)
             load_ckpt(f"{base_path}/fuser.pt", fuser)
             load_ckpt(f"{base_path}/flow_model.pt", flow_model)
             
         else:
             print(f"⚠️  Pretrained path {args.pretrained} not found!")

    # Flow & Fuser Checkpoints (Ground Truth Providers)
    # Flow & Fuser Checkpoints (Ground Truth Providers) - DISABLED to respect pretrained V4
    # fuser_ckpt_path = "checkpoints/checkpoints_flow_v2/fuser_epoch20.pt"
    # flow_ckpt_path = "checkpoints/checkpoints_flow_v2/flow_epoch20.pt"
    # 
    # print(f"  Loading Fuser from {fuser_ckpt_path}")
    # fuser_ckpt = torch.load(fuser_ckpt_path, map_location=device)
    # if 'model_state_dict' in fuser_ckpt: fuser_ckpt = fuser_ckpt['model_state_dict']
    # fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fuser_ckpt.items()})
    # 
    # print(f"  Loading Flow from {flow_ckpt_path}")
    # flow_ckpt = torch.load(flow_ckpt_path, map_location=device)
    # if 'model_state_dict' in flow_ckpt: flow_ckpt = flow_ckpt['model_state_dict']
    # flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()})

    # Freeze everything except Factorizer (Initial setup)
    print("Freezing downstream models...")
    for m in [sem_vq, pro_vq, spk_pq, fuser, flow_model]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
            
    # torch.compile optimization (MOVED AFTER LOADING)
    # torch.compile optimization (User requested)
    if hasattr(torch, 'compile'):
        print("🚀 Compiling models with torch.compile...")
        try:
             factorizer = torch.compile(factorizer)
             fuser = torch.compile(fuser)
             flow_model = torch.compile(flow_model)
             # Quantizers are frozen and small, typically not worth compiling or might have issues
             # Discriminator will be compiled if used
        except Exception as e:
             print(f"⚠️  torch.compile failed: {e}")
    else:
       print("⚠️  torch.compile not found. Skipping.")
            
    # =========================================================================
    # Data
    # =========================================================================
    dataset = DistilHuBERTFeatureDataset(
        features_dir=args.features_dir,
        audio_dir=args.audio_dir,
        max_frames=500,
        include_audio=True # We need audio for Mel Targets
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # =========================================================================
    # Optimizer & Loss
    # =========================================================================
    optimizer = optim.AdamW(factorizer.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(dataloader),
        eta_min=1e-6
    )
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, n_mels=80, hop_length=320
    ).to(device)
    
    def get_mel(x):
        # Normalize: (log(mel) - (-5.0)) / 3.5
        mel = mel_transform(x).clamp(min=1e-5).log()
        mel = (mel - (-5.0)) / 3.5 
        return mel.transpose(1, 2) # (B, T, 80)
    
    # =========================================================================
    # Training
    # =========================================================================
    # =========================================================================
    # GAN Setup
    # =========================================================================
    if args.use_gan:
        print("Initializing BitNet GAN (Spectrogram Discriminator)...")
        discriminator = BitSpectrogramDiscriminator().to(device)
        if hasattr(torch, 'compile'):
             discriminator = torch.compile(discriminator)
        print(f"  Discriminator params: {sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M")
        
        # Restore GAN LR for Sharpening (Phase 5)
        # gan_lr = args.lr * 0.5 # Removed
        opt_d = optim.AdamW(discriminator.parameters(), lr=args.lr, weight_decay=0.01)
        sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs * len(dataloader), eta_min=1e-6)
        
        criterion_gan = HingeGANLoss().to(device)
        criterion_fm = FeatureMatchingLoss().to(device)
    else:
        discriminator = None
        
    
    # =========================================================================
    # Training
    # =========================================================================
    print(f"\nStarting Flow-Guided Factorizer Training for {args.epochs} epochs...")
    
    scaler = torch.amp.GradScaler('cuda')
    best_loss = float('inf')
    
    log_file = open(f"{args.checkpoint_dir}/training.log", "w")
    log_file.write(f"Factorizer Fine-tuning (Flow Strategy)\n")
    log_file.write("=" * 60 + "\n")
    
    global_step = 0
    
    for epoch in range(args.epochs):
        # UNFREEZE FLOW MODEL
        if epoch == args.unfreeze_flow_epoch:
             print(f"\n❄️ Unfreezing Flow Model at Epoch {epoch}...")
             for p in flow_model.parameters():
                 p.requires_grad = True
             optimizer.add_param_group({'params': flow_model.parameters()})
             print(f"   Optimizer now tracks {len(list(factorizer.parameters())) + len(list(flow_model.parameters()))} parameters.")

        factorizer.train()
        if epoch >= args.unfreeze_flow_epoch:
            flow_model.train()
            
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(device)
            audio = batch['audio'].to(device)
            
            if audio is None: continue
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # 1. Get Target Mel
                mel_gt = get_mel(audio) # (B, T_mel, 80)
                
                # 2. Factorize MicroHuBERT features
                sem, pro, spk = factorizer(features)
                
                # 3. Quantize
                sem_z, q_loss_sem, _ = sem_vq(sem)
                pro_z, q_loss_pro, _ = pro_vq(pro)
                spk_z, q_loss_spk, _ = spk_pq(spk)
                
                # 4. Fuse
                # Need to align lengths. Fuser handles upsampling.
                # Mel length is roughly Input/320 * 256 / 256 = Input
                # 320 hop for Mel? Config says 256 hop for Mel in get_mel.
                # Inference uses 320 hop (16k -> 50Hz). 
                # Wait, Inference uses: mel_transform default (hop 256?? No).
                # In inference_microhubert_pipeline.py: 
                # target_mel_len = samples // 320.
                # So we must match that.
                
                target_mel_len = mel_gt.shape[1]
                cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
                
                # Align conditioning to Mel if slight mismatch
                if cond.shape[1] != mel_gt.shape[1]:
                     min_len = min(cond.shape[1], mel_gt.shape[1])
                     cond = cond[:, :min_len, :]
                     mel_gt = mel_gt[:, :min_len, :]
                
                # Global Step Check: Save Spectrogram for monitoring
                if global_step % 500 == 0:
                    with torch.no_grad():
                        # Try to generate from noise to see IF flow is working
                        test_cond = cond[0:1]
                        generated_mel = flow_model.solve_ode(test_cond, steps=20)
                        
                        spec_path = f"{args.checkpoint_dir}/spectrograms/step_{global_step}.png"
                        os.makedirs(os.path.dirname(spec_path), exist_ok=True)
                        
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(10, 4))
                        plt.imshow(generated_mel[0].T.cpu().float().numpy(), aspect='auto', origin='lower')
                        plt.colorbar()
                        fm_val = fm_loss.item() if 'fm_loss' in locals() else -1.0
                        plt.title(f"Step {global_step} (FM={fm_val:.3f})")
                        plt.savefig(spec_path)
                        plt.close()
                
                # 5. Compute Loss
                # GAN Training (If enabled and active)
                if args.use_gan and epoch >= args.gan_start_epoch:
                    # ==========================
                    # 1. Train Discriminator
                    # ==========================
                    # Generate Fake (Straight Path)
                    # We need to construct the graph for G update later, so we capture the variables
                    t = torch.rand(mel_gt.shape[0], device=device)
                    x_t = (1 - t[:, None, None]) * torch.randn_like(mel_gt) + t[:, None, None] * mel_gt
                    v_pred = flow_model(x_t, t, cond)
                    
                    # Fake Mel for D (Detached for D training)
                    x_1_hat = x_t + (1 - t[:, None, None]) * v_pred
                    mel_fake_d = x_1_hat.detach()
                    mel_real_d = mel_gt
                    
                    # Forward D
                    # Reshape for BitDiscriminator (B, 1, 80, T)
                    mf_d_img = mel_fake_d.unsqueeze(1).transpose(2, 3)
                    mr_d_img = mel_real_d.unsqueeze(1).transpose(2, 3)
                    
                    y_dr, y_df, _, _ = discriminator(mr_d_img, mf_d_img)
                    
                    loss_d = criterion_gan.discriminator_loss(y_dr, y_df)
                    
                    opt_d.zero_grad()
                    scaler.scale(loss_d).backward()
                    scaler.step(opt_d)
                    sched_d.step()
                    
                    # ==========================
                    # 2. Train Generator
                    # ==========================
                    # Freeze D for G update to avoid unnecessary grad computation
                    for p in discriminator.parameters(): p.requires_grad = False
                    
                    # Re-forward D with attached fake
                    # Note: x_1_hat graph is still valid (v_pred -> cond -> factorizer)
                    mf_g_img = x_1_hat.unsqueeze(1).transpose(2, 3)
                    mr_g_img = mel_gt.unsqueeze(1).transpose(2, 3)
                    
                    y_dr_g, y_df_g, fmap_r, fmap_f = discriminator(mr_g_img, mf_g_img)
                    
                    loss_gen = criterion_gan.generator_loss(y_df_g)
                    loss_fm_gan = criterion_fm(fmap_r, fmap_f)
                    
                    # Rectified Flow Loss (Regression)
                    target_v = mel_gt - ((x_t - t[:, None, None]*mel_gt) / (1 - t[:, None, None])) # Recovery of V?
                    # Simpler: target_v = x_1 - x_0. But we didn't save x_0 explicitly in line 352 logic above.
                    # Let's preserve x_0 logic:
                    # x_t = (1-t)x_0 + t*x_1
                    # x_0 = (x_t - t*x_1) / (1-t)
                    # target_v = x_1 - x_0
                    # This is numerically unstable at t=1. 
                    
                    # Retrying logic for G step safety:
                    # Let's just create a FRESH sample for G regression loss to be safe/standard
                    # and use the x_1_hat from above for the GAN loss.
                    # Actually, we can just compute regression loss on the v_pred we already have!
                    # We just need the Target V.
                    # x_t was created as (1-t)N(0,1) + t*x_1.
                    # We didn't save N(0,1).
                    # Optimization: Create x_0 explicitly.
                else:
                    loss_d = torch.tensor(0.0)
                    
                # Re-do the Flow Step cleanly for G Update (Unified Path)
                if args.use_gan and epoch >= args.gan_start_epoch:
                     # We need to re-compute v_pred for G backward if we want to be safe, 
                     # OR use the v_pred from D-step if we are careful.
                     # But we didn't calculate regression loss in D-step.
                     # Let's do it properly:
                     
                     # 1. Flow Matching Loss (Regression)
                     # We can use standard compute_loss (it samples new t, x_0)
                     # This provides strong regression signal.
                     loss_reg = flow_model.compute_loss(mel_gt, cond)
                     
                     # 2. GAN Loss (Adversarial)
                     # derived from the previous v_pred which is connected to graph
                     # v_pred -> x_1_hat -> D -> loss_gen
                     # Is v_pred graph alive? Yes, we didn't backward through it yet.
                     # BUT we updated D weights.
                     # So D forward MUST be re-run.
                     # We did re-run D (y_df_g).
                     # So loss_gen is valid.
                # Re-do the Flow Step cleanly for G Update (Unified Path)
                if args.use_gan and epoch >= args.gan_start_epoch:
                     # 1. Flow Matching Loss (Regression)
                     loss_reg = flow_model.compute_loss(mel_gt, cond)
                     
                     loss = loss_reg + (loss_gen * 2.5) + loss_fm_gan + (q_loss_sem + q_loss_pro + q_loss_spk)*0.25
                     
                     # Unfreeze D
                     for p in discriminator.parameters(): p.requires_grad = True
                     
                     fm_loss = loss_reg

                else:
                     fm_loss = flow_model.compute_loss(mel_gt, cond)
                     loss = fm_loss + (q_loss_sem + q_loss_pro + q_loss_spk) * 0.25
            
            if torch.isnan(loss):
                 scaler.update() # Fix for RuntimeError: step() called without update()
                 continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # 6. Global Gradient Clipping (Stability Fix)
            active_params = list(factorizer.parameters())
            if epoch >= args.unfreeze_flow_epoch:
                active_params += list(flow_model.parameters())
            
            torch.nn.utils.clip_grad_norm_(active_params, 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.2f}",
                "fm": f"{fm_loss.item():.2f}",
                "q": f"{(q_loss_sem + q_loss_pro + q_loss_spk).item():.2f}"
            })
            
            if global_step % 100 == 0:
                log_file.write(f"Step {global_step}: loss={loss.item():.3f}, fm={fm_loss.item():.3f}\n")
                log_file.flush()

            # AGGRESSIVE CHECKPOINTING (Quality-based)
            if fm_loss.item() < 0.25:
                print(f"  🌟 Excellent Quality detected (FM={fm_loss.item():.3f})! Saving step checkpoint...")
                torch.save(factorizer.state_dict(), f"{args.checkpoint_dir}/factorizer_best_step.pt")
                torch.save(fuser.state_dict(), f"{args.checkpoint_dir}/fuser_best_step.pt")
                if args.unfreeze_flow_epoch <= epoch:
                    torch.save(flow_model.state_dict(), f"{args.checkpoint_dir}/flow_best_step.pt")
                if discriminator is not None:
                    torch.save(discriminator.state_dict(), f"{args.checkpoint_dir}/discriminator_best_step.pt")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        # Save checkpoint (Every epoch)
        save_dir = f"{args.checkpoint_dir}/epoch_{epoch+1}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(factorizer.state_dict(), f"{save_dir}/factorizer.pt")
        torch.save(fuser.state_dict(), f"{save_dir}/fuser.pt")
        if args.unfreeze_flow_epoch <= epoch:
             torch.save(flow_model.state_dict(), f"{save_dir}/flow_model.pt")
        if discriminator is not None:
             torch.save(discriminator.state_dict(), f"{save_dir}/discriminator.pt")
        
        # Save Best Model (Based on training loss for now, should be Validation but..)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(factorizer.state_dict(), f"{args.checkpoint_dir}/factorizer.pt")
            torch.save(fuser.state_dict(), f"{args.checkpoint_dir}/fuser.pt")
            if args.unfreeze_flow_epoch <= epoch:
                torch.save(flow_model.state_dict(), f"{args.checkpoint_dir}/flow_model.pt")
            if discriminator is not None:
                torch.save(discriminator.state_dict(), f"{args.checkpoint_dir}/discriminator.pt")
            print(f"  New best model!")
                
    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")
    log_file.close()

if __name__ == "__main__":
    main()
