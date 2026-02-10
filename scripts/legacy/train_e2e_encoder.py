#!/usr/bin/env python3
"""
End-to-End Training: MicroEncoder → Factorizer → Flow → Vocoder

No HuBERT dependency. Train encoder from scratch with reconstruction loss.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# Models
from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoder, MicroEncoderTiny
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet
from torch.amp import autocast, GradScaler


class AudioDataset(Dataset):
    """Simple audio dataset"""
    def __init__(self, audio_dir, sample_rate=16000, max_len=48000):
        self.files = []
        for f in os.listdir(audio_dir):
            if f.endswith(('.wav', '.flac', '.mp3')):
                self.files.append(os.path.join(audio_dir, f))
        self.sr = sample_rate
        self.max_len = max_len
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        import soundfile as sf
        wav, sr = sf.read(path)
        wav = torch.tensor(wav, dtype=torch.float32)
        if wav.dim() > 1 or (wav.dim() == 1 and len(wav.shape) > 1):
            wav = wav.mean(dim=-1) if wav.dim() > 1 else wav
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        
        # Random crop or pad
        if wav.shape[0] > self.max_len:
            start = torch.randint(0, wav.shape[0] - self.max_len, (1,)).item()
            wav = wav[start:start + self.max_len]
        else:
            wav = F.pad(wav, (0, self.max_len - wav.shape[0]))
        
        return wav


def collate_fn(batch):
    return torch.stack(batch)


def compute_mel(wav, mel_transform):
    """Compute log-mel spectrogram"""
    mel = mel_transform(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel


def save_spectrogram(mel, path, title=""):
    plt.figure(figsize=(10, 4))
    if mel.dim() == 3:
        mel = mel.squeeze(0)
    plt.imshow(mel.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


class MultiResolutionSpectralLoss(nn.Module):
    # Keep class for potential future use or evaluation, but remove from active training loop if desired.
    # Actually, let's remove it to clean up the script as per plan.
    pass

def get_optimal_batch_size():
    """Dynamically determine batch size based on available GPU memory"""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        free_mb = int(result.stdout.strip().split('\n')[0])
        
        # Rough estimate: ~500MB per sample for this pipeline (3s audio)
        # Reserve 2GB for CUDA overhead
        usable_mb = free_mb - 2000
        
        # Each batch item ~350MB with fp16
        mb_per_sample = 350
        
        raw_optimal = max(4, usable_mb // mb_per_sample)
        
        # Snap to nearest power of 2 (downwards)
        # e.g., 50 -> 32, 24 -> 16
        import math
        optimal = 2 ** int(math.log2(raw_optimal))
        
        # Cap at reasonable maximum
        optimal = min(optimal, 128)
        
        print(f"  GPU Free VRAM: {free_mb} MB -> Auto batch_size: {optimal}")
        return optimal
    except Exception as e:
        print(f"  Could not detect GPU memory: {e}, using default batch_size=32")
        return 32


# ... (imports remain the same) ...

# ==============================================================================
# Deep Diagnostics
# ==============================================================================
class DeepDiagnostics:
    def __init__(self, writer):
        self.writer = writer
    
    def log_grad_norms(self, model, name, step):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.writer.add_scalar(f'GradNorm/{name}', total_norm, step)
        print(f"  [Diag] GradNorm/{name}: {total_norm:.4f}", flush=True)
        return total_norm

    def log_feature_stats(self, features, name, step):
        # features: (B, T, D)
        if features.numel() == 0: return
        mean = features.mean().item()
        std = features.std().item()
        self.writer.add_scalar(f'Stats/{name}_mean', mean, step)
        self.writer.add_scalar(f'Stats/{name}_std', std, step)
        print(f"  [Diag] Stats/{name}: mean={mean:.3f}, std={std:.3f}", flush=True)
        
    def log_codebook_usage(self, indices, name, step, vocab_size):
        # indices: (B, T, num_levels) or (B, T)
        if indices.numel() == 0: return
        
        # Handle multi-level/group indices
        if indices.dim() > 2:
            # Flatten B and T, keep levels/groups
            indices = indices.reshape(-1, indices.shape[-1]) # (N, levels)
            num_levels = indices.shape[-1]
            
            total_unique = 0
            level_utilizations = []
            
            for i in range(num_levels):
                level_indices = indices[:, i]
                unique = torch.unique(level_indices).numel()
                utilization = unique / vocab_size
                level_utilizations.append(utilization)
                total_unique += unique
                
                # Log detailed stats for first few levels to debug
                if i < 4:
                     self.writer.add_scalar(f'Codebook/{name}_L{i}_utilization', utilization, step)

            avg_utilization = sum(level_utilizations) / len(level_utilizations)
            self.writer.add_scalar(f'Codebook/{name}_avg_utilization', avg_utilization, step)
            print(f"  [Diag] Codebook/{name}: avg_util={avg_utilization*100:.1f}% (L0: {level_utilizations[0]*100:.1f}%)", flush=True)
            
        else:
            # Single level case
            unique = torch.unique(indices).numel()
            utilization = unique / vocab_size
            self.writer.add_scalar(f'Codebook/{name}_utilization', utilization, step)
            self.writer.add_scalar(f'Codebook/{name}_unique', unique, step)
            print(f"  [Diag] Codebook/{name}: unique={unique}/{vocab_size} ({utilization*100:.1f}%)", flush=True)

# ... (rest of the file) ...

def main():
    parser = argparse.ArgumentParser()
    # ... (args remain same) ...
    parser.add_argument("--data_dir", default="data/audio")
    parser.add_argument("--output_dir", default="checkpoints/microencoder_e2e")
    parser.add_argument("--config", default="src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=0, help="0=auto detect based on GPU VRAM")
    parser.add_argument("--lr", type=float, default=1e-4) # Encoder LR
    parser.add_argument("--lr_flow", type=float, default=2e-5)  # Lower LR for pretrained Flow
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    # Model options
    parser.add_argument("--tiny", action="store_true", help="Use MicroEncoderTiny (0.5M params)")
    parser.add_argument("--encoder_hidden", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--freeze_flow", action="store_true", help="Freeze Flow model")
    parser.add_argument("--freeze_vocoder", action="store_true", default=True)
    
    # Checkpoints to load
    parser.add_argument("--flow_ckpt", default="checkpoints/checkpoints_flow_v2/flow_epoch31.pt")
    parser.add_argument("--fuser_ckpt", default="checkpoints/checkpoints_flow_v2/fuser_epoch31.pt")
    parser.add_argument("--vocoder_ckpt", default="checkpoints/vocoder_mel/vocoder_latest.pt")
    
    # Resume from E2E training
    parser.add_argument("--encoder_ckpt", default=None, help="Resume encoder from E2E checkpoint")
    parser.add_argument("--factorizer_ckpt", default=None, help="Resume factorizer from E2E checkpoint")
    parser.add_argument("--fuser_ckpt_e2e", default=None, help="Resume fuser from E2E checkpoint")
    
    # CFG
    parser.add_argument("--cond_drop_prob", type=float, default=0.1, help="Probability of dropping conditioning for CFG")
    
    args = parser.parse_args()
    
    # Auto-detect batch size
    if args.batch_size == 0:
        args.batch_size = get_optimal_batch_size()
    else:
        print(f"  Using specified batch_size: {args.batch_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    diagnostics = DeepDiagnostics(writer)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # ========== Models ==========
    print("=" * 60)
    print("Loading Models...")
    print("=" * 60)

    # 1. MicroEncoder (Auto-detect logic)
    # Check encoder checkpoint to determine if Tiny or Large
    is_tiny = args.tiny
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        try:
            ckpt = torch.load(args.encoder_ckpt, map_location='cpu')
            state_dict = ckpt if not 'model_state_dict' in ckpt else ckpt['model_state_dict']
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            
            # Check hidden dim from state dict
            if 'output_proj.weight' in state_dict:
                hidden_dim = state_dict['output_proj.weight'].shape[1]
                if hidden_dim == 128:
                    print("  [Auto-Detect] Checkpoint is Tiny (128-dim). Switching to MicroEncoderTiny.")
                    is_tiny = True
                elif hidden_dim == 256:
                    print("  [Auto-Detect] Checkpoint is Standard (256-dim).")
                    is_tiny = False
        except Exception as e:
            print(f"  [Warning] Failed to inspect checkpoint: {e}")

    if is_tiny:
        encoder = MicroEncoderTiny(
            hidden_dim=128, # Fixed for Tiny
            output_dim=768,
            num_layers=2
        ).to(device)
        print(f"  MicroEncoderTiny: {sum(p.numel() for p in encoder.parameters()):,} params")
    else:
        encoder = MicroEncoder(
            hidden_dim=args.encoder_hidden,
            output_dim=768,
            num_layers=args.encoder_layers
        ).to(device)
        print(f"  MicroEncoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    
    # Load encoder checkpoint
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location=device)
        encoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
        print(f"  Encoder: loaded from {args.encoder_ckpt}")
    
    # 2. Factorizer
    factorizer = InformationFactorizerV2(config).to(device)
    print(f"  Factorizer: {sum(p.numel() for p in factorizer.parameters()):,} params")
    if args.factorizer_ckpt and os.path.exists(args.factorizer_ckpt):
        ckpt = torch.load(args.factorizer_ckpt, map_location=device)
        factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
        print(f"  Factorizer: loaded from {args.factorizer_ckpt}")
    
    # 3. Quantizers
    sem_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=8
    ).to(device)
    pro_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=8
    ).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    # 4. Fuser (load pretrained)
    fuser = ConditionFuserV2(
        sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512,
        sem_upsample=4, pro_upsample=8
    ).to(device)
    fuser_ckpt_to_load = args.fuser_ckpt_e2e if args.fuser_ckpt_e2e and os.path.exists(args.fuser_ckpt_e2e) else args.fuser_ckpt
    if fuser_ckpt_to_load and os.path.exists(fuser_ckpt_to_load):
        ckpt = torch.load(fuser_ckpt_to_load, map_location=device)
        fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
        print(f"  Fuser: loaded from {fuser_ckpt_to_load}")
    
    # 5. Flow
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow = ConditionalFlowMatching(config).to(device)
    if os.path.exists(args.flow_ckpt):
        ckpt = torch.load(args.flow_ckpt, map_location=device)
        flow.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}, strict=False)
        print(f"  Flow: loaded from {args.flow_ckpt}")
    
    if args.freeze_flow:
        print("  [INFO] Flow model FROZEN requested.")
        for p in flow.parameters():
            p.requires_grad = False
    else:
        print("  [INFO] Flow model UNFRAMED (Joint Training Enabled).")
    
    # Compile
    torch.set_float32_matmul_precision('high')
    # encoder = torch.compile(encoder)
    # factorizer = torch.compile(factorizer)
    # fuser = torch.compile(fuser)
    # flow = torch.compile(flow)
    print("  Models compiled (SKIPPED for debugging)")
    
    # 6. Vocoder (frozen)
    vocoder = MelVocoderBitNet().to(device)
    if os.path.exists(args.vocoder_ckpt):
        ckpt = torch.load(args.vocoder_ckpt, map_location=device)
        vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
    for p in vocoder.parameters():
        p.requires_grad = False
    vocoder.eval()
    
    # Mel Transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
    ).to(device)
    MEAN = -5.0
    STD = 3.5
    
    # Dataset
    dataset = AudioDataset(args.data_dir, max_len=48000)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )
    print(f"Dataset: {len(dataset)} samples")
    
    # Optimizer
    encoder_params = list(encoder.parameters()) + list(factorizer.parameters()) + \
                     list(fuser.parameters()) + list(sem_vq.parameters()) + \
                     list(pro_vq.parameters()) + list(spk_pq.parameters())
    
    if not args.freeze_flow:
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': args.lr},
            {'params': flow.parameters(), 'lr': args.lr_flow}
        ], betas=(0.8, 0.99))
    else:
        optimizer = torch.optim.AdamW(encoder_params, lr=args.lr, betas=(0.8, 0.99))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler('cuda') if args.fp16 else None
    
    # Training Loop
    print("=" * 60)
    print("Starting E2E Training (with Deep Diagnostics)...")
    print("=" * 60)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        encoder.train()
        factorizer.train()
        fuser.train()
        flow.train() if not args.freeze_flow else flow.eval()
        
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, wav in enumerate(pbar):
            wav = wav.to(device)
            gt_mel = mel_transform(wav)
            gt_mel = torch.log(torch.clamp(gt_mel, min=1e-5))
            target_mel_len = gt_mel.shape[-1]
            
            x1 = (gt_mel - MEAN) / STD
            x1 = x1.transpose(1, 2)
            
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=args.fp16):
                features_gt = encoder(wav)
                sem, pro, spk = factorizer(features_gt)
                sem_z, _, sem_idx = sem_vq(sem)
                pro_z, _, pro_idx = pro_vq(pro)
                spk_z, _, spk_idx = spk_pq(spk)
                
                cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
                
                if args.cond_drop_prob > 0:
                     mask_prob = torch.rand(wav.shape[0], device=device)
                     drop_mask = (mask_prob < args.cond_drop_prob).float().view(-1, 1, 1)
                     cond = cond * (1 - drop_mask)
                
                flow_loss = flow.compute_loss(x1, cond)
                loss = flow_loss
            
            # Backward
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # Diagnostics: Grad Norms
                if global_step % 100 == 0:
                    diagnostics.log_grad_norms(encoder, "Encoder", global_step)
                    diagnostics.log_grad_norms(factorizer, "Factorizer", global_step)
                    diagnostics.log_grad_norms(fuser, "Fuser", global_step)
                    if not args.freeze_flow:
                        diagnostics.log_grad_norms(flow, "Flow", global_step)
                
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
                if not args.freeze_flow:
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                if global_step % 100 == 0:
                    diagnostics.log_grad_norms(encoder, "Encoder", global_step)
                    diagnostics.log_grad_norms(factorizer, "Factorizer", global_step)
                
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
                if not args.freeze_flow:
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Diagnostics: Stats & Codebooks
            if global_step % 100 == 0:
                with torch.no_grad():
                    # Feature Stats
                    diagnostics.log_feature_stats(features_gt, "Encoder_Feat", global_step)
                    diagnostics.log_feature_stats(sem, "Sem_Latent", global_step)
                    diagnostics.log_feature_stats(pro, "Pro_Latent", global_step)
                    diagnostics.log_feature_stats(cond, "Conditioning", global_step)
                    
                    # Codebook Usage (approx unique codes in batch)
                    diagnostics.log_codebook_usage(sem_idx, "Semantic", global_step, 4096) # Approx
                    diagnostics.log_codebook_usage(pro_idx, "Prosody", global_step, 4096)
                    
                writer.add_scalar('Loss/flow', flow_loss.item(), global_step)
            
            pbar.set_postfix({'flow': f'{flow_loss.item():.3f}'})
        
        # Epoch End
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        writer.add_scalar('Epoch/loss', avg_loss, epoch)
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(encoder.state_dict(), f"{args.output_dir}/encoder_ep{epoch+1}.pt")
            torch.save(factorizer.state_dict(), f"{args.output_dir}/factorizer_ep{epoch+1}.pt")
            torch.save(fuser.state_dict(), f"{args.output_dir}/fuser_ep{epoch+1}.pt")
            if not args.freeze_flow:
                torch.save(flow.state_dict(), f"{args.output_dir}/flow_ep{epoch+1}.pt")
            print(f"  Saved checkpoints at epoch {epoch+1}")
            
            # Visualization
            with torch.no_grad():
                sample_wav = wav[:1]
                sample_gt = gt_mel[:1]
                
                feat = encoder(sample_wav)
                s, p, spk = factorizer(feat)
                sz, _, _ = sem_vq(s)
                pz, _, _ = pro_vq(p)
                spkz, _, _ = spk_pq(spk)
                c = fuser(sz, pz, spkz, sample_gt.shape[-1])
                
                pred = flow.solve_ode(c, steps=50, solver='rk4')
                pred = pred * STD + MEAN
                pred = pred.transpose(1, 2)
                
                save_spectrogram(sample_gt, f"{args.output_dir}/ep{epoch+1}_gt.png", "Ground Truth")
                save_spectrogram(pred, f"{args.output_dir}/ep{epoch+1}_pred.png", f"Prediction EP{epoch+1}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), f"{args.output_dir}/encoder_best.pt")
            torch.save(factorizer.state_dict(), f"{args.output_dir}/factorizer_best.pt")
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    writer.close()

if __name__ == "__main__":
    main()
