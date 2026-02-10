#!/usr/bin/env python3
"""
End-to-End Training V2: MicroEncoder → Factorizer → Flow → Vocoder

Improvements over V1 (train_e2e_encoder_aug.py):
- B3: Smaller Prosody VQ (4D × 2 levels) — 66% prosody bitrate reduction
- B2: F0 Prosody Reconstruction Loss (auxiliary)  
- D2: Contrastive Semantic Invariance Loss
- C3: EMA on Flow model weights
- D1: Optional CTC Loss on Semantics (--use_ctc)
- Aug V2: Uses data/audio_aug_v2 (no pitch shift)
"""

import os
import sys
import copy
import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F_fn
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


# ==============================================================================
# Dataset
# ==============================================================================
class DualAudioDataset(Dataset):
    """Loads pairs of (augmented, original) audio for Siamese training."""
    def __init__(self, audio_dir, aug_dir, sample_rate=16000, max_len=48000):
        self.audio_dir = audio_dir
        self.aug_dir = aug_dir
        self.files = []
        
        for root, dirs, files in os.walk(audio_dir):
            for f in files:
                if f.endswith(('.wav', '.flac', '.mp3')):
                    rel_path = os.path.relpath(os.path.join(root, f), audio_dir)
                    self.files.append(rel_path)
        
        self.sr = sample_rate
        self.max_len = max_len
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        rel_path = self.files[idx]
        orig_path = os.path.join(self.audio_dir, rel_path)
        aug_path = os.path.join(self.aug_dir, rel_path)
        
        import soundfile as sf
        try:
            wav, sr = sf.read(orig_path)
            wav = torch.tensor(wav, dtype=torch.float32)
        except Exception:
            wav = torch.zeros(self.max_len)
            sr = 16000

        try:
            if os.path.exists(aug_path):
                aug_wav, aug_sr = sf.read(aug_path)
                aug_wav = torch.tensor(aug_wav, dtype=torch.float32)
            else:
                aug_wav = wav.clone()
                aug_sr = sr
        except Exception:
            aug_wav = wav.clone()
            aug_sr = sr

        def process(w, s):
            if w.dim() > 1: w = w.mean(dim=-1)
            if s != self.sr: w = torchaudio.functional.resample(w, s, self.sr)
            peak = w.abs().max()
            if peak > 0: w = w / (peak + 1e-6)
            return w

        wav = process(wav, sr)
        aug_wav = process(aug_wav, aug_sr)
        
        if wav.shape[0] > self.max_len:
            start = torch.randint(0, wav.shape[0] - self.max_len, (1,)).item()
            wav = wav[start:start + self.max_len]
            if aug_wav.shape[0] >= start + self.max_len:
                aug_wav = aug_wav[start:start + self.max_len]
            else:
                aug_wav = aug_wav[:self.max_len]
        else:
            wav = F_fn.pad(wav, (0, self.max_len - wav.shape[0]))
            
        if aug_wav.shape[0] > self.max_len:
            aug_wav = aug_wav[:self.max_len]
        elif aug_wav.shape[0] < self.max_len:
            aug_wav = F_fn.pad(aug_wav, (0, self.max_len - aug_wav.shape[0]))
            
        return aug_wav, wav


def collate_fn(batch):
    aug_batch = torch.stack([b[0] for b in batch])
    orig_batch = torch.stack([b[1] for b in batch])
    return aug_batch, orig_batch


def compute_mel(wav, mel_transform):
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


def get_optimal_batch_size():
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        free_mb = int(result.stdout.strip().split('\n')[0])
        usable_mb = free_mb - 2000
        mb_per_sample = 350
        raw_optimal = max(4, usable_mb // mb_per_sample)
        optimal = 2 ** int(math.log2(raw_optimal))
        optimal = min(optimal, 128)
        print(f"  GPU Free VRAM: {free_mb} MB -> Auto batch_size: {optimal}")
        return optimal
    except Exception as e:
        print(f"  Could not detect GPU memory: {e}, using default batch_size=32")
        return 32


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
        return total_norm

    def log_feature_stats(self, features, name, step):
        if features.numel() == 0: return
        mean = features.mean().item()
        std = features.std().item()
        self.writer.add_scalar(f'Stats/{name}_mean', mean, step)
        self.writer.add_scalar(f'Stats/{name}_std', std, step)
        
    def log_codebook_usage(self, indices, name, step, vocab_size):
        if indices.numel() == 0: return
        if indices.dim() > 2:
            indices = indices.reshape(-1, indices.shape[-1])
            num_levels = indices.shape[-1]
            level_utilizations = []
            for i in range(num_levels):
                unique = torch.unique(indices[:, i]).numel()
                utilization = unique / vocab_size
                level_utilizations.append(utilization)
                if i < 4:
                    self.writer.add_scalar(f'Codebook/{name}_L{i}_utilization', utilization, step)
            avg_utilization = sum(level_utilizations) / len(level_utilizations)
            self.writer.add_scalar(f'Codebook/{name}_avg_utilization', avg_utilization, step)
        else:
            unique = torch.unique(indices).numel()
            utilization = unique / vocab_size
            self.writer.add_scalar(f'Codebook/{name}_utilization', utilization, step)


# ==============================================================================
# EMA Helper
# ==============================================================================
class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
    
    def update(self, model):
        with torch.no_grad():
            for k, v in model.named_parameters():
                if v.requires_grad:
                    # Handle torch.compile prefix
                    clean_k = k.replace("_orig_mod.", "")
                    if clean_k in self.shadow:
                        v = v.data
                        self.shadow[clean_k] = self.decay * self.shadow[clean_k] + (1 - self.decay) * v
    
    def apply(self, model):
        """Load EMA weights into model (for inference/saving)."""
        model.load_state_dict(self.shadow)
    
    def state_dict(self):
        return self.shadow
    
    def save(self, path):
        torch.save(self.shadow, path)


# ==============================================================================
# F0 Extraction (for B2: Prosody Reconstruction Loss)
# ==============================================================================
def extract_f0(wav, sr=16000, hop_length=320, f0_min=50, f0_max=600):
    """
    Extract F0 contour from waveform using autocorrelation (torchaudio).
    Returns: (B, T_mel) tensor of F0 values in Hz (0 = unvoiced).
    """
    # torchaudio.functional.detect_pitch_frequency expects (B, T)
    f0 = torchaudio.functional.detect_pitch_frequency(
        wav, sr, 
        frame_time=hop_length / sr,
        freq_low=f0_min, freq_high=f0_max
    )
    return f0  # (B, T_frames)


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="E2E Training V2 with improved losses and smaller prosody")
    
    # Data
    parser.add_argument("--data_dir", default="data/audio")
    parser.add_argument("--aug_dir", default="data/audio_aug_v2", help="Aug V2 (no pitch shift)")
    parser.add_argument("--output_dir", default="checkpoints/microencoder_v2")
    parser.add_argument("--config", default="src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=0, help="0=auto detect")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_flow", type=float, default=2e-5)
    parser.add_argument("--fp16", action="store_true")
    
    # Model options
    parser.add_argument("--tiny", action="store_true", help="Use MicroEncoderTiny")
    parser.add_argument("--encoder_hidden", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--freeze_flow", action="store_true")
    
    # Checkpoints
    parser.add_argument("--flow_ckpt", default="checkpoints/checkpoints_flow_v2/flow_epoch31.pt")
    parser.add_argument("--fuser_ckpt", default="checkpoints/checkpoints_flow_v2/fuser_epoch31.pt")
    parser.add_argument("--vocoder_ckpt", default="checkpoints/vocoder_mel/vocoder_latest.pt")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--factorizer_ckpt", default=None)
    parser.add_argument("--fuser_ckpt_e2e", default=None)
    
    # CFG
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    
    # V2 Features
    parser.add_argument("--pro_dim", type=int, default=4, help="Prosody VQ dim (B3: 4 instead of 8)")
    parser.add_argument("--pro_levels", type=int, default=2, help="Prosody RFSQ levels (B3: 2 instead of 3)")
    parser.add_argument("--w_contrastive", type=float, default=0.1, help="Contrastive loss weight (D2)")
    parser.add_argument("--w_f0", type=float, default=0.1, help="F0 reconstruction loss weight (B2)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay (C3)")
    parser.add_argument("--use_ctc", action="store_true", help="Enable CTC loss on semantics (D1)")
    parser.add_argument("--w_ctc", type=float, default=0.05, help="CTC loss weight")
    
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
    print("Loading Models (V2 — Improved)...")
    print("=" * 60)

    # 1. MicroEncoder (Auto-detect logic)
    is_tiny = args.tiny
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        try:
            ckpt = torch.load(args.encoder_ckpt, map_location='cpu')
            state_dict = ckpt if 'model_state_dict' not in ckpt else ckpt['model_state_dict']
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            if 'output_proj.weight' in state_dict:
                hidden_dim = state_dict['output_proj.weight'].shape[1]
                if hidden_dim == 128:
                    print("  [Auto-Detect] Tiny (128-dim).")
                    is_tiny = True
                elif hidden_dim == 256:
                    print("  [Auto-Detect] Standard (256-dim).")
                    is_tiny = False
        except Exception as e:
            print(f"  [Warning] Failed to inspect checkpoint: {e}")

    if is_tiny:
        encoder = MicroEncoderTiny(hidden_dim=128, output_dim=768, num_layers=2).to(device)
    else:
        encoder = MicroEncoder(hidden_dim=args.encoder_hidden, output_dim=768, num_layers=args.encoder_layers).to(device)
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params ({'Tiny' if is_tiny else 'Standard'})")
    
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location=device)
        encoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
        print(f"  Encoder: loaded from {args.encoder_ckpt}")
    
    # 2. Factorizer
    # Override prosody output_dim in config for B3
    config['model']['prosody']['output_dim'] = args.pro_dim
    factorizer = InformationFactorizerV2(config).to(device)
    print(f"  Factorizer: {sum(p.numel() for p in factorizer.parameters()):,} params (pro_dim={args.pro_dim})")
    if args.factorizer_ckpt and os.path.exists(args.factorizer_ckpt):
        ckpt = torch.load(args.factorizer_ckpt, map_location=device)
        # Partial load — prosody head dimensions changed
        current_state = factorizer.state_dict()
        loaded_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        compatible = {}
        for k, v in loaded_state.items():
            if k in current_state and v.shape == current_state[k].shape:
                compatible[k] = v
            else:
                print(f"  [Skip] {k}: shape mismatch ({v.shape} vs {current_state[k].shape})")
        current_state.update(compatible)
        factorizer.load_state_dict(current_state)
        print(f"  Factorizer: partial load from {args.factorizer_ckpt} ({len(compatible)}/{len(loaded_state)} keys)")
    
    # 3. Quantizers (B3: Prosody 4D × 2 levels)
    sem_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=8
    ).to(device)
    
    # B3: Smaller Prosody VQ
    # Use config levels but slice to pro_dim
    full_levels = config['model']['fsq_levels']
    if len(full_levels) < args.pro_dim:
        # Fallback if config is too short
        pro_fsq_levels = [8] * args.pro_dim
    else:
        pro_fsq_levels = full_levels[:args.pro_dim]
        
    pro_vq = ResidualFSQ(
        levels=pro_fsq_levels,
        num_levels=args.pro_levels,  # 2 residual levels
        input_dim=args.pro_dim
    ).to(device)
    print(f"  Prosody VQ: {args.pro_dim}D × {args.pro_levels} levels (FSQ levels: {pro_fsq_levels})")
    
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    # 4. Fuser (B3: pro_dim changed)
    fuser = ConditionFuserV2(
        sem_dim=8, pro_dim=args.pro_dim, spk_dim=256, out_dim=512,
        sem_upsample=4, pro_upsample=8
    ).to(device)
    fuser_ckpt_to_load = args.fuser_ckpt_e2e if args.fuser_ckpt_e2e and os.path.exists(args.fuser_ckpt_e2e) else args.fuser_ckpt
    if fuser_ckpt_to_load and os.path.exists(fuser_ckpt_to_load):
        ckpt = torch.load(fuser_ckpt_to_load, map_location=device)
        loaded = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        current = fuser.state_dict()
        compatible = {k: v for k, v in loaded.items() if k in current and v.shape == current[k].shape}
        current.update(compatible)
        fuser.load_state_dict(current)
        print(f"  Fuser: partial load ({len(compatible)}/{len(loaded)} keys) from {fuser_ckpt_to_load}")
    
    # 5. Flow
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow = ConditionalFlowMatching(config).to(device)
    if os.path.exists(args.flow_ckpt):
        ckpt = torch.load(args.flow_ckpt, map_location=device)
        flow.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}, strict=False)
        print(f"  Flow: loaded from {args.flow_ckpt}")
    
    if args.freeze_flow:
        print("  [INFO] Flow model FROZEN.")
        for p in flow.parameters():
            p.requires_grad = False
    else:
        print("  [INFO] Flow model UNFROZEN (Joint Training).")
    
    # C3: EMA on Flow
    ema_flow = EMA(flow, decay=args.ema_decay)
    print(f"  EMA: decay={args.ema_decay}")
    
    # Compile
    torch.set_float32_matmul_precision('high')
    # Optimizing: Compile the heavy lifters
    print("  Compiling models with torch.compile(mode='default')...")
    
    # Use default for stability
    compile_mode = 'default' 
    
    encoder = torch.compile(encoder, mode=compile_mode)
    factorizer = torch.compile(factorizer, mode=compile_mode)
    fuser = torch.compile(fuser, mode=compile_mode)
    if not args.freeze_flow:
        flow = torch.compile(flow, mode=compile_mode)
    # vocoder is frozen/eval only, no need to compile
    
    # 6. Vocoder (frozen, for visualization only)
    vocoder = MelVocoderBitNet().to(device)
    if os.path.exists(args.vocoder_ckpt):
        ckpt = torch.load(args.vocoder_ckpt, map_location=device)
        vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
    for p in vocoder.parameters():
        p.requires_grad = False
    vocoder.eval()
    
    # B2: F0 Prosody Head
    pro_f0_head = nn.Linear(args.pro_dim, 1).to(device)
    print(f"  F0 Head: {args.pro_dim} -> 1 (weight={args.w_f0})")
    
    # D1: CTC (optional)
    ctc_head = None
    ctc_processor = None
    ctc_model = None
    if args.use_ctc:
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            ctc_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            ctc_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
            ctc_model.eval()
            for p in ctc_model.parameters():
                p.requires_grad = False
            num_classes = ctc_model.config.vocab_size  # ~32
            ctc_head = nn.Linear(8, num_classes).to(device)
            print(f"  CTC Head: 8 -> {num_classes} (weight={args.w_ctc})")
        except Exception as e:
            print(f"  [Warning] CTC disabled: {e}")
            args.use_ctc = False
    
    # Mel Transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
    ).to(device)
    MEAN = -5.0
    STD = 3.5
    
    # Dataset (V2: audio_aug_v2)
    print(f"Dataset: Clean={args.data_dir}, Aug={args.aug_dir}")
    dataset = DualAudioDataset(args.data_dir, args.aug_dir, max_len=48000)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )
    print(f"Dataset: {len(dataset)} samples")
    
    # Optimizer (include F0 head and optional CTC head)
    encoder_params = list(encoder.parameters()) + list(factorizer.parameters()) + \
                     list(fuser.parameters()) + list(sem_vq.parameters()) + \
                     list(pro_vq.parameters()) + list(spk_pq.parameters()) + \
                     list(pro_f0_head.parameters())
    if ctc_head is not None:
        encoder_params += list(ctc_head.parameters())
    
    if not args.freeze_flow:
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': args.lr},
            {'params': flow.parameters(), 'lr': args.lr_flow}
        ], betas=(0.8, 0.99))
    else:
        optimizer = torch.optim.AdamW(encoder_params, lr=args.lr, betas=(0.8, 0.99))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler('cuda') if args.fp16 else None
    
    # ========== Training Loop ==========
    print("=" * 60)
    print("Starting E2E Training V2")
    print(f"  Losses: flow + {args.w_contrastive}*contrastive + {args.w_f0}*f0" + 
          (f" + {args.w_ctc}*ctc" if args.use_ctc else ""))
    print("=" * 60)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        encoder.train()
        factorizer.train()
        fuser.train()
        pro_f0_head.train()
        flow.train() if not args.freeze_flow else flow.eval()
        
        epoch_loss = 0
        epoch_flow = 0
        epoch_contrastive = 0
        epoch_f0 = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (aug_wav, wav) in enumerate(pbar):
            aug_wav = aug_wav.to(device)
            wav = wav.to(device)
            
            gt_mel = mel_transform(wav)
            gt_mel = torch.log(torch.clamp(gt_mel, min=1e-5))
            target_mel_len = gt_mel.shape[-1]
            
            x1 = (gt_mel - MEAN) / STD
            x1 = x1.transpose(1, 2)
            
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=args.fp16):
                # 1. Siamese Encoding
                combined_wav = torch.cat([aug_wav, wav], dim=0)
                combined_feat = encoder(combined_wav)
                
                B = wav.shape[0]
                feat_aug = combined_feat[:B]
                feat_orig = combined_feat[B:]
                
                # 2. Factorizer: Semantic from Aug, Prosody/Speaker from Orig
                sem_aug, _, _ = factorizer(feat_aug)
                sem_orig, pro_orig, spk_orig = factorizer(feat_orig)
                
                # 3. Quantization
                sem_z, _, sem_idx = sem_vq(sem_aug)
                pro_z, _, pro_idx = pro_vq(pro_orig)
                spk_z, _, spk_idx = spk_pq(spk_orig)
                
                # 4. Fusion
                cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
                
                # CFG: Conditioning dropout
                if args.cond_drop_prob > 0:
                    mask_prob = torch.rand(B, device=device)
                    drop_mask = (mask_prob < args.cond_drop_prob).float().view(-1, 1, 1)
                    cond = cond * (1 - drop_mask)
                
                # === Loss Computation ===
                
                # Flow Matching Loss (primary)
                flow_loss = flow.compute_loss(x1, cond)
                
                # D2: Contrastive Semantic Invariance Loss
                # sem_aug should match sem_orig (same content, different style)
                contrastive_loss = F_fn.mse_loss(sem_aug, sem_orig.detach())
                
                # B2: F0 Prosody Reconstruction Loss
                with torch.no_grad():
                    gt_f0 = extract_f0(wav, sr=16000, hop_length=320)
                
                # Align F0 length with prosody latent length
                # pro_orig: (B, T_pro, pro_dim), gt_f0: (B, T_f0)
                T_pro = pro_orig.shape[1]
                if gt_f0.shape[1] != T_pro:
                    gt_f0 = F_fn.interpolate(gt_f0.unsqueeze(1), size=T_pro, mode='linear', align_corners=False).squeeze(1)
                
                # Predict F0 from prosody (before quantization for better gradients)
                pred_f0 = pro_f0_head(pro_orig).squeeze(-1)  # (B, T_pro)
                
                # Log-scale F0 for stability (avoid huge values)
                gt_f0_log = torch.log(gt_f0.clamp(min=1) + 1)
                pred_f0_log = torch.log(pred_f0.abs().clamp(min=1) + 1)
                f0_loss = F_fn.mse_loss(pred_f0_log, gt_f0_log.detach())
                
                # D1: CTC Loss (optional)
                ctc_loss = torch.tensor(0.0, device=device)
                if args.use_ctc and ctc_head is not None:
                    with torch.no_grad():
                        # Get pseudo-labels from wav2vec2
                        inputs = ctc_processor(wav.cpu().numpy(), sampling_rate=16000, 
                                              return_tensors="pt", padding=True).input_values.to(device)
                        logits = ctc_model(inputs).logits
                        pseudo_labels = logits.argmax(dim=-1)  # (B, T_w2v)
                    
                    # Project semantic to phone space
                    sem_logits = ctc_head(sem_z)  # (B, T_sem, num_classes)
                    sem_log_probs = F_fn.log_softmax(sem_logits, dim=-1).transpose(0, 1)  # (T, B, C)
                    
                    # Align lengths
                    T_sem = sem_log_probs.shape[0]
                    T_lab = pseudo_labels.shape[1]
                    input_lengths = torch.full((B,), T_sem, dtype=torch.long, device=device)
                    target_lengths = torch.full((B,), T_lab, dtype=torch.long, device=device)
                    
                    if T_sem >= T_lab:
                        ctc_loss = F_fn.ctc_loss(sem_log_probs, pseudo_labels, input_lengths, target_lengths, blank=0, zero_infinity=True)
                
                # Total Loss
                loss = flow_loss + args.w_contrastive * contrastive_loss + args.w_f0 * f0_loss
                if args.use_ctc:
                    loss = loss + args.w_ctc * ctc_loss
            
            # Backward
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
                if not args.freeze_flow:
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
                if not args.freeze_flow:
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                optimizer.step()
            
            # C3: EMA update
            if not args.freeze_flow:
                ema_flow.update(flow)
            
            epoch_loss += loss.item()
            epoch_flow += flow_loss.item()
            epoch_contrastive += contrastive_loss.item()
            epoch_f0 += f0_loss.item()
            global_step += 1
            
            # Diagnostics
            if global_step % 100 == 0:
                with torch.no_grad():
                    diagnostics.log_feature_stats(feat_orig, "Encoder_Feat_Orig", global_step)
                    diagnostics.log_feature_stats(sem_aug, "Sem_Latent_Aug", global_step)
                    diagnostics.log_feature_stats(pro_orig, "Pro_Latent_Orig", global_step)
                    diagnostics.log_codebook_usage(sem_idx, "Semantic", global_step, 4096)
                    diagnostics.log_codebook_usage(pro_idx, "Prosody", global_step, 
                                                   pro_vq.vocab_size if hasattr(pro_vq, 'vocab_size') else 4096)
                    
                    diagnostics.log_grad_norms(encoder, "Encoder", global_step)
                    diagnostics.log_grad_norms(factorizer, "Factorizer", global_step)
                    if not args.freeze_flow:
                        diagnostics.log_grad_norms(flow, "Flow", global_step)
                    
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/flow', flow_loss.item(), global_step)
                writer.add_scalar('Loss/contrastive', contrastive_loss.item(), global_step)
                writer.add_scalar('Loss/f0', f0_loss.item(), global_step)
                if args.use_ctc:
                    writer.add_scalar('Loss/ctc', ctc_loss.item(), global_step)
            
            pbar.set_postfix({
                'flow': f'{flow_loss.item():.3f}',
                'ctr': f'{contrastive_loss.item():.3f}',
                'f0': f'{f0_loss.item():.3f}'
            })
        
        # Epoch End
        n_batches = len(dataloader)
        avg_loss = epoch_loss / n_batches
        avg_flow = epoch_flow / n_batches
        avg_ctr = epoch_contrastive / n_batches
        avg_f0 = epoch_f0 / n_batches
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Total: {avg_loss:.4f} | Flow: {avg_flow:.4f} | "
              f"Contrastive: {avg_ctr:.4f} | F0: {avg_f0:.4f}")
        writer.add_scalar('Epoch/loss', avg_loss, epoch)
        writer.add_scalar('Epoch/flow', avg_flow, epoch)
        writer.add_scalar('Epoch/contrastive', avg_ctr, epoch)
        writer.add_scalar('Epoch/f0', avg_f0, epoch)
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(encoder.state_dict(), f"{args.output_dir}/encoder_ep{epoch+1}.pt")
            torch.save(factorizer.state_dict(), f"{args.output_dir}/factorizer_ep{epoch+1}.pt")
            torch.save(fuser.state_dict(), f"{args.output_dir}/fuser_ep{epoch+1}.pt")
            torch.save(pro_f0_head.state_dict(), f"{args.output_dir}/f0_head_ep{epoch+1}.pt")
            if not args.freeze_flow:
                torch.save(flow.state_dict(), f"{args.output_dir}/flow_ep{epoch+1}.pt")
                ema_flow.save(f"{args.output_dir}/flow_ema_ep{epoch+1}.pt")
            if ctc_head is not None:
                torch.save(ctc_head.state_dict(), f"{args.output_dir}/ctc_head_ep{epoch+1}.pt")
            print(f"  Saved checkpoints at epoch {epoch+1}")
            
            # Visualization
            with torch.no_grad():
                sample_wav = wav[:1]
                sample_aug = aug_wav[:1]
                sample_gt_mel = gt_mel[:1]
                
                feat_o = encoder(sample_wav)
                feat_a = encoder(sample_aug)
                
                s_a, _, _ = factorizer(feat_a)
                _, p_o, spk_o = factorizer(feat_o)
                
                sz, _, _ = sem_vq(s_a)
                pz, _, _ = pro_vq(p_o)
                spkz, _, _ = spk_pq(spk_o)
                
                c = fuser(sz, pz, spkz, sample_gt_mel.shape[-1])
                
                pred = flow.solve_ode(c, steps=50, solver='midpoint')
                pred = pred * STD + MEAN
                pred = pred.transpose(1, 2)
                
                save_spectrogram(sample_gt_mel, f"{args.output_dir}/ep{epoch+1}_gt.png", "Ground Truth")
                save_spectrogram(pred, f"{args.output_dir}/ep{epoch+1}_pred.png", f"Ep{epoch+1} Recon")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), f"{args.output_dir}/encoder_best.pt")
            torch.save(factorizer.state_dict(), f"{args.output_dir}/factorizer_best.pt")
            if not args.freeze_flow:
                ema_flow.save(f"{args.output_dir}/flow_ema_best.pt")
    
    print("=" * 60)
    print("Training V2 Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    writer.close()

if __name__ == "__main__":
    main()
