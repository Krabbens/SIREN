#!/usr/bin/env python3
"""
Train MicroHuBERT via Distillation (Single GPU)

Trains the <5MB MicroHuBERT to output vectors matching Official HuBERT Layer 9.
Input: Raw Audio
Target: Precomputed HuBERT Features (from scripts/precompute_hubert_features.py)
Loss: L1 + Cosine Similarity + Weighted Variance (100x)

Usage:
    python scripts/train_microhubert.py \
        --audio_dir data/audio \
        --features_dir data/features_hubert_layer9 \
        --output_dir checkpoints/microhubert \
        --resume checkpoints/microhubert/microhubert_ep45.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import argparse
import torchaudio
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultra_low_bitrate_codec.models.micro_hubert import MicroHuBERT

class DistillationDataset(Dataset):
    def __init__(self, audio_dir, features_dir, max_len=32000*5): # 10s max
        self.audio_dir = audio_dir
        self.features_dir = features_dir
        self.max_len = max_len
        
        # Find matches
        self.files = []
        feat_files = glob.glob(f"{features_dir}/*.pt")
        for f in feat_files:
            name = os.path.basename(f).replace('.pt', '')
            self.files.append(name)
            
        print(f"Found {len(self.files)} paired samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        
        # Load Features
        feat_path = os.path.join(self.features_dir, f"{name}.pt")
        feat_data = torch.load(feat_path, map_location='cpu')
        features = feat_data['features'] # (T_frames, 768)
        
        # Use map created in init
        audio_path = self.audio_map.get(name)
        if audio_path is None:
             # Fallback: dummy
             return torch.zeros(16000), torch.zeros(50, 768)

        try:
            wav, sr = sf.read(audio_path)
            wav = torch.tensor(wav, dtype=torch.float32)
            if wav.dim() > 1: wav = wav.mean(dim=0)
            if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
            
            # Align
            # HuBERT rate is 320 samples/frame
            target_samples = features.shape[0] * 320
            if wav.shape[0] > target_samples:
                wav = wav[:target_samples]
            elif wav.shape[0] < target_samples:
                wav = torch.nn.functional.pad(wav, (0, target_samples - wav.shape[0]))
                
            # Random Crop if too long
            if wav.shape[0] > self.max_len:
                # crop frames
                max_frames = self.max_len // 320
                start_frame = torch.randint(0, features.shape[0] - max_frames, (1,)).item()
                features = features[start_frame:start_frame+max_frames]
                wav = wav[start_frame*320 : (start_frame+max_frames)*320]
                
            return wav, features
            
        except:
            return torch.zeros(16000), torch.zeros(50, 768)

    # Helper to build map
    def build_map(self):
        self.audio_map = {}
        # This is slow for huge dirs, but fine for <50k
        for root, dirs, files in os.walk(self.audio_dir):
            for f in files:
                if f.endswith(('.wav', '.flac', '.m4a', '.mp3')):
                    base = os.path.splitext(f)[0]
                    self.audio_map[base] = os.path.join(root, f)

def collate(batch):
    # simple collate with padding
    # sort by length
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    wavs, feats = zip(*batch)
    
    # Pad wavs
    max_len = wavs[0].shape[0]
    wav_batch = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        wav_batch[i, :w.shape[0]] = w
        
    # Pad feats
    max_frames = feats[0].shape[0]
    feat_batch = torch.zeros(len(feats), max_frames, 768)
    # Mask
    mask = torch.zeros(len(feats), max_frames).bool()
    
    for i, f in enumerate(feats):
        feat_batch[i, :f.shape[0]] = f
        mask[i, :f.shape[0]] = True
        
    return wav_batch, feat_batch, mask

def main():
    # Enable TF32
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MicroHuBERT().to(device)
    print(f"MicroHuBERT Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        
        # Handle state dict keys (strip _orig_mod. or module. if present)
        state_dict = ckpt
        if 'model' in ckpt: state_dict = ckpt['model']
        clean_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('_orig_mod.', '').replace('module.', '')
            clean_state_dict[k] = v
                
        miss, unexp = model.load_state_dict(clean_state_dict, strict=False)
        print(f"Loaded checkpoint keys. Missing: {len(miss)}, Unexpected: {len(unexp)}")

        try:
            base = os.path.basename(args.resume)
            start_epoch = int(base.split('_ep')[1].split('.pt')[0]) + 1
            print(f"Resuming at epoch {start_epoch}")
        except:
            print("Could not infer epoch, starting at 0 but with loaded weights")
            
    # Compile
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)
    
    dataset = DistillationDataset(args.audio_dir, args.features_dir)
    dataset.build_map()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=True,
        num_workers=4, 
        collate_fn=collate
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            
        for batch in pbar:
            wav, target, mask = batch
            wav, target, mask = wav.to(device), target.to(device), mask.to(device)
            
            pred = model(wav)
            
            min_t = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_t]
            target = target[:, :min_t]
            mask = mask[:, :min_t]
            
            # 1. L1 Loss
            l1_loss = (pred - target).abs()[mask].mean()
            
            # 2. Cosine Similarity Loss
            pred_flat = pred[mask]
            target_flat = target[mask]
            cos_sim = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
            cos_loss = 1.0 - cos_sim
            
            # 3. Variance Loss (Force expand dynamic range)
            std_pred = pred_flat.std()
            std_target = target_flat.std()
            var_loss = (std_pred - std_target).abs()
            
            # 4. Delta Loss (Temporal Smoothness / Fix "Brain Damage" sound)
            # Penalty for jittery feature transitions
            delta_pred = pred[:, 1:] - pred[:, :-1]
            delta_target = target[:, 1:] - target[:, :-1]
            delta_mask = mask[:, 1:] & mask[:, :-1]
            delta_loss = (delta_pred - delta_target).abs()[delta_mask].mean()
            
            # Weighted Total Loss 
            # 20x variance weight to maintain distribution
            # 10x delta weight for temporal stability
            loss = l1_loss + cos_loss + (20.0 * var_loss) + (10.0 * delta_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(l1=l1_loss.item(), cos=cos_loss.item(), var=var_loss.item(), dlt=delta_loss.item(), total=loss.item())
            
        scheduler.step()
        print(f"Epoch {epoch} Loss: {total_loss/len(dataloader):.4f}")
        
        if epoch % 5 == 0:
            # Unwrap compile for saving
            torch.save(model.state_dict(), f"{args.output_dir}/microhubert_ep{epoch}.pt")

if __name__ == "__main__":
    main()
