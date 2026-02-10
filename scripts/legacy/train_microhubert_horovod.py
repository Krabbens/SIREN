#!/usr/bin/env python3
"""
Train MicroHuBERT via Distillation (Horovod)

Trains the <5MB MicroHuBERT to output vectors matching Official HuBERT Layer 9.

Usage:
    horovodrun -np 2 -H host1:1,host2:1 python scripts/train_microhubert_horovod.py ...
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

import horovod.torch as hvd

from ultra_low_bitrate_codec.models.micro_hubert import MicroHuBERT

class DistillationDataset(Dataset):
    def __init__(self, audio_dir, features_dir, max_len=32000*5):
        self.audio_dir = audio_dir
        self.features_dir = features_dir
        self.max_len = max_len
        
        self.files = []
        feat_files = glob.glob(f"{features_dir}/*.pt")
        
        self.audio_map = {}
        for ext in ['*.wav', '*.flac', '*.m4a', '*.mp3', '*.ogg']:
            for af in glob.glob(f"{audio_dir}/**/{ext}", recursive=True):
                name = os.path.splitext(os.path.basename(af))[0]
                self.audio_map[name] = af
        
        for f in feat_files:
            name = os.path.basename(f).replace('.pt', '')
            if name in self.audio_map:
                self.files.append(name)
            
        if hvd.rank() == 0:
            print(f"Found {len(self.files)} paired samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        
        feat_path = os.path.join(self.features_dir, f"{name}.pt")
        feat_data = torch.load(feat_path, map_location='cpu')
        features = feat_data['features']
        
        audio_path = self.audio_map.get(name)
        if audio_path is None:
             return torch.zeros(16000), torch.zeros(50, 768)

        try:
            wav, sr = sf.read(audio_path)
            wav = torch.tensor(wav, dtype=torch.float32)
            if wav.dim() > 1: wav = wav.mean(dim=0)
            if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
            
            target_samples = features.shape[0] * 320
            if wav.shape[0] > target_samples:
                wav = wav[:target_samples]
            elif wav.shape[0] < target_samples:
                wav = torch.nn.functional.pad(wav, (0, target_samples - wav.shape[0]))
                
            if wav.shape[0] > self.max_len:
                max_frames = self.max_len // 320
                start_frame = torch.randint(0, features.shape[0] - max_frames, (1,)).item()
                features = features[start_frame:start_frame+max_frames]
                wav = wav[start_frame*320 : (start_frame+max_frames)*320]
                
            return wav, features
            
        except:
            return torch.zeros(16000), torch.zeros(50, 768)


def collate(batch):
    wavs, feats = zip(*batch)
    
    max_wav = max(w.shape[0] for w in wavs)
    max_feat = max(f.shape[0] for f in feats)
    
    wav_batch = torch.zeros(len(wavs), max_wav)
    feat_batch = torch.zeros(len(feats), max_feat, feats[0].shape[1])
    
    for i, (w, f) in enumerate(zip(wavs, feats)):
        wav_batch[i, :w.shape[0]] = w
        feat_batch[i, :f.shape[0], :] = f
        
    return wav_batch, feat_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--features_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # Initialize Horovod
    hvd.init()
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    world_size = hvd.size()
    
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    if rank == 0:
        print(f"Horovod Distributed: {world_size} workers")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Model
    model = MicroHuBERT(hidden_dim=256, num_layers=4, num_heads=4, output_dim=768)
    model = model.to(device)
    
    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        print(f"MicroHuBERT Params: {total/1e6:.2f}M")
    
    # LR scaled by workers
    base_lr = args.lr * world_size
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    
    # Horovod DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        
        state_dict = ckpt
        if 'model' in ckpt: state_dict = ckpt['model']
        
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                k = k.replace('_orig_mod.', '')
            if k.startswith('module.'):
                k = k.replace('module.', '')
            clean_state_dict[k] = v
                
        miss, unexp = model.load_state_dict(clean_state_dict, strict=False)
        if rank == 0:
            print(f"Loaded checkpoint keys. Missing: {len(miss)}, Unexpected: {len(unexp)}")
        try:
            base = os.path.basename(args.resume)
            start_epoch = int(base.split('_ep')[1].split('.pt')[0]) + 1
            if rank == 0:
                print(f"Resuming at epoch {start_epoch}")
        except:
            pass
    
    # Broadcast initial state
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    # Dataset
    dataset = DistillationDataset(args.audio_dir, args.features_dir)
    
    # DistributedSampler for Horovod
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate, num_workers=4, pin_memory=True
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))
    criterion = nn.L1Loss()
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        if rank == 0:
            pbar = tqdm(loader, desc=f"Epoch {epoch}")
        else:
            pbar = loader
            
        for batch_idx, (wavs, targets) in enumerate(pbar):
            wavs = wavs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            features = model(wavs)
            
            min_len = min(features.shape[1], targets.shape[1])
            features = features[:, :min_len, :]
            targets = targets[:, :min_len, :]
            
            loss = criterion(features, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix(loss=loss.item())
        
        if rank == 0:
            ckpt_path = os.path.join(args.output_dir, f"microhubert_ep{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved {ckpt_path}")
            
            if epoch > 5:
                old_ckpt = os.path.join(args.output_dir, f"microhubert_ep{epoch-5}.pt")
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)


if __name__ == "__main__":
    main()
