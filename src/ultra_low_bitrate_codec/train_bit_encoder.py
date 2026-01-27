#!/usr/bin/env python3
"""
Train BitEncoder (Ternary) via Distillation from MicroEncoder (Float).

Strategy:
1. Load pretrained MicroEncoder (Teacher)
2. Initialize BitEncoder (Student)
3. Distill knowledge:
   - Loss: L1 between student and teacher features
   - STE (Straight Through Estimator) handles gradients for ternary weights

Usage:
    python train_bit_encoder.py --teacher_checkpoint checkpoints/checkpoints_micro_encoder/best_model.pt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import glob
import random
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.micro_encoder import MicroEncoder
from models.bit_encoder import BitEncoder


class AudioDataset(Dataset):
    """Same dataset as MicroEncoder training."""
    def __init__(self, audio_dir, sample_rate=16000, segment_length=48000):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        
        self.files = []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            self.files.extend(glob.glob(os.path.join(audio_dir, '**', ext), recursive=True))
        
        self.files = [f for f in self.files if os.path.getsize(f) > 20000]
        print(f"Found {len(self.files)} audio files")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        try:
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            audio = audio.squeeze(0)
            
            if len(audio) > self.segment_length:
                start = random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start + self.segment_length]
            elif len(audio) < self.segment_length:
                audio = F.pad(audio, (0, self.segment_length - len(audio)))
            
            return audio
        except:
            return torch.zeros(self.segment_length)


def train_epoch(model, teacher, dataloader, optimizer, scheduler, device, epoch, writer):
    model.train()
    teacher.eval()
    
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, audio in enumerate(pbar):
        audio = audio.to(device)
        
        # Get teacher targets (float32 features)
        with torch.no_grad():
            target_features = teacher(audio)
            
        optimizer.zero_grad()
        
        # Student prediction (ternary weights used in forward)
        pred_features = model(audio)
        
        # Ensure lengths match
        min_len = min(pred_features.shape[1], target_features.shape[1])
        pred = pred_features[:, :min_len]
        target = target_features[:, :min_len]
        
        # Loss: Deep Feature Matching (L1)
        # Using L1 is standard for regression tasks in BitNet context
        loss = F.l1_loss(pred, target)
        
        loss.backward()
        
        # Clip gradients (crucial for BitNet stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        if batch_idx % 50 == 0:
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + batch_idx)
            
    if scheduler:
        scheduler.step()
        
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, default="data/audio")
    parser.add_argument("--output_dir", type=str, default="checkpoints/checkpoints_bit_encoder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4) # Lower LR for BitNet fine-tuning
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Teacher (MicroEncoder)
    print("Loading Teacher (MicroEncoder)...")
    teacher = MicroEncoder().to(device)
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    # Handle state dict keys if needed
    state = checkpoint['model_state_dict']
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    
    # 2. Init Student (BitEncoder)
    print("Initializing Student (BitEncoder)...")
    model = BitEncoder().to(device)
    
    # 3. Optimizer with parameter groups
    # Snake parameters need smaller LR
    snake_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'act' in name or 'alpha' in name or 'phi' in name or 'beta' in name:
            snake_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': snake_params, 'lr': args.lr * 0.1}
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Dataset
    dataset = AudioDataset(args.audio_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    best_loss = float('inf')
    
    print("Starting Distillation...")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, teacher, dataloader, optimizer, scheduler, device, epoch, writer)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print("  New best model saved!")
            
        # Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"))

    # Final
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print("Training complete.")

if __name__ == "__main__":
    main()
