#!/usr/bin/env python3
"""
Train MicroEncoder via Knowledge Distillation from HuBERT.

The MicroEncoder learns to produce representations similar to HuBERT layer 9,
enabling it to replace HuBERT in the SIREN pipeline.

Training Strategy:
1. Load HuBERT and freeze it
2. For each audio batch:
   - Extract HuBERT features (target)
   - Run MicroEncoder on raw audio (prediction)
   - Compute distillation loss (L1 + Cosine)
3. Optionally: joint training with downstream reconstruction loss

Usage:
    python train_micro_encoder.py --audio_dir data/audio --epochs 50
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import glob
import random
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.micro_encoder import MicroEncoder


class AudioDataset(Dataset):
    """Dataset for loading raw audio files."""
    
    def __init__(self, audio_dir, sample_rate=16000, segment_length=48000, min_length=16000):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.min_length = min_length
        
        # Find all audio files
        self.files = []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            self.files.extend(glob.glob(os.path.join(audio_dir, '**', ext), recursive=True))
        
        # Filter by minimum size
        self.files = [f for f in self.files if os.path.getsize(f) > 20000]
        print(f"Found {len(self.files)} audio files")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        
        try:
            audio, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            audio = audio.squeeze(0)
            
            # Random crop or pad
            if len(audio) > self.segment_length:
                start = random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start + self.segment_length]
            elif len(audio) < self.segment_length:
                audio = F.pad(audio, (0, self.segment_length - len(audio)))
            
            return audio
            
        except Exception as e:
            # Return silence on error
            return torch.zeros(self.segment_length)


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
    - L1 loss for feature matching
    - Cosine similarity loss for direction matching
    """
    def __init__(self, l1_weight=1.0, cosine_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.cosine_weight = cosine_weight
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, D) predicted features
            target: (B, T, D) target features (from HuBERT)
        """
        # Ensure same length
        min_len = min(pred.shape[1], target.shape[1])
        pred = pred[:, :min_len]
        target = target[:, :min_len]
        
        # L1 loss
        l1_loss = F.l1_loss(pred, target)
        
        # Cosine similarity loss (1 - cosine_sim)
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
        cosine_loss = 1 - cosine_sim
        
        total_loss = self.l1_weight * l1_loss + self.cosine_weight * cosine_loss
        
        return total_loss, {
            'l1': l1_loss.item(),
            'cosine_sim': cosine_sim.item(),
            'total': total_loss.item()
        }


def load_hubert(device):
    """Load HuBERT model for feature extraction."""
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    
    print("Loading HuBERT...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()
    
    # Freeze HuBERT
    for param in model.parameters():
        param.requires_grad = False
    
    return processor, model


@torch.no_grad()
def extract_hubert_features(audio, processor, hubert, device, layer=9):
    """
    Extract features from HuBERT layer.
    
    Args:
        audio: (B, T) raw waveform
        processor: HuBERT feature extractor
        hubert: HuBERT model
        device: torch device
        layer: which layer to extract (default: 9)
    
    Returns:
        features: (B, T', 768) HuBERT features
    """
    # Process audio
    inputs = processor(
        audio.cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    input_values = inputs.input_values.to(device)
    
    # Extract features
    outputs = hubert(input_values, output_hidden_states=True)
    features = outputs.hidden_states[layer]
    
    return features


def train_epoch(model, hubert, processor, dataloader, optimizer, scheduler, 
                loss_fn, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_l1 = 0
    total_cosine = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, audio in enumerate(pbar):
        audio = audio.to(device)
        
        optimizer.zero_grad()
        
        # Extract HuBERT features (target)
        hubert_features = extract_hubert_features(audio, processor, hubert, device)
        
        # Forward through MicroEncoder (prediction)
        micro_features = model(audio)
        
        # Compute loss
        loss, metrics = loss_fn(micro_features, hubert_features)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += metrics['total']
        total_l1 += metrics['l1']
        total_cosine += metrics['cosine_sim']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total']:.4f}",
            'l1': f"{metrics['l1']:.4f}",
            'cos': f"{metrics['cosine_sim']:.3f}"
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % 50 == 0:
            writer.add_scalar('train/loss', metrics['total'], global_step)
            writer.add_scalar('train/l1_loss', metrics['l1'], global_step)
            writer.add_scalar('train/cosine_sim', metrics['cosine_sim'], global_step)
    
    # Step scheduler
    if scheduler is not None:
        scheduler.step()
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'l1': total_l1 / n,
        'cosine_sim': total_cosine / n
    }


def main():
    parser = argparse.ArgumentParser(description="Train MicroEncoder")
    parser.add_argument("--audio_dir", type=str, default="data/audio",
                        help="Directory with training audio files")
    parser.add_argument("--output_dir", type=str, default="checkpoints/checkpoints_micro_encoder",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--segment_length", type=int, default=48000,
                        help="Audio segment length in samples (3 seconds at 16kHz)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load HuBERT (teacher)
    processor, hubert = load_hubert(device)
    
    # Create MicroEncoder (student)
    model = MicroEncoder().to(device)
    
    counts = model.count_parameters()
    sizes = model.estimate_size()
    print(f"\nMicroEncoder:")
    print(f"  Parameters: {counts['total']:,}")
    print(f"  FP32 Size: {sizes['fp32_mb']:.2f} MB")
    print(f"  Ternary Size: {sizes['ternary_mb']:.2f} MB")
    
    # Dataset
    dataset = AudioDataset(
        audio_dir=args.audio_dir,
        segment_length=args.segment_length
    )
    
    if len(dataset) == 0:
        print(f"ERROR: No audio files found in {args.audio_dir}")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Loss
    loss_fn = DistillationLoss(l1_weight=1.0, cosine_weight=0.5)
    
    # Optimizer with parameter groups
    # Snake parameters (frequencies/phase) need smaller LR to avoid instability
    snake_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'act' in name or 'alpha' in name or 'phi' in name or 'beta' in name:
            snake_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': snake_params, 'lr': args.lr * 0.1}  # 10x smaller LR for Snake
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Training loop
    best_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        metrics = train_epoch(
            model, hubert, processor, dataloader, optimizer, scheduler,
            loss_fn, device, epoch, writer
        )
        
        print(f"\nEpoch {epoch}: loss={metrics['loss']:.4f}, "
              f"l1={metrics['l1']:.4f}, cosine={metrics['cosine_sim']:.3f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"))
        
        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  → New best model saved!")
    
    # Save final
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
    }, os.path.join(args.output_dir, "final_model.pt"))
    
    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
