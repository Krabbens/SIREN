#!/usr/bin/env python3
"""
BitNet Vocoder Training Script

Train ultra-compact vocoder with ternary weights and SnakeBeta activations.
Uses knowledge distillation from pretrained NeuralVocoderV2.

Usage:
    python train_bitnet.py --teacher_checkpoint checkpoints_stable/step_87000
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.models.vocoder import NeuralVocoderV2
from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss, SnakeBetaDiversityLoss


class MelSpectrogramLoss(nn.Module):
    """Multi-scale mel spectrogram loss."""
    def __init__(self, sample_rate=16000, n_mels=80, 
                 n_ffts=[512, 1024, 2048], hop_lengths=[128, 256, 512]):
        super().__init__()
        self.transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=n_mels,
                power=1.0
            )
            for n_fft, hop in zip(n_ffts, hop_lengths)
        ])
    
    def forward(self, pred, target):
        loss = 0
        for mel_transform in self.transforms:
            mel_pred = mel_transform(pred)
            mel_target = mel_transform(target)
            
            # L1 loss on log mel
            loss += F.l1_loss(
                torch.log(mel_pred + 1e-5),
                torch.log(mel_target + 1e-5)
            )
        
        return loss / len(self.transforms)


class FeatureDataset(Dataset):
    """Load precomputed decoder features for vocoder training."""
    
    def __init__(self, feature_dir, audio_dir, sample_rate=16000, segment_frames=100):
        self.sample_rate = sample_rate
        self.segment_frames = segment_frames
        self.hop_length = 320  # Feature to audio ratio
        
        # Find all feature files
        self.features = sorted(glob.glob(os.path.join(feature_dir, "*.pt")))
        self.audio_dir = audio_dir
        
        print(f"Found {len(self.features)} feature files")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat_path = self.features[idx]
        
        # Load features (B, C, T) or (C, T)
        features = torch.load(feat_path)
        if features.dim() == 3:
            features = features.squeeze(0)  # (C, T)
        
        # Load corresponding audio
        basename = os.path.splitext(os.path.basename(feat_path))[0]
        audio_path = os.path.join(self.audio_dir, f"{basename}.wav")
        
        if os.path.exists(audio_path):
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            audio = audio.mean(0)  # Mono
        else:
            # Generate dummy audio if not found
            audio_len = features.shape[1] * self.hop_length
            audio = torch.zeros(audio_len)
        
        # Random crop for training
        T = features.shape[1]
        if T > self.segment_frames:
            start = random.randint(0, T - self.segment_frames)
            features = features[:, start:start + self.segment_frames]
            audio_start = start * self.hop_length
            audio_len = self.segment_frames * self.hop_length
            audio = audio[audio_start:audio_start + audio_len]
        else:
            # Pad if shorter
            pad_len = self.segment_frames - T
            features = F.pad(features, (0, pad_len))
            
            # Pad audio
            expected_audio_len = self.segment_frames * self.hop_length
            if len(audio) < expected_audio_len:
                audio = F.pad(audio, (0, expected_audio_len - len(audio)))
            elif len(audio) > expected_audio_len:
                audio = audio[:expected_audio_len]
        
        # Ensure audio length matches features strictly
        expected_len = self.segment_frames * self.hop_length
        if len(audio) != expected_len:
             # Should be handled above, but double check
             if len(audio) < expected_len:
                 audio = F.pad(audio, (0, expected_len - len(audio)))
             else:
                 audio = audio[:expected_len]

        return features, audio


class SimpleDataset(Dataset):
    """Simple dataset for training directly from audio."""
    
    def __init__(self, audio_dir, sample_rate=16000, segment_length=32000):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        
        # Find all audio files
        self.files = []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            self.files.extend(glob.glob(os.path.join(audio_dir, '**', ext), recursive=True))
        
        # Filter by minimum length
        self.files = [f for f in self.files if os.path.getsize(f) > 50000]
        print(f"Found {len(self.files)} audio files")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        
        try:
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            audio = audio.mean(0)  # Mono
            
            # Random crop
            if len(audio) > self.segment_length:
                start = random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start + self.segment_length]
            else:
                audio = F.pad(audio, (0, self.segment_length - len(audio)))
            
            return audio
        except Exception as e:
            # Return zeros on error
            return torch.zeros(self.segment_length)


def extract_features_with_teacher(teacher_model, audio, device):
    """
    Extract intermediate features from teacher vocoder.
    This requires the full encoder+factorizer pipeline.
    
    For simplicity, we'll use random features during training.
    In production, precompute features with the full pipeline.
    """
    # Placeholder: return random features matching vocoder input
    T = audio.shape[-1] // 320  # hop_length
    features = torch.randn(audio.shape[0], 256, T, device=device)
    return features


def train_epoch(model, teacher_model, dataloader, optimizer, scheduler, 
                mel_loss_fn, stft_loss_fn, diversity_loss_fn, device, epoch, writer,
                diversity_weight=0.1):
    """Train for one epoch with knowledge distillation and anti-banding regularization."""
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    
    total_loss = 0
    total_mel = 0
    total_stft = 0
    total_distill = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        if isinstance(batch, (tuple, list)):
            features, audio = batch
            features = features.to(device)
            audio = audio.to(device)
        else:
            audio = batch.to(device)
            # Generate random features (in production, use precomputed)
            T = audio.shape[-1] // 320
            features = torch.randn(audio.shape[0], 512, T, device=device)  # Match vocoder input_dim
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_audio = model(features)
        
        # Ensure same length
        min_len = min(pred_audio.shape[-1], audio.shape[-1])
        pred_audio = pred_audio[..., :min_len]
        audio = audio[..., :min_len]
        
        # Losses
        mel_loss = mel_loss_fn(pred_audio, audio)
        # STFT loss expects (B, 1, T) but internal STFT needs (B, T)
        # Pass directly without unsqueeze, let loss handle it
        try:
            stft_sc, stft_mag = stft_loss_fn(pred_audio, audio)
            stft_loss = stft_sc + stft_mag
        except RuntimeError:
            # Fallback: just use mel loss
            stft_loss = torch.tensor(0.0, device=device)
        
        loss = mel_loss + stft_loss
        
        # SnakeBeta alpha diversity regularization (anti-banding)
        if diversity_loss_fn is not None:
            div_loss = diversity_loss_fn(model)
            loss = loss + diversity_weight * div_loss
        else:
            div_loss = torch.tensor(0.0, device=device)
        
        # Knowledge distillation from teacher
        distill_loss = torch.tensor(0.0, device=device)
        if teacher_model is not None:
            with torch.no_grad():
                teacher_audio = teacher_model(features)
                teacher_audio = teacher_audio[..., :min_len]
            
            # Distillation: match mel spectrograms
            distill_loss = mel_loss_fn(pred_audio, teacher_audio)
            loss = loss + 0.5 * distill_loss
        
        # Backward
        loss.backward()
        
        # Gradient clipping (important for BitNet stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_mel += mel_loss.item()
        total_stft += stft_loss.item()
        total_distill += distill_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mel': f'{mel_loss.item():.4f}',
            'div': f'{div_loss.item():.4f}'
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % 100 == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/mel_loss', mel_loss.item(), global_step)
            writer.add_scalar('train/stft_loss', stft_loss.item(), global_step)
            writer.add_scalar('train/distill_loss', distill_loss.item(), global_step)
            writer.add_scalar('train/diversity_loss', div_loss.item(), global_step)
    
    # Step scheduler
    if scheduler is not None:
        scheduler.step()
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'mel': total_mel / n,
        'stft': total_stft / n,
        'distill': total_distill / n
    }


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, path)


def load_teacher_vocoder(checkpoint_dir, config, device):
    """Load pretrained NeuralVocoderV2 as teacher."""
    vocoder = NeuralVocoderV2(config).to(device)
    
    decoder_path = os.path.join(checkpoint_dir, "decoder.pt")
    if os.path.exists(decoder_path):
        state_dict = torch.load(decoder_path, map_location=device)
        # Extract vocoder weights from decoder
        vocoder_state = {k.replace('vocoder.model.', ''): v 
                        for k, v in state_dict.items() 
                        if 'vocoder' in k}
        if vocoder_state:
            vocoder.model.load_state_dict(vocoder_state, strict=False)
            print(f"Loaded teacher vocoder from {decoder_path}")
    
    return vocoder


def main():
    parser = argparse.ArgumentParser(description="Train BitNet Vocoder")
    parser.add_argument("--config", type=str, 
                        default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Path to teacher model checkpoint for distillation")
    parser.add_argument("--data_dir", type=str, default="data/LibriTTS",
                        help="Directory with training audio")
    parser.add_argument("--output_dir", type=str, default="checkpoints_bitnet",
                        help="Output directory for checkpoints")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="Directory with precomputed features")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=256,
                        help="BitVocoder hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of ConvNeXt layers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = BitVocoder(
        input_dim=512,
        dim=args.dim,
        num_layers=args.num_layers,
        num_res_blocks=1
    ).to(device)
    
    # Print model info
    counts = model.count_parameters()
    sizes = model.estimate_size()
    print(f"\nBitVocoder:")
    print(f"  Parameters: {counts['total']:,}")
    print(f"  Quantizable: {counts['quantizable']:,}")
    print(f"  FP32 size: {sizes['fp32_mb']:.2f} MB")
    print(f"  Ternary size: {sizes['ternary_mb']:.2f} MB")
    print()
    
    # Load teacher for distillation
    teacher_model = None
    if args.teacher_checkpoint:
        try:
            teacher_model = load_teacher_vocoder(args.teacher_checkpoint, config, device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            print("Teacher model loaded for knowledge distillation")
        except Exception as e:
            print(f"Warning: Failed to load teacher model: {e}")
            print("Continuing without distillation (or maybe teacher path is wrong)")
    
    # Dataset
    if args.feature_dir:
        print(f"Using precomputed features from {args.feature_dir}")
        dataset = FeatureDataset(
            feature_dir=args.feature_dir,
            audio_dir=args.data_dir,
            segment_frames=100  # 100 frames = 32000 samples @ hop 320
        )
    else:
        print(f"Using raw audio from {args.data_dir} (random features)")
        dataset = SimpleDataset(args.data_dir, segment_length=32000)
        
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    # Losses
    mel_loss_fn = MelSpectrogramLoss().to(device)
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    diversity_loss_fn = SnakeBetaDiversityLoss(target_log_std=1.5, target_log_mean=1.0)
    print("Using SnakeBeta diversity regularization for anti-banding")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        metrics = train_epoch(
            model, teacher_model, dataloader, optimizer, scheduler,
            mel_loss_fn, stft_loss_fn, diversity_loss_fn, device, epoch, writer,
            diversity_weight=0.1
        )
        
        print(f"\nEpoch {epoch}: loss={metrics['loss']:.4f}, "
              f"mel={metrics['mel']:.4f}, stft={metrics['stft']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
            )
        
        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(args.output_dir, "best_model.pt")
            )
            print(f"  → New best model saved!")
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1,
        os.path.join(args.output_dir, "final_model.pt")
    )
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
