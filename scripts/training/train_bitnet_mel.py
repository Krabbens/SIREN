#!/usr/bin/env python3
"""
BitNet Mel-Vocoder Training Script
Train ultra-compact vocoder mapping 100-band Mel Spectrograms to 24kHz Audio.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import random
import soundfile as sf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.models.vocoder import NeuralVocoderV2
from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss, SnakeBetaDiversityLoss


class MelSpectrogramLoss(nn.Module):
    """Multi-scale mel spectrogram loss for 24kHz."""
    def __init__(self, sample_rate=24000, n_mels=100, 
                 n_ffts=[512, 1024, 2048], hop_lengths=[128, 256, 512]):
        super().__init__()
        self.transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=n_mels,
                power=2.0 # Match Input Feature Scaling (Power Spectrogram)
            )
            for n_fft, hop in zip(n_ffts, hop_lengths)
        ])
    
    def forward(self, pred, target):
        loss = 0
        for mel_transform in self.transforms:
            mel_pred = mel_transform(pred)
            mel_target = mel_transform(target)
            loss += F.l1_loss(torch.log(mel_pred + 1e-5), torch.log(mel_target + 1e-5))
        return loss / len(self.transforms)


class MelFeatureDataset(Dataset):
    """Load precomputed Mel features (100-band) and matching 24kHz Audio."""
    
    def __init__(self, feature_dir, audio_dir, sample_rate=24000, segment_frames=100):
        self.sample_rate = sample_rate
        self.segment_frames = segment_frames
        self.hop_length = 256  # Ratio for 24kHz Mel (n_fft=1024, hop=256)
        
        self.features = sorted(glob.glob(os.path.join(feature_dir, "*.pt")))
        self.audio_dir = audio_dir
        print(f"Found {len(self.features)} feature files")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat_path = self.features[idx]
        
        # Load precomputed dict
        try:
            data = torch.load(feat_path, map_location='cpu')
            # Handle both raw tensor and dict
            if isinstance(data, dict):
                features = data['mel'].float() # (1, T, 100) or (T, 100)
            else:
                features = data.float()
                
            if features.dim() == 3:
                features = features.squeeze(0)  # (T, 100)
            
            # We need (C, T) for BitVocoder internal logic
            features = features.transpose(0, 1) # (100, T)
            
            # Load corresponding audio
            basename = os.path.splitext(os.path.basename(feat_path))[0]
            audio_path = os.path.join(self.audio_dir, f"{basename}.wav")
            
            if os.path.exists(audio_path):
                audio, sr = sf.read(audio_path)
                audio = torch.from_numpy(audio).float()
                if sr != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
                if audio.dim() > 1:
                    audio = audio.mean(1)
            else:
                audio = torch.zeros(features.shape[1] * self.hop_length)
            
            # Random crop
            T = features.shape[1]
            target_audio_len = self.segment_frames * self.hop_length
            
            if T > self.segment_frames:
                start = random.randint(0, T - self.segment_frames)
                features = features[:, start:start + self.segment_frames]
                audio_start = start * self.hop_length
                audio = audio[audio_start : audio_start + target_audio_len]
            else:
                features = F.pad(features, (0, self.segment_frames - T))
                audio = F.pad(audio, (0, target_audio_len - len(audio)))
            
            # Final safety check
            if len(audio) < target_audio_len:
                audio = F.pad(audio, (0, target_audio_len - len(audio)))
            elif len(audio) > target_audio_len:
                audio = audio[:target_audio_len]
                
            return features, audio
        except Exception as e:
            print(f"Error loading {feat_path}: {e}")
            return torch.zeros(100, self.segment_frames), torch.zeros(self.segment_frames * self.hop_length)


def train_epoch(model, teacher_model, dataloader, optimizer, scheduler, 
                mel_loss_fn, stft_loss_fn, diversity_loss_fn, device, epoch, writer):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (features, audio) in enumerate(pbar):
        features, audio = features.to(device), audio.to(device)
        optimizer.zero_grad()
        
        pred_audio = model(features)
        min_len = min(pred_audio.shape[-1], audio.shape[-1])
        pred_audio, audio = pred_audio[..., :min_len], audio[..., :min_len]
        
        mel_loss = mel_loss_fn(pred_audio, audio)
        stft_sc, stft_mag = stft_loss_fn(pred_audio, audio)
        stft_loss = stft_sc + stft_mag
        
        # Anti-banding
        div_loss = diversity_loss_fn(model) if diversity_loss_fn else torch.tensor(0.0, device=device)
        
        loss = mel_loss + stft_loss + 0.1 * div_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mel': f'{mel_loss.item():.4f}'})
        
        if batch_idx % 100 == 0:
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + batch_idx)

    if scheduler: scheduler.step()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", default="data/flow_dataset_24k")
    parser.add_argument("--audio_dir", default="data/audio")
    parser.add_argument("--output_dir", default="checkpoints/checkpoints_bitnet_mel_v3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16) # Increased for stability
    parser.add_argument("--lr", type=float, default=2e-4) # Slightly higher than 1e-4 for faster startup, will decay
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    model = BitVocoder(
        input_dim=100, 
        dim=args.dim,
        num_layers=args.num_layers,
        num_res_blocks=2,
        hop_length=256 
    ).to(device)
    
    dataset = MelFeatureDataset(args.feature_dir, args.audio_dir, segment_frames=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Updated to power=2.0 to match Input Features (Log-Power-Mel)
    mel_loss_fn = MelSpectrogramLoss(sample_rate=24000, n_mels=100).to(device)
    # Note: MelSpectrogramLoss inside should be updated to use power=2.0.
    # But wait, MelSpectrogramLoss class definition is ABOVE. I need to update the Class __init__ or the instantiation.
    # The class hardcodes power=1.0 in __init__.
    # I should modify the Class definition too.
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    diversity_loss_fn = SnakeBetaDiversityLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, None, dataloader, optimizer, None, mel_loss_fn, stft_loss_fn, diversity_loss_fn, device, epoch, writer)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"mel_vocoder_epoch{epoch}.pt"))

if __name__ == "__main__":
    main()
