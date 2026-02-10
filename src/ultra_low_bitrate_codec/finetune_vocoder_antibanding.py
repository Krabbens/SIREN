"""
Finetune BitVocoder to Fix Banding Artifacts

This script finetunes an existing BitVocoder checkpoint with:
1. SnakeBeta alpha diversity regularization (prevents frequency clustering)
2. Enhanced phase head with per-frequency cumsum weights

Run: python finetune_vocoder_antibanding.py

The new phase head parameters will be randomly initialized, so finetuning
is required even with an existing checkpoint.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import glob
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.training.losses import (
    MultiResolutionSTFTLoss, 
    BandwiseMelLoss,
    SnakeBetaDiversityLoss
)


class AudioDataset(Dataset):
    """Simple audio dataset for finetuning."""
    def __init__(self, audio_dir, sample_rate=16000, segment_length=32000):
        self.files = glob.glob(f"{audio_dir}/**/*.wav", recursive=True)
        if not self.files:
            self.files = glob.glob(f"{audio_dir}/**/*.flac", recursive=True)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        print(f"Found {len(self.files)} audio files")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.files[idx])
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audio = audio.mean(0)  # Mono
        
        # Random crop
        if len(audio) > self.segment_length:
            start = torch.randint(0, len(audio) - self.segment_length, (1,)).item()
            audio = audio[start:start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - len(audio)))
        
        return audio


def load_config():
    with open("ultra_low_bitrate_codec/configs/sub100bps.yaml") as f:
        return yaml.safe_load(f)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    
    # Paths
    checkpoint_in = "checkpoints/checkpoints_bitnet/best_model.pt"
    checkpoint_out = "checkpoints/checkpoints_bitnet_antibanding"
    audio_dir = "data/audio"
    
    os.makedirs(checkpoint_out, exist_ok=True)
    
    # Create model with new architecture
    voc_conf = config['model'].get('vocoder', {})
    vocoder = BitVocoder(
        input_dim=512,
        dim=256,
        n_fft=1024,
        hop_length=320,
        num_layers=voc_conf.get('num_convnext_layers', 8),
        num_res_blocks=voc_conf.get('num_res_blocks', 3)
    ).to(device)
    
    # Load old checkpoint (partial - phase head is new)
    print(f"Loading checkpoint from {checkpoint_in}...")
    ckpt = torch.load(checkpoint_in, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    
    # Remove 'model.' prefix if present
    state = {k[6:] if k.startswith('model.') else k: v for k, v in state.items()}
    
    # Filter out phase_head keys - they have incompatible shapes and will be freshly initialized
    phase_head_keys = [k for k in state.keys() if 'phase_head' in k]
    for k in phase_head_keys:
        del state[k]
    print(f"Removed {len(phase_head_keys)} incompatible phase_head keys from checkpoint")
    
    # Load remaining weights
    missing, unexpected = vocoder.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint. Missing keys (phase head initialized fresh): {len(missing)}")
    
    # Loss functions
    mr_stft_loss = MultiResolutionSTFTLoss().to(device)
    mel_loss = BandwiseMelLoss(device=str(device)).to(device)
    diversity_loss = SnakeBetaDiversityLoss()
    
    # Dataset
    dataset = AudioDataset(audio_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # Optimizer with different LR for new vs old parameters
    old_params = []
    new_params = []
    for name, p in vocoder.named_parameters():
        if 'phase_head' in name:
            new_params.append(p)
        else:
            old_params.append(p)
    
    optimizer = torch.optim.AdamW([
        {'params': old_params, 'lr': 1e-5},      # Lower LR for existing params
        {'params': new_params, 'lr': 1e-4},       # Higher LR for new phase head
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # Training config
    num_steps = 2000
    log_interval = 50
    save_interval = 500
    diversity_weight = 0.1
    
    print(f"\nFinetuning for {num_steps} steps with anti-banding losses...")
    print(f"  Diversity loss weight: {diversity_weight}")
    print(f"  Old params LR: 1e-5, New params LR: 1e-4")
    
    vocoder.train()
    step = 0
    best_loss = float('inf')
    
    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break
            
            audio = batch.to(device)
            
            # Forward: we need features, so use a simple mel -> vocoder approach
            # For proper training you'd use MicroEncoder + Adapter
            # Here we use mel as a proxy feature for quick finetuning
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
            ).to(device)
            
            mel = mel_transform(audio)  # (B, 80, T)
            
            # Project mel to vocoder input dim (simple linear projection)
            if not hasattr(main, 'mel_proj'):
                main.mel_proj = nn.Linear(80, 512).to(device)
            
            mel = mel.transpose(1, 2)  # (B, T, 80)
            features = main.mel_proj(mel)  # (B, T, 512)
            features = features.transpose(1, 2)  # (B, 512, T)
            
            # Generate audio
            audio_rec = vocoder(features)
            
            # Align lengths
            min_len = min(audio.shape[-1], audio_rec.shape[-1])
            audio = audio[..., :min_len]
            audio_rec = audio_rec[..., :min_len]
            
            # Losses
            sc, mag = mr_stft_loss(audio_rec, audio)
            mel_l = mel_loss(audio_rec, audio)
            div_l = diversity_loss(vocoder)
            
            loss = sc + mag + mel_l + diversity_weight * div_l
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vocoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            if step % log_interval == 0:
                print(f"Step {step}: loss={loss.item():.4f} (sc={sc.item():.4f}, "
                      f"mag={mag.item():.4f}, mel={mel_l.item():.4f}, div={div_l.item():.4f})")
            
            # Save
            if step % save_interval == 0 and step > 0:
                save_path = f"{checkpoint_out}/checkpoint_step{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': vocoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, save_path)
                print(f"Saved {save_path}")
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save({
                        'step': step,
                        'model_state_dict': vocoder.state_dict(),
                    }, f"{checkpoint_out}/best_model.pt")
                    print(f"New best model saved!")
            
            step += 1
    
    # Final save
    torch.save({
        'step': step,
        'model_state_dict': vocoder.state_dict(),
    }, f"{checkpoint_out}/final_model.pt")
    print(f"\nTraining complete. Final model saved to {checkpoint_out}/final_model.pt")
    print(f"Best model at {checkpoint_out}/best_model.pt")


if __name__ == "__main__":
    main()
