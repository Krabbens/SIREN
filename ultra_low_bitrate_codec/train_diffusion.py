"""
Train TinyDiffusion Enhancer for mel-spectrogram refinement.

Flow:
1. Audio -> MelSpectrogram -> Add noise (forward diffusion)
2. TinyDiffusion predicts noise
3. Loss: MSE(noise_pred, noise)

After training, enhancer can refine degraded mel-spectrograms.
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

from models.tiny_diffusion import TinyDiffusionEnhancer


class MelDataset(Dataset):
    """Dataset that returns mel spectrograms from audio files."""
    def __init__(self, audio_dir, segment_length=32000, n_mels=80, hop_length=320):
        self.files = glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True)
        self.files = [f for f in self.files if os.path.getsize(f) > 50000]
        self.segment_length = segment_length
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=hop_length, n_mels=n_mels, power=1.0
        )
        print(f"Found {len(self.files)} files")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        try:
            audio, sr = torchaudio.load(self.files[idx])
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            audio = audio.mean(0)
            
            if len(audio) > self.segment_length:
                start = random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start+self.segment_length]
            else:
                audio = F.pad(audio, (0, self.segment_length - len(audio)))
            
            # Convert to mel
            mel = self.mel_transform(audio)
            
            # Log scale
            mel = torch.log(mel + 1e-5)
            
            return mel
        except:
            # Return dummy mel on error
            return torch.zeros(80, self.segment_length // 320)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="data/audio")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    model = TinyDiffusionEnhancer(
        n_mels=80,
        hidden_dim=args.hidden_dim,
        n_steps=1000
    ).to(device)
    
    # Count params
    counts = model.count_parameters()
    print(f"\nTinyDiffusionEnhancer:")
    print(f"  Parameters: {counts['total']:,}")
    print(f"  Quantizable: {counts['quantizable']:,}")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Data
    dataset = MelDataset(args.audio_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Logging
    os.makedirs("checkpoints_diffusion", exist_ok=True)
    writer = SummaryWriter("checkpoints_diffusion/tensorboard")
    
    print("Starting Training...")
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for mel in pbar:
            mel = mel.to(device)
            
            # Forward: predict noise
            noise_pred, noise = model(mel, return_noise=True)
            
            # Loss: MSE between predicted and actual noise
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            avg_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            global_step += 1
            
            if global_step % 100 == 0:
                writer.add_scalar("loss", loss.item(), global_step)
        
        scheduler.step()
        avg_loss /= len(dataloader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints_diffusion/best_model.pt")
            print("  → New best model saved!")
            
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints_diffusion/checkpoint_epoch{epoch+1}.pt")
    
    torch.save(model.state_dict(), "checkpoints_diffusion/final_model.pt")
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
