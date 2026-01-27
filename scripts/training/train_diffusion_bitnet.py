
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import glob
from tqdm import tqdm
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.tiny_diffusion import TinyDiffusionEnhancer

class PairedMelDataset(Dataset):
    def __init__(self, data_dir, segment_length=128): # 128 frames ~ 2.5s
        self.clean_dir = os.path.join(data_dir, "clean")
        self.degraded_dir = os.path.join(data_dir, "degraded")
        self.files = [os.path.basename(f) for f in glob.glob(os.path.join(self.clean_dir, "*.pt"))]
        self.segment_length = segment_length
        print(f"Found {len(self.files)} paired files")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        fname = self.files[idx]
        try:
            clean = torch.load(os.path.join(self.clean_dir, fname), map_location='cpu')
            degraded = torch.load(os.path.join(self.degraded_dir, fname), map_location='cpu')
            
            # Ensure shape is (80, T)
            if clean.dim() == 3: clean = clean.squeeze(0)
            if degraded.dim() == 3: degraded = degraded.squeeze(0)

            # Crop or Pad
            T = clean.shape[1]
            if T > self.segment_length:
                start = random.randint(0, T - self.segment_length)
                clean = clean[:, start:start+self.segment_length]
                degraded = degraded[:, start:start+self.segment_length]
            elif T < self.segment_length:
                pad = self.segment_length - T
                clean = F.pad(clean, (0, pad))
                degraded = F.pad(degraded, (0, pad))
                
            return clean.contiguous(), degraded.contiguous()
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            return torch.zeros(80, self.segment_length), torch.zeros(80, self.segment_length)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/diffusion_pairs_bitnet")
    parser.add_argument("--checkpoint_dir", default="checkpoints/checkpoints_diffusion_finetune")
    parser.add_argument("--pretrained_ckpt", default="checkpoints/checkpoints_diffusion/best_model.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4) # Low LR for fine-tuning
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.checkpoint_dir, "tensorboard"))
    
    # Model
    model = TinyDiffusionEnhancer(n_mels=80, hidden_dim=64).to(device)
    if os.path.exists(args.pretrained_ckpt):
        print(f"Loading pretrained: {args.pretrained_ckpt}")
        model.load_state_dict(torch.load(args.pretrained_ckpt, map_location=device))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Dataset
    dataset = PairedMelDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for clean, degraded in pbar:
            clean = clean.to(device)
            degraded = degraded.to(device)
            
            # Goal: Input -> Degraded Mel (x_noisy)
            # Target -> Residual (Degraded - Clean) OR just train to denoising objective
            
            # METHOD: Iterative Denoising Training
            # The model is `predict_noise(x_noisy, t)`.
            # In Sample loop: x_prev = x - scale * noise_pred
            # So noise_pred should be proportional to (x - target).
            
            # Let's train with t=0 (direct restoration) mixed with t>0?
            # Or assume Degraded corresponds to some timestep T?
            # Let's try training it to predict the residual directly at random timesteps?
            # NO. The pre-trained model knows how to remove Gaussian noise at step t.
            # We want it to remove "BitNet Artifacts" which we pretend are noise at step t.
            
            # Simplest Formulation (Regression / Denoising Autoencoder)
            # We treat the diffusion model as a UNet.
            # We want predict_noise(degraded, t) ~ (degraded - clean)
            # But t affects the scaling in the UNet (time embeddings).
            
            # Let's sample random t, but feed `degraded` as the "noisy input".
            # And train it to predict `degraded - clean`.
            # This teaches it: "At time t, if you see this, the noise component is this".
            
            B = clean.shape[0]
            t = torch.randint(0, 1000, (B,), device=device) # Random timesteps
            
            # Target residual
            noise_target = degraded - clean
            
            # Predict
            noise_pred = model.predict_noise(degraded, t)
            
            loss = F.mse_loss(noise_pred, noise_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            global_step += 1
            
            if global_step % 100 == 0:
                writer.add_scalar("loss", loss.item(), global_step)
        
        avg_loss /= len(dataloader)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))
            print("Saved Best")
            
        if (epoch + 1) % 5 == 0:
             torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    main()
