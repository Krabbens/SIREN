import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import glob
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torchaudio
from ultra_low_bitrate_codec.models.flow_matching import FlowMatchingHead

def save_spectrogram(data, path):
    # data: (1026, T) or (1, 1026, T)
    if data.dim() == 3:
        data = data.squeeze(0)
    
    # Check if Complex STFT (1026 channels)
    if data.shape[0] == 1026:
        real = data[:513, :]
        imag = data[513:, :]
        mag = torch.sqrt(real**2 + imag**2 + 1e-6)
        
        # Aggressive clamping on Mag to avoid infs before Log?
        # No, Mag is positive.
        
        # Mel Scale
        mel_transform = torchaudio.transforms.MelScale(
            n_mels=80,
            sample_rate=16000,
            n_stft=513,
            f_min=0,
            f_max=8000
        ).to(data.device)
        
        mel = mel_transform(mag)
        
        # Log Mel
        log_mel = torch.log10(mel + 1e-5)
        
        # Normalize/Clamp for viz
        # Typical log-mel with defaults is around -4 to 2
        viz = log_mel.cpu().numpy()
        vmin, vmax = -4.0, 2.0
    else:
        # Fallback for other data types (e.g. if we were training on Mels directly)
        viz = data.cpu().numpy()
        vmin, vmax = -4.0, 4.0

    plt.figure(figsize=(10, 4))
    plt.imshow(viz, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



class FlowDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        print(f"Found {len(self.files)} flow samples")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], map_location='cpu')
            return data['target'], data['cond']
        except:
            # Return silence (-4.0) if broken, NOT zeros (which is mean/grey)
            # Assuming normalized range is roughly [-4, 4] where -4 is silence
            return torch.ones(1026, 100) * -4.0, torch.zeros(512, 100)

def collate_fn(batch):
    # Determine min length
    min_len = min([x[0].shape[-1] for x in batch])
    # ... allow simple truncation for collation ...
    min_len = min(min_len, 256) # Limit context
    
    mels = []
    conds = []
    
    for m, c in batch:
        start = 0
        if m.shape[-1] > min_len:
            start = torch.randint(0, m.shape[-1] - min_len + 1, (1,)).item()
        m_crop = m[:, start:start+min_len]
        c_crop = c[:, start:start+min_len]
        mels.append(m_crop)
        conds.append(c_crop)
        
    return torch.stack(mels), torch.stack(conds)

from ultra_low_bitrate_codec.models.discriminator import DiscriminatorR

class MelDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # DiscriminatorR is a 2D CNN (PatchGAN-like)
        # We process (B, 1, 80, T) Mels
        # Standard HiFiGAN config uses (1024, 120, 600) for R1
        # Here we just use one generic resolution appropriate for Mels
        self.disc = DiscriminatorR(resolution=(1024, 120, 600)) 
        
    def forward(self, x):
        # x: (B, 1, 80, T)
        return self.disc(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/flow_dataset")
    parser.add_argument("--output_dir", default="checkpoints_flow_matching")
    parser.add_argument("--lr", type=float, default=1e-4) # Generator LR (Reduced 5e-4 -> 1e-4 for stability)
    parser.add_argument("--lr_d", type=float, default=5e-5) # Discriminator LR (Reduced 2e-4 -> 5e-5 to prevent overpowering G)
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from (e.g. warmup)")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    
    # Model
    model = FlowMatchingHead(
        in_channels=1026,
        cond_channels=512,
        hidden_dim=256,
        depth=6,
        heads=8
    ).to(device)
    
    # Discriminator
    discriminator = MelDiscriminator().to(device)
    
    if args.resume_checkpoint:
        print(f"Resuming from {args.resume_checkpoint}...")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt, strict=False) # Simplified loading
    
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.8, 0.99))
    
    # Mel Transform for Loss
    mel_transform = torchaudio.transforms.MelScale(
        n_mels=80, sample_rate=16000, n_stft=513, f_min=0, f_max=8000
    ).to(device)

    def to_mel(x_complex):
        # x_complex: (B, 1026, T)
        real = x_complex[:, :513, :]
        imag = x_complex[:, 513:, :]
        mag = torch.sqrt(real**2 + imag**2 + 1e-6)
        mel = mel_transform(mag)
        log_mel = torch.log10(mel + 1e-5)
        return log_mel.unsqueeze(1) # (B, 1, 80, T)

    # Dataset
    dataset = FlowDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn
    )
    
    sigma_min = 1e-4
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        avg_loss_g = 0
        avg_loss_d = 0
        
        for mels, conds in pbar:
            mels = mels.to(device) # x1 (Data)
            conds = conds.to(device) # Conditioning
            
            B, C, T = mels.shape
            # Raw Complex STFT range is approx -40 to +40.
            # Flow Matching prefers N(0, 1). Scaling by 20.0 puts it in [-2, 2].
            mels = mels / 20.0
            
            # 1. Sample t ~ U[0, 1]
            t = torch.rand(B, device=device)
            
            # 2. Linear Interpolation (OT-CFM)
            x0 = torch.randn_like(mels)
            x1 = mels
            
            xt = (1 - (1 - sigma_min) * t[:, None, None]) * x0 + t[:, None, None] * x1
            ut = x1 - (1 - sigma_min) * x0
            
            # --- Generator Step ---
            vt = model(xt, t, conds)
            
            loss_flow = F.mse_loss(vt, ut)
            
            # Adversarial Loss (on estimated x1)
            # x1_est = x_t + (1 - t) * v_pred (Approximate integration for 1 step? No, directly form vector field)
            # Actually, flow matching v_t points to target.
            # v_t = (x1 - x0) / (1) roughly.
            # So x1_est = x_t + (1 - t) * vt
            x1_est = xt + (1.0 - t[:, None, None]) * vt
            
            mel_est = to_mel(x1_est * 20.0)
            mel_real = to_mel(x1 * 20.0)  # We calculate mel_real inside step now for correctness
            
            # --- Spectral Loss (L1 on Log-Mels) ---
            # This is crucial for Complex STFT. MSE on Re/Im is not enough.
            # We explicitly force the Estimated Target's Spectrogram to match ground truth.
            loss_spectral = F.l1_loss(mel_est, mel_real)
            
            # NOTE: D forward returns (score, fmaps), we just take score
            d_fake, _ = discriminator(mel_est)
            loss_adv = torch.mean((d_fake - 1.0) ** 2)
            
            # Weighted Loss
            # flow: 1.0
            # spectral: 45.0 (Matches typical scale of Mel Loss vs MSE)
            # adv: 1.0 (Standard GAN weight)
            loss_total = loss_flow + 45.0 * loss_spectral + 1.0 * loss_adv 
            
            optimizer_g.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_g.step()
            
            # --- Discriminator Step ---
            # Real Mels (already computed above?) No, we need to detach for D if we reused
            # But let's just recompute or use above.
            
            # Detach fake
            d_fake_det, _ = discriminator(mel_est.detach())
            d_real, _ = discriminator(mel_real)
            
            loss_d_real = torch.mean((d_real - 1.0) ** 2)
            loss_d_fake = torch.mean(d_fake_det ** 2)
            loss_d = loss_d_real + loss_d_fake
            
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            
            avg_loss_g += loss_total.item()
            avg_loss_d += loss_d.item()
            
            pbar.set_postfix({'L_G': f"{loss_total.item():.3f}", 'Spec': f"{loss_spectral.item():.3f}"})
            
            if global_step % 100 == 0:
                writer.add_scalar("loss/g_total", loss_total.item(), global_step)
                writer.add_scalar("loss/g_flow", loss_flow.item(), global_step)
                writer.add_scalar("loss/g_spec", loss_spectral.item(), global_step)
                writer.add_scalar("loss/g_adv", loss_adv.item(), global_step)
                writer.add_scalar("loss/d_total", loss_d.item(), global_step)
            global_step += 1
            
        print(f"Epoch {epoch} G: {avg_loss_g / len(dataloader):.4f} D: {avg_loss_d / len(dataloader):.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"flow_epoch{epoch}.pt"))
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f"disc_epoch{epoch}.pt"))
        
        # Visualization (Last batch)
        with torch.no_grad():
            x_t = torch.randn_like(mels[0:1])
            cond_viz = conds[0:1]
            steps = 50
            dt = 1.0 / steps
            
            for i in range(steps):
                t_scalar = torch.tensor([i / steps], device=device)
                v_pred = model(x_t, t_scalar, cond_viz)
                x_t = x_t + v_pred * dt
                
            # Unscale for visualization
            save_spectrogram(mels[0].cpu() * 20.0, os.path.join(args.output_dir, f"epoch_{epoch}_target.png"))
            save_spectrogram(x_t[0].cpu() * 20.0, os.path.join(args.output_dir, f"epoch_{epoch}_pred.png"))
            print(f"Saved visualization to {args.output_dir}/epoch_{epoch}_*.png")
        
        # Save backup
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"flow_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    main()
