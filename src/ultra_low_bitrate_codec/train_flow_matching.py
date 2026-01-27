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
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuser

def save_spectrogram(data, path):
    # data: (80, T) or (1, 80, T)
    if data.dim() == 3:
        data = data.squeeze(0)
    
    viz = data.cpu().numpy()
    vmin, vmax = -8.0, 2.0 # Adjusted for Log-Mel range

    plt.figure(figsize=(10, 4))
    plt.imshow(viz, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



class FlowDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        print(f"Found {len(self.files)} flow samples")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], map_location='cpu')
            # 'mel' (80, T), 'sem' (D_s, T_s), 'pro' (D_p, T_p), 'spk' (D_spk)
            return data['mel'], data['sem'], data['pro'], data['spk']
        except:
            return torch.ones(80, 100) * -11.5, torch.zeros(8, 25), torch.zeros(8, 12), torch.zeros(256)

def collate_fn(batch):
    # Determine min length for Mel
    min_len = min([x[0].shape[-1] for x in batch])
    min_len = min(min_len, 256) 
    
    mels = []
    sems = []
    pros = []
    spks = []
    
    for m, s, p, spk in batch:
        # Mel is (80, T)
        start = 0
        if m.shape[-1] > min_len:
            start = torch.randint(0, m.shape[-1] - min_len + 1, (1,)).item()
        
        # Calculate ratios for feature alignment
        # This is strictly proportional to Mel length
        m_crop = m[:, start:start+min_len]
        
        # For simplicity in training, we assume features are ALREADY aligned in time 
        # or we interpolate them inside the Fuser. 
        # But we need to crop them proportionally.
        # If Mel is T, Sem is T/4, Pro is T/8...
        # Wait, the dataset pre-computed them. Let's look at ratios.
        # Based on preprocess_data.py, HuBERT is 50Hz (320 hop at 16k). 
        # Mel is also 50Hz (aligned).
        # Factorizer might compress. Let's check config: sem factor 2, pro factor 8?
        
        # In this dataset, sem/pro are usually already aligned or use a fixed ratio.
        # Let's crop sem and pro using the same start/min_len but scaled.
        # Ratio sem/mel: s.shape[-1] / m.shape[-1]
        r_s = s.shape[-1] / m.shape[-1]
        r_p = p.shape[-1] / m.shape[-1]
        
        s_start = int(start * r_s)
        s_len = int(min_len * r_s)
        p_start = int(start * r_p)
        p_len = int(min_len * r_p)
        
        s_crop = s[:, s_start:s_start+s_len]
        p_crop = p[:, p_start:p_start+p_len]
        
        mels.append(m_crop)
        sems.append(s_crop) # (T_s, D_s) - no transpose needed if data is (T, D)
        pros.append(p_crop) # (T_p, D_p)
        spks.append(spk) # (D_spk)
        
    return torch.stack(mels), sems, pros, torch.stack(spks)

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
    parser.add_argument("--output_dir", default="checkpoints/checkpoints_flow_matching")
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
    # Config for ConditionalFlowMatching
    config = {
        'model': {
            'decoder': {
                'fusion_dim': 80,          # Out dim (Mel)
                'hidden_dim': 512,         # Internal dim
                'fusion_heads': 8,
                'dropout': 0.1
            },
            'flow_matching_layers': 8
        }
    }
    
    # Models
    model = ConditionalFlowMatching(config).to(device)
    
    # Fuser: 8 (sem) + 8 (pro) + 256 (spk) -> 512
    # These dims match ultra200bps_large.yaml and check_dims.py
    fuser = ConditionFuser(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512).to(device)
    
    # Discriminator
    discriminator = MelDiscriminator().to(device)
    
    if args.resume_checkpoint:
        print(f"Resuming from {args.resume_checkpoint}...")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt, strict=False) 
    
    # Joint optimization for Flow Model + Fuser
    optimizer_g = torch.optim.AdamW(
        list(model.parameters()) + list(fuser.parameters()), 
        lr=args.lr, betas=(0.8, 0.99)
    )
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.8, 0.99))
    
    # Mel Transform for Data Prep
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
        return log_mel # (B, 80, T)

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
    
    # Normalization constants for Log-Mel
    # Typical range is [-11.5, 2.0]
    # We map roughly [-12, 3] -> [-3, 3] for soft clamping approx 
    # Center: -4.5, Scale: 2.5
    MEL_MEAN = -5.0
    MEL_STD = 2.0
    
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        avg_loss_g = 0
        avg_loss_d = 0
        
        for mel_gt, sems, pros, spks in pbar:
            mel_gt = mel_gt.to(device) # (B, 80, T)
            spks = spks.to(device) # (B, 256)
            
            # Pad/Stack sems and pros (they are lists of T-varying tensors)
            # Actually collate_fn should have handle this, but for non-integer ratios 
            # we do it here or inside Fuser. Linear interpolate inside Fuser is best.
            sems_batch = torch.nn.utils.rnn.pad_sequence(sems, batch_first=True).to(device)
            pros_batch = torch.nn.utils.rnn.pad_sequence(pros, batch_first=True).to(device)
            
            # 1. Fuse Conditioning
            # target_len is the length of Mel spectrogram
            target_len = mel_gt.shape[-1]
            conds_model = fuser(sems_batch, pros_batch, spks, target_len) # (B, T, 512)
            
            # 2. Normalize Mel
            x1 = (mel_gt - MEL_MEAN) / MEL_STD # (B, 80, T)
            
            # Transpose for Model: (B, T, C)
            x1_model = x1.transpose(1, 2)
            
            B_size, T_size, C_size = x1_model.shape
            
            # 3. Sample t ~ U[0, 1]
            t = torch.rand(B_size, device=device)
            
            # 4. OT-CFM Formulation
            x0 = torch.randn_like(x1_model)
            
            # xt = (1 - (1-min)t) * x0 + t * x1
            # Simple version: t * x1 + (1-t) * x0
            t_expand = t[:, None, None]
            xt = (1 - (1 - sigma_min) * t_expand) * x0 + t_expand * x1_model
            ut = x1_model - (1 - sigma_min) * x0
            
            # --- Generator Step ---
            # Predict velocity field v_t(x_t, t|cond)
            vt = model(xt, t, conds_model)
            
            # Flow Matching Loss
            loss_flow = F.mse_loss(vt, ut)
            
            # Reconstruct x1 estimate for auxiliary losses
            # x1_est = x_t + (1 - t) * vt
            # Note: This is an approximation of the trajectory endpoint
            x1_est = xt + (1.0 - t_expand) * vt
            
            # Spectral Loss (L1 on Mels)
            # Since our target IS Mel, this is just L1 reconstruction
            loss_spectral = F.l1_loss(x1_est, x1_model)
            
            # Adversarial Loss
            # Disc works on (B, 1, 80, T)
            # x1_est is (B, T, 80) -> transpose -> (B, 80, T) -> unsqueeze -> (B, 1, 80, T)
            mel_est_viz = x1_est.transpose(1, 2).unsqueeze(1)
            d_fake, _ = discriminator(mel_est_viz)
            loss_adv = torch.mean((d_fake - 1.0) ** 2)
            
            # Total G Loss
            # Weights: Flow=1, Spec=45, Adv=1
            loss_total = loss_flow + 45.0 * loss_spectral + 1.0 * loss_adv 
            
            optimizer_g.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_g.step()
            
            # --- Discriminator Step ---
            # Real samples need distinct view
            mel_real_viz = x1.unsqueeze(1)
            
            d_fake_det, _ = discriminator(mel_est_viz.detach())
            d_real, _ = discriminator(mel_real_viz)
            
            loss_d_real = torch.mean((d_real - 1.0) ** 2)
            loss_d_fake = torch.mean(d_fake_det ** 2)
            loss_d = loss_d_real + loss_d_fake
            
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            
            avg_loss_g += loss_total.item()
            avg_loss_d += loss_d.item()
            
            pbar.set_postfix({'Flow': f"{loss_flow.item():.3f}", 'Spec': f"{loss_spectral.item():.3f}"})
            
            if global_step % 100 == 0:
                writer.add_scalar("loss/g_total", loss_total.item(), global_step)
                writer.add_scalar("loss/g_flow", loss_flow.item(), global_step)
                writer.add_scalar("loss/g_spec", loss_spectral.item(), global_step)
                writer.add_scalar("loss/d_total", loss_d.item(), global_step)
            global_step += 1
            
        print(f"Epoch {epoch} G: {avg_loss_g / len(dataloader):.4f} D: {avg_loss_d / len(dataloader):.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"flow_epoch{epoch}.pt"))
        torch.save(fuser.state_dict(), os.path.join(args.output_dir, f"fuser_epoch{epoch}.pt"))
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f"disc_epoch{epoch}.pt"))
        
        # Visualization (Last batch)
        with torch.no_grad():
            x_t = torch.randn_like(x1_model[0:1])
            # Use current batch conditioning (already fused)
            cond_viz = conds_model[0:1]
            steps = 50
            dt = 1.0 / steps
            
            for i in range(steps):
                t_scalar = torch.tensor([i / steps], device=device)
                v_pred = model(x_t, t_scalar, cond_viz)
                x_t = x_t + v_pred * dt
                
            # Denormalize: (x * 2.0) - 5.0
            # Target is x1 (B, 80, T) -> x1[0] is (80, T)
            tgt_viz = x1[0].cpu() * 2.0 + (-5.0)
            
            # Pred is x_t (B, T, 80) -> x_t[0] is (T, 80) -> transpose -> (80, T)
            pred_viz = x_t[0].transpose(0, 1).cpu() * 2.0 + (-5.0)
            
            save_spectrogram(tgt_viz, os.path.join(args.output_dir, f"epoch_{epoch}_target.png"))
            save_spectrogram(pred_viz, os.path.join(args.output_dir, f"epoch_{epoch}_pred.png"))
            print(f"Saved visualization to {args.output_dir}/epoch_{epoch}_*.png")
        
        # Save backup
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"flow_epoch{epoch+1}.pt"))
            torch.save(fuser.state_dict(), os.path.join(args.output_dir, f"fuser_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    main()
