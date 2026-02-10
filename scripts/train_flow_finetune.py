
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
import yaml

# Models
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.discriminator import DiscriminatorR

def save_spectrogram(data, path):
    if data.dim() == 3:
        data = data.squeeze(0)
    viz = data.cpu().numpy()
    vmin, vmax = -11.5, 3.0 
    plt.figure(figsize=(10, 4))
    plt.imshow(viz, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

class PrecomputedDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        print(f"Found {len(self.files)} samples in {data_dir}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], map_location='cpu')
            # 'mel' (80, T), 'cond' (512, T)
            return data['mel'], data['cond']
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(80, 100), torch.zeros(512, 100)

def collate_fn(batch):
    # Determine min length (usually alignment is handled in precompute, but let's be safe)
    # Both mel and cond should have same T
    min_len = min([x[0].shape[-1] for x in batch])
    min_len = min(min_len, 256) # Max crop length
    
    mels = []
    conds = []
    
    for m, c in batch:
        # m: (80, T), c: (512, T)
        start = 0
        if m.shape[-1] > min_len:
            start = torch.randint(0, m.shape[-1] - min_len + 1, (1,)).item()
            
        m_crop = m[:, start:start+min_len]
        c_crop = c[start:start+min_len] # c is (T, 512), slice dim 0
        
        mels.append(m_crop)
        conds.append(c_crop)
        
    return torch.stack(mels), torch.stack(conds)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Wrapper to match checkpoint keys (disc.*)
class MelDiscriminator(nn.Module):
    def __init__(self, resolution=(1024, 120, 600)):
        super().__init__()
        self.disc = DiscriminatorR(resolution=resolution)
        
    def forward(self, x):
        return self.disc(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/flow_dataset_student")
    parser.add_argument("--output_dir", default="checkpoints/flow_finetune_student")
    parser.add_argument("--pipeline_ckpt_dir", default="checkpoints/checkpoints_flow_v2")
    parser.add_argument("--config", default="src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    
    parser.add_argument("--lr", type=float, default=2e-5) # Lower LR for finetuning
    parser.add_argument("--lr_d", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    
    config = load_config(args.config)
    
    # 1. Models
    # Flow
    config['model']['decoder']['fusion_dim'] = 80 # Mel dim
    config['model']['decoder']['hidden_dim'] = 512
    model = ConditionalFlowMatching(config).to(device)
    
    # Load Flow Checkpoint
    fl_path = os.path.join(args.pipeline_ckpt_dir, "flow_epoch31.pt")
    if not os.path.exists(fl_path): fl_path = os.path.join(args.pipeline_ckpt_dir, "flow_epoch20.pt")
    print(f"Loading Flow: {fl_path}")
    ckpt = torch.load(fl_path, map_location=device)
    if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}, strict=False)
    
    # Discriminator
    discriminator = MelDiscriminator(resolution=(1024, 120, 600)).to(device)
    disc_path = os.path.join(args.pipeline_ckpt_dir, "disc_epoch31.pt")
    if os.path.exists(disc_path):
        print(f"Loading Discriminator: {disc_path}")
        d_ckpt = torch.load(disc_path, map_location=device)
        discriminator.load_state_dict(d_ckpt)
    else:
        print("Initializing Discriminator from scratch")
    
    # Fuser? 
    # In this new dataset setup, 'cond' is ALREADY fused (output of Fuser).
    # So we don't need Fuser here! We just train Flow to map Cond -> Mel.
    # This is much faster and simpler.

    # Optimize
    torch.set_float32_matmul_precision('high')
    # Compile
    print("Compiling models...")
    model = torch.compile(model)
    discriminator = torch.compile(discriminator)
    
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.8, 0.99))
    
    dataset = PrecomputedDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    sigma_min = 1e-4
    global_step = 0
    
    # Denormalization for Viz
    # We assume data is Log-Mel ~[-11.5, 2.0]
    # Flow Matching usually works on standard normal.
    # We need to normalize Mel to N(0, 1) approx.
    # Mean: -5.0, Std: 3.5 (Approx from previous scripts)
    MEAN = -5.0
    STD = 3.5
    
    print("Starting Finetuning...")
    
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        avg_flow = 0
        avg_spec = 0
        avg_adv = 0
        
        for mel_gt, cond_batch in pbar:
            mel_gt = mel_gt.to(device) # (B, 80, T)
            cond_batch = cond_batch.to(device) # (B, T, 512)
            
            # Normalize Mel for Training
            x1 = (mel_gt - MEAN) / STD
            x1_model = x1.transpose(1, 2) # (B, T, 80)
            
            B_size = x1.shape[0]
            
            # FM Interpolation
            t = torch.rand(B_size, device=device)
            x0 = torch.randn_like(x1_model)
            
            t_expand = t[:, None, None]
            # xt = (1 - (1-min)t)x0 + t*x1
            xt = (1 - (1 - sigma_min) * t_expand) * x0 + t_expand * x1_model
            ut = x1_model - (1 - sigma_min) * x0
            
            # Predict
            vt = model(xt, t, cond_batch)
            loss_flow = F.mse_loss(vt, ut)
            
            # Aux Losses
            x1_est = xt + (1.0 - t_expand) * vt
            loss_spectral = F.l1_loss(x1_est, x1_model)
            
            # Adversarial
            # x1_est (B, T, 80) -> (B, 1, 80, T)
            mel_est_viz = x1_est.transpose(1, 2).unsqueeze(1)
            d_fake, _ = discriminator(mel_est_viz)
            loss_adv = torch.mean((d_fake - 1.0) ** 2)
            
            loss_total = loss_flow + 45.0 * loss_spectral + 1.0 * loss_adv
            
            optimizer_g.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_g.step()
            
            # Disc Step
            d_fake_det, _ = discriminator(mel_est_viz.detach())
            mel_real_viz = x1.unsqueeze(1)
            d_real, _ = discriminator(mel_real_viz)
            
            loss_d = torch.mean((d_real - 1.0) ** 2) + torch.mean(d_fake_det ** 2)
            
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            
            avg_flow += loss_flow.item()
            avg_spec += loss_spectral.item()
            avg_adv += loss_adv.item()
            
            pbar.set_postfix(flow=loss_flow.item(), spec=loss_spectral.item(), adv=loss_adv.item())
            global_step += 1
            
        # Log
        print(f"Ep {epoch+1} | Flow: {avg_flow/len(dataloader):.4f} | Spec: {avg_spec/len(dataloader):.4f}")
        
        # Save
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/flow_ft_epoch{epoch+1}.pt")
            
        # Viz (Periodic)
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                # Take last batch sample
                cond_viz = cond_batch[0:1] # (1, T, 512)
                
                # Solve ODE
                mel_pred = model.solve_ode(cond_viz, steps=20, solver='euler', cfg_scale=1.0)
                # mel_pred (1, T, 80)
                
                # Denorm
                mel_pred = mel_pred.transpose(1, 2) # (1, 80, T)
                mel_pred = mel_pred * STD + MEAN
                
                mel_tgt = mel_gt[0].cpu() # (80, T)
                
                save_spectrogram(mel_tgt, f"{args.output_dir}/ep{epoch+1}_target.png")
                save_spectrogram(mel_pred, f"{args.output_dir}/ep{epoch+1}_pred.png")

if __name__ == "__main__":
    main()
