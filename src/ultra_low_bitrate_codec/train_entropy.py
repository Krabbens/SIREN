#!/usr/bin/env python3
"""
Standalone Entropy Model Training Script
Trains an autoregressive prior (EntropyModel) on the frozen latent codes of the pre-trained backbone.
Uses SnakeBeta activation in the Transformer if configured in entropy_coding.py.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from ultra_low_bitrate_codec.data.feature_dataset import PrecomputedFeatureDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def fix_state_dict(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--backbone_step", type=int, default=87000)
    parser.add_argument("--output_dir", type=str, default="checkpoints/checkpoints_entropy_snake")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4) # Lower LR for sensitive autoregressive training
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    
    # =========================================================================
    # LOAD BACKBONE (FROZEN)
    # =========================================================================
    print("Loading Backbone...")
    factorizer = InformationFactorizerV2(config).to(device)
    sem_vq = ResidualFSQ(config['model']['fsq_levels'], config['model']['rfsq_num_levels'], config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(config['model']['fsq_levels'], config['model']['rfsq_num_levels'], config['model']['prosody']['output_dim']).to(device)
    
    backbone_path = f"checkpoints/checkpoints_stable/step_{args.backbone_step}"
    if not os.path.exists(backbone_path):
         # Try looking one level up just in case
        if os.path.exists(f"../{backbone_path}"):
            backbone_path = f"../{backbone_path}"
    
    print(f"Reading from {backbone_path}")
    
    factorizer.load_state_dict(fix_state_dict(torch.load(f"{backbone_path}/factorizer.pt", map_location=device)), strict=False)
    sem_vq.load_state_dict(fix_state_dict(torch.load(f"{backbone_path}/sem_rfsq.pt", map_location=device)), strict=False)
    pro_vq.load_state_dict(fix_state_dict(torch.load(f"{backbone_path}/pro_rfsq.pt", map_location=device)), strict=False)
    
    factorizer.eval()
    sem_vq.eval()
    pro_vq.eval()
    for p in factorizer.parameters(): p.requires_grad = False
    for p in sem_vq.parameters(): p.requires_grad = False
    for p in pro_vq.parameters(): p.requires_grad = False
    
    # =========================================================================
    # ENTROPY MODEL
    # =========================================================================
    print("Initializing Entropy Model (with SnakeBeta if patched)...")
    entropy_model = EntropyModel(config).to(device)
    
    optimizer = optim.AdamW(entropy_model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # =========================================================================
    # DATASET
    # =========================================================================
    # Assume we run from root
    train_ds = PrecomputedFeatureDataset(
        features_dir=config['data']['feature_dir'],
        manifest_path=config['data']['train_manifest'],
        max_frames=500 
    )
    
    # Use smaller validation set or subset
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    writer = SummaryWriter(log_dir=f"{args.output_dir}/logs")
    
    global_step = 0
    best_loss = float('inf')
    
    print("Starting Training...")
    
    for epoch in range(args.epochs):
        entropy_model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            features = batch['features'].to(device) # (B, T, D)
            
            # Extract Indices
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    sem, pro, _ = factorizer(features)
                    _, _, sem_idx = sem_vq(sem) # (B, T, 1) or similar
                    _, _, pro_idx = pro_vq(pro)
                    
            # Flatten indices for entropy coding
            # sem_idx is (B, T, num_levels). We assume num_levels=1 or we flatten
            sem_idx = sem_idx.reshape(sem_idx.size(0), -1)
            pro_idx = pro_idx.reshape(pro_idx.size(0), -1)
            
            # Forward Entropy Model
            # Takes indices, converts to bytes, predicts next byte
            optimizer.zero_grad()
            
            # We construct targets for loss calculation
            # forward() returns logits for the sequence
            # But ProbabilisticLM predicts next token given context?
            # Let's inspect ProbabilisticLM.forward:
            #   mask = triu(...) -> Causal
            #   logits = head(h)
            # So logits[t] is prediction for targets[t]?
            # No, usually in AR: logits[t] predicts x[t+1] (or x[t] given x[:t]?)
            # Standard GPT: P(x_i | x_<i). 
            # If input is x, output h represents context x. 
            # If we project h[t] -> logits, we predict x[t+1]?
            # Usually: Input=x[:-1], Target=x[1:].
            
            # EntropyModel.forward returns (sem_logits, pro_logits) on FULL input?
            # Let's check estimate_bits in entropy_coding.py
            # It does: logits = model(bytes_seq[:, :-1])
            #          targets = bytes_seq[:, 1:]
            # So manual forward pass here should mimic that or use estimate_bits directly (but we need logits for training usually? estimate_bits gives bits/loss)
            
            # Let's use estimate_bits logic but return loss
            # Wait, estimate_bits returns BITS (detached?). 
            # Ah, in previous file view:
            # nll = F.cross_entropy(..., reduction='none')
            # total_nats = nll.sum()
            # total_bits = total_nats ...
            # If I trust estimate_bits uses grad, I can just minimize it.
            # But estimate_bits used .detach() on output? 
            # Lines 116/117: `sem_bits = calc_stream_bits(..., sem_idx)`
            # Inside `calc_stream_bits`: `logits = model(...)`, `nll = ...`.
            # Nothing detached *inside* the calculation.
            # So I can optimize `sem_bits.mean()`.
            
            sem_bits, pro_bits = entropy_model.estimate_bits(sem_idx, pro_idx)
            
            # Loss = Total Bits / (B * T * Duration?)
            # Just minimizing total bits is fine.
            loss = sem_bits.mean() + pro_bits.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            
            # Calculate BPS for display (Approx duration)
            # 50Hz for S/P approx
            # Duration ~ T / 50?
            # But T varies.
            # Roughly: Bits / Duration
            # We can use the batch duration
            # BPS = Loss / Duration
            # Assume 1 frame = 20ms (50Hz)
            # Just display Loss (Total Bits per sequence)
            
            metrics = {
                "loss": f"{loss.item():.2f}",
                "sem_bits": f"{sem_bits.mean().item():.1f}", 
                "pro_bits": f"{pro_bits.mean().item():.1f}"
            }
            pbar.set_postfix(metrics)
            
            if global_step % 50 == 0:
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/SemBits", sem_bits.mean().item(), global_step)
                writer.add_scalar("Train/ProBits", pro_bits.mean().item(), global_step)
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        scheduler.step()
        
        # Save Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(entropy_model.state_dict(), f"{args.output_dir}/entropy_best.pt")
            print(f"Saved Best Model (Loss: {best_loss:.4f})")
            
        torch.save(entropy_model.state_dict(), f"{args.output_dir}/entropy_latest.pt")

    print("Done. Saved to", args.output_dir)

if __name__ == "__main__":
    main()
