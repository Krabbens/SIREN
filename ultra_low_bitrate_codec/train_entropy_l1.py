
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import itertools

# Add project root
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from ultra_low_bitrate_codec.data.feature_dataset import PrecomputedFeatureDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_l1_entropy")
    parser.add_argument("--load_from", type=str, default="checkpoints_stable/step_87000")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # --- MODELS ---
    print(f"Loading pretrained models from {args.load_from}...")
    
    # 1. Factorizer (Frozen)
    factorizer = InformationFactorizerV2(config).to(device)
    sd = torch.load(f"{args.load_from}/factorizer.pt", map_location=device)
    factorizer.load_state_dict({k.replace("_orig_mod.", ""):v for k,v in sd.items()})
    factorizer.eval()
    for p in factorizer.parameters(): p.requires_grad = False
    
    # 2. Quantizers (Frozen)
    sem_rfsq = ResidualFSQ(config['model']['fsq_levels'], config['model']['rfsq_num_levels'], config['model']['semantic']['output_dim']).to(device)
    sd = torch.load(f"{args.load_from}/sem_rfsq.pt", map_location=device)
    sem_rfsq.load_state_dict({k.replace("_orig_mod.", ""):v for k,v in sd.items()})
    sem_rfsq.eval()
    for p in sem_rfsq.parameters(): p.requires_grad = False
    
    pro_rfsq = ResidualFSQ(config['model']['fsq_levels'], config['model']['rfsq_num_levels'], config['model']['prosody']['output_dim']).to(device)
    sd = torch.load(f"{args.load_from}/pro_rfsq.pt", map_location=device)
    pro_rfsq.load_state_dict({k.replace("_orig_mod.", ""):v for k,v in sd.items()})
    pro_rfsq.eval()
    for p in pro_rfsq.parameters(): p.requires_grad = False
    
    # 3. Entropy Model (Trainable)
    # We init FRESH to learn specific L1 patterns distributions from scratch (or fine-tune?)
    # Transforming L2..8 to 0 is a massive shift. Fresh start is safer/cleaner.
    entropy_model = EntropyModel(config).to(device)
    # entropy_model.train() # Default
    
    # Optimizer - Only Entropy params
    optimizer = optim.AdamW(entropy_model.parameters(), lr=1e-4) # Higher LR for fresh start
    
    # Dataset
    max_duration = config['data'].get('max_duration', 15.0)
    # HuBERT hop=320, sr=16000 => 50 fps.
    max_frames = int(max_duration * 50)
    
    train_dataset = PrecomputedFeatureDataset(
        manifest_path=config['data']['train_manifest'], 
        max_frames=max_frames,
        features_dir=config['data']['feature_dir']
    )
    val_dataset = PrecomputedFeatureDataset(
        manifest_path=config['data']['val_manifest'], 
        max_frames=max_frames,
        features_dir=config['data']['feature_dir']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    
    writer = SummaryWriter(log_dir=f"{args.checkpoint_dir}/logs")
    
    print("Starting L1-Entropy Training...")
    step = 0
    
    from tqdm import tqdm
    
    for epoch in itertools.count():
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            feats = batch['features'].to(device) # (B, T, D)
            
            # Extract & Quantize (Frozen)
            with torch.no_grad():
                sem, pro, _ = factorizer(feats)
                _, _, sem_idx = sem_rfsq(sem) 
                _, _, pro_idx = pro_rfsq(pro)
                
                # --- CRITICAL: MASK L2..L8 ---
                # Indices shape: (B, T, Levels)
                # Set Levels 1..7 to 0
                sem_idx[:, :, 1:] = 0
                pro_idx[:, :, 1:] = 0
                
            # Train Entropy
            # entropy_model(sem, pro) calculates loss internally
            # It will try to predict 0s for L2..8 (easy) and correct L1s (hard).
            # Loss is average over all bytes.
            
            optimizer.zero_grad()
            
            # Forward (returns logits)
            sem_logits, pro_logits = entropy_model(sem_idx, pro_idx)
            
            # Prepare Targets (Bytes)
            B = sem_idx.shape[0]
            sem_target_bytes = entropy_model.indices_to_bytes(sem_idx.view(B, -1)) # (B, T*3)
            pro_target_bytes = entropy_model.indices_to_bytes(pro_idx.view(B, -1))
            
            # Loss (Shifted)
            # Logits: (B, L, 256). Predict next token.
            # l_i predicts x_{i+1}
            loss_fn = nn.CrossEntropyLoss()
            
            # Reshape for CE: (N, C) vs (N)
            # Remove last logit, remove first target
            sem_loss = loss_fn(sem_logits[:, :-1].reshape(-1, 256), sem_target_bytes[:, 1:].reshape(-1))
            pro_loss = loss_fn(pro_logits[:, :-1].reshape(-1, 256), pro_target_bytes[:, 1:].reshape(-1))
            
            loss = sem_loss + pro_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            
            loss_bits = loss.item() / 0.693147 # ln(2)
            
            if step % 10 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'bits': f"{loss_bits:.2f}"})
                
            if step % 50 == 0:
                print(f"Step {step}: Loss {loss.item():.4f} ({loss_bits:.2f} bits)")
                writer.add_scalar("Train/Loss", loss.item(), step)
                
            if step % 200 == 0:
                # Validation
                entropy_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for v_batch in val_loader:
                        v_feats = v_batch['features'].to(device)
                        sem, pro, _ = factorizer(v_feats)
                        _, _, sem_idx = sem_rfsq(sem)
                        _, _, pro_idx = pro_rfsq(pro)
                        sem_idx[:, :, 1:] = 0
                        pro_idx[:, :, 1:] = 0
                        
                        s_logits, p_logits = entropy_model(sem_idx, pro_idx)
                        B = sem_idx.shape[0]
                        s_tgt = entropy_model.indices_to_bytes(sem_idx.view(B, -1))
                        p_tgt = entropy_model.indices_to_bytes(pro_idx.view(B, -1))
                        sl = F.cross_entropy(s_logits[:, :-1].reshape(-1, 256), s_tgt[:, 1:].reshape(-1))
                        pl = F.cross_entropy(p_logits[:, :-1].reshape(-1, 256), p_tgt[:, 1:].reshape(-1))
                        
                        val_loss += (sl + pl).item()
                
                val_loss /= len(val_loader)
                print(f"VALIDATION Step {step}: {val_loss:.4f}")
                writer.add_scalar("Val/Loss", val_loss, step)
                
                # Save
                torch.save(entropy_model.state_dict(), f"{args.checkpoint_dir}/entropy_l1_latest.pt")
                entropy_model.train()
                
            if step >= 10000: # Short training
                print("Training complete.")
                return

if __name__ == "__main__":
    main()
