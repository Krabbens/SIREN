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
import glob
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def save_spectrogram(data, path, title="Spectrogram"):
    # data: (T, D)
    viz = data.T.cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(viz, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Models
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.bitlinear import BitConv2d # New import
from transformers import Wav2Vec2FeatureExtractor, HubertModel

class ConditionFuser(nn.Module):
    def __init__(self, sem_dim, pro_dim, spk_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(sem_dim + pro_dim + spk_dim, out_dim)
        # Learnable smoothing to prevent aliasing from upsampling
        # Using nn.Conv2d (Float) because BitConv2d's RMSNorm on 1 channel causes collapse to [1, -1] and silence.
        # We treat the spectrogram as an image with 1 channel. (B, 1, T, D)
        self.smooth = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        
        # Initialize latent weights with 2D Gaussian to encourage smoothing
        with torch.no_grad():
            sigma = 1.5
            k = 5
            x = torch.arange(k).float() - k//2
            y = torch.arange(k).float() - k//2
            xv, yv = torch.meshgrid(x, y, indexing='ij')
            gauss = torch.exp(-0.5 * (xv**2 + yv**2) / sigma**2)
            gauss = gauss / gauss.sum()
            # self.smooth.weight is (Out, In, H, W) -> (1, 1, 5, 5)
            self.smooth.weight.data.copy_(gauss.view(1, 1, k, k))

    def forward(self, s, p, spk, target_len):
        # s, p: (B, T, D)
        # spk: (B, D)
        
        # Upsample to target length smoothly (Fixes blocky resolution)
        s = s.transpose(1, 2) # (B, D, T)
        p = p.transpose(1, 2)
        
        s = F.interpolate(s, size=target_len, mode='linear', align_corners=False)
        p = F.interpolate(p, size=target_len, mode='linear', align_corners=False)
        
        s = s.transpose(1, 2) # (B, T, D)
        p = p.transpose(1, 2)
        
        spk = spk.unsqueeze(1).expand(-1, target_len, -1)
        # Concat
        cat = torch.cat([s, p, spk], dim=-1)
        x = self.proj(cat) # (B, T, D)

        # Apply Conv2d smoothing
        # Reshape to (B, 1, T, D)
        x = x.unsqueeze(1)
        x = self.smooth(x)
        x = x.squeeze(1) # (B, T, D)

        return x

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class CachedDataset(Dataset):
    def __init__(self, data_dir, segment_length=None): 
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        print(f"Found {len(self.files)} cached files")
        self.max_len = 1000 # Unused
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        try:
            # We must load on CPU first
            d = torch.load(self.files[idx], map_location='cpu')
            return d
        except:
            return None

class AudioDataset(Dataset):
    def __init__(self, data_dir, segment_length=16000*2): 
        self.files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
        self.files = [f for f in self.files if os.path.getsize(f) > 32000]
        self.segment_length = segment_length
        print(f"Found {len(self.files)} files")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        try:
            wav, sr = torchaudio.load(self.files[idx])
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.mean(0, keepdim=True)
            
            if wav.shape[1] < self.segment_length:
                pad = self.segment_length - wav.shape[1]
                wav = F.pad(wav, (0, pad))
            else:
                start = random.randint(0, wav.shape[1] - self.segment_length)
                wav = wav[:, start:start+self.segment_length]
            
            wav = wav / (wav.abs().max() + 1e-6) * 0.95
            return wav.squeeze(0)
        except:
            return torch.zeros(self.segment_length)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to foundation model (decoder.pt)")
    parser.add_argument("--data_dir", default="data/flow_dataset_24k")
    parser.add_argument("--output_dir", default="checkpoints/checkpoints_flow_new")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2) 
    parser.add_argument("--lr", type=float, default=4e-4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    
    config = load_config(args.config)
    
    print("Initializing Foundation Models (Frozen)...")
    # 1. Foundation Models (Conditioning and Target)
    # We use SpeechDecoderV2 just to hold Reconstructor.
    # Note: We need FeatureReconstructor to get the TARGET latent features that we want to generate.
    # Actually, wait. Flow Matching replaces the diffusion/GAN.
    # Input: Condition (sem, pro, spk)
    # Target: Latent Features (before Vocoder) OR Spectrogram.
    # Since we have a BitVocoder trained on Fusion Tokens, we should actually train Flow Matching
    # to generate the *Fused Tokens* (output of FeatureReconstructor) from (sem, pro, spk) + noise?
    # NO. FeatureReconstructor IS deterministic. It combines (sem, pro, spk).
    # The variable part is usually the high-freq detail.
    # But wait, FeatureReconstructor outputs deterministic embeddings.
    # If the user wants "detail", maybe we should train Flow Matching on MEL SPECTROGRAMS directly?
    # Then use a HiFiGAN vocoder (or BitVocoder trained on Mel)?
    # User's BitVocoder takes "Fusion Features" (dim=256) as input.
    # So we should train FM to generate Fusion Features? 
    # But FeatureReconstructor already generates Fusion Features deterministically from tokens.
    # ERROR in logic: If FeatureReconstructor is deterministic, where is the "lost detail"?
    # The lost detail is in the Quantization (Factorizer).
    # The (sem, pro, spk) are quantized. The "Target" should be UNQUANTIZED or "GT" features?
    # No, we only have audio.
    #
    # Improved Plan:
    # 1. Extract GT HuBERT features -> Factorizer -> (sem, pro, spk) [Quantized Condition]
    # 2. Extract GT Fusion Features? We don't have them. We have Audio.
    # 3. Audio -> Encodec/DAC/Mel? 
    #
    # Wait, the user has a `BitVocoder` that takes `fusion_dim` input.
    # We can train FM to map (sem, pro, spk) -> [GT Feature Representation?]
    # What is the GT? We only have (HuBERT -> Tokens). 
    # We lack a "Ground Truth" for the intermediate latent space because it's learned end-to-end.
    #
    # Alternative: Latent Flow Matching.
    # We need a target satisfying: Vocoder(Target) = HighQualityAudio.
    # We don't have such a target yet because our previous training was joint.
    #
    # PIVOT: We must train Flow Matching to generate MEL SPECTROGRAMS (standard high quality rep).
    # Then finetune BitVocoder to take Mels? Or use standard Vocos/HifiGAN.
    # User wants "detail". Mel Spectrograms are good targets.
    #
    # Let's adjust:
    # Target: Mel Spectrograms (80 bins).
    # Condition: (sem, pro, spk) embeddings.
    # Model: Flow Matching U-Net/DiT mapping Cond -> Mel.
    # Vocoder: Retrain/Finetune BitVocoder on Mel, or use pre-trained HifiGAN.
    # This guarantees quality because Mels are dense.
    
    # Let's use 100-band Mel as target (dim=100) for 24kHz Vocos quality.
    config['model']['decoder']['fusion_dim'] = 100 
    
    # 1. Load Factorizer (Conditioning)
    factorizer = InformationFactorizerV2(config).to(device).eval()
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device).eval()
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device).eval()
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device).eval()
    
    # Helper for loading
    ckpt_dir = os.path.dirname(args.checkpoint)
    def load_helper(name, obj):
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location=device)
            new_d = {k.replace("_orig_mod.", ""): v for k,v in d.items()}
            obj.load_state_dict(new_d)
            print(f"Loaded {name}")
            
    load_helper("factorizer", factorizer)
    load_helper("sem_rfsq", sem_vq)
    load_helper("pro_rfsq", pro_vq)
    load_helper("spk_pq", spk_pq)
    
    # 2. Flow Model
    # We only need the Alignment Fuser and the Flow Model.
    
    print("Initializing Flow Matching DiT...")
    model = ConditionalFlowMatching(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    fuser = ConditionFuser(
        config['model']['semantic']['output_dim'],
        config['model']['prosody']['output_dim'],
        256, 
        512 # Match model hidden_dim
    ).to(device)
    optimizer.add_param_group({'params': fuser.parameters()})
    
    dataset = CachedDataset(args.data_dir)
    print(f"DEBUG: Using DATA_DIR: {args.data_dir}")
    print(f"DEBUG: Found {len(dataset.files)} files")
    if len(dataset.files) > 0:
        print(f"DEBUG: Sample files: {dataset.files[:3]}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=12, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_fn
    )
    
    print("Starting Flow Matching Training (Cached)...")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            if batch is None: continue
            
            mel = batch['mel'].to(device).float()
            sem = batch['sem'].to(device).float()
            pro = batch['pro'].to(device).float()
            spk = batch['spk'].to(device).float()
            
            if mel.dim() == 4: mel = mel.squeeze(1)
            if sem.dim() == 4: sem = sem.squeeze(1)
            if pro.dim() == 4: pro = pro.squeeze(1)
            if spk.dim() == 3: spk = spk.squeeze(1)
            
            # Mel is already (B, T, 100) from precompute
            x_1 = mel
            target_len = x_1.shape[1]
            
            # Normalize Mel (Approximate standardization based on -11 to 2 range)
            MEL_MEAN, MEL_STD = -2.9, 4.3
            x_1 = (x_1 - MEL_MEAN) / MEL_STD
            
            # Fuse condition (Upsample tokens to Mel length)
            cond = fuser(sem, pro, spk, target_len) # (B, T, D)
            
            # CFG: Condition Dropout
            if random.random() < 0.1:
                cond = torch.zeros_like(cond)
            
            # Flow Matching Loss
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0], device=device)
            t_expand = t.view(-1, 1, 1)
            
            x_t = (1 - t_expand) * x_0 + t_expand * x_1
            v_target = x_1 - x_0
            v_pred = model(x_t, t, cond)
            
            # Correct Masking
            mask = batch['mask'].to(device).float().unsqueeze(-1) # (B, T, 1)
            
            # Weighted Loss (Ignore padding)
            loss = F.l1_loss(v_pred * mask, v_target * mask, reduction='sum') / (mask.sum() * v_pred.shape[-1] + 1e-6)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if global_step % 10 == 0:
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                writer.add_scalar("loss/flow", loss.item(), global_step)
            
            if global_step % 200 == 0:
                model.eval()
                with torch.no_grad():
                    # Generate one sample
                    x_t_v = torch.randn(1, target_len, 100, device=device)
                    cond_v = cond[0:1] # (1, T, D)
                    
                    # Ensure cond_v matches mel length if masking cut it? 
                    # Actually mask handles training, here we gen for full length of batch[0] (which is padded? No, we take index 0)
                    # Ideally we should unpad for viz, but taking [0] is fine for now if it was the longest.
                    
                    steps_v = 50
                    dt_v = 1.0 / steps_v
                    for i_v in range(steps_v):
                        t_sv = torch.tensor([i_v / steps_v], device=device).view(1)
                        v_pv = model(x_t_v, t_sv, cond_v)
                        x_t_v = x_t_v + v_pv * dt_v
                    
                    # Un-standardize for plot
                    MEL_MEAN, MEL_STD = -2.9, 4.3
                    spec_gt = mel[0] # GT is already Mel range
                    spec_pred = x_t_v[0] * MEL_STD + MEL_MEAN
                    
                    save_spectrogram(spec_gt, os.path.join(args.output_dir, f"step_{global_step}_gt.png"), "Ground Truth Mel")
                    save_spectrogram(spec_pred, os.path.join(args.output_dir, f"step_{global_step}_pred.png"), "Flow Predicted Mel")
                    
                    # Visualize Condition
                    cond_viz = cond[0].transpose(0, 1) # (D, T)
                    save_spectrogram(cond_viz, os.path.join(args.output_dir, f"step_{global_step}_cond.png"), "Conditioning Features")
                    
                    print(f"\nSaved visualization to {args.output_dir}/step_{global_step}_*.png")
                model.train()
                
            global_step += 1
            
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"flow_epoch{epoch}.pt"))
        torch.save(fuser.state_dict(), os.path.join(args.output_dir, f"fuser_epoch{epoch}.pt"))

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    # Filter very long files to avoid OOM
    batch = [b for b in batch if b['mel'].shape[-2] < 2000]
    if not batch: return None

    b_mask = []
    b_mel = []
    b_sem = []
    b_pro = []
    b_spk = []
    
    for b in batch:
        mel = b['mel'] # (1, T_mel, 100)
        sem = b['sem'] # (1, T_sem, D)
        pro = b['pro'] # (1, T_pro, D)
        spk = b['spk'] # (1, D) or (D)
        
        # Ensure spk is 1D or 2D
        if spk.dim() == 2: spk = spk.squeeze(0)
        
        T_target = mel.shape[-2] # T is second to last dim
        
        # 1. Align Time (Upsample tokens to match Mel length)
        # Interpolate expects (B, C, T) -> need to transpose
        # sem: (1, T_sem, D) -> (1, D, T_sem)
        
        def align_seq(x, t_target):
            # x: (1, T, D)
            if x.shape[1] != t_target:
                x = x.transpose(1, 2) # (1, D, T)
                x = F.interpolate(x, size=t_target, mode='linear', align_corners=False)
                x = x.transpose(1, 2) # (1, T, D)
            return x
            
        sem = align_seq(sem, T_target)
        pro = align_seq(pro, T_target)
            
        b_mel.append(mel)
        b_sem.append(sem)
        b_pro.append(pro)
        b_spk.append(spk)
        
    # 2. Pad to max length
    # Dim 1 is Time
    max_len = max([m.shape[1] for m in b_mel])
    
    def pad_t(x, target_t, pad_val=0):
        # x: (1, T, D)
        curr = x.shape[1]
        if curr < target_t:
            # Pad dim 1. (left, right, top, bottom) for pad
            # F.pad for 3D input (1, T, D): last dim is D, then T.
            # (pad_last_dim_left, pad_last_dim_right, pad_second_last_left, pad_second_last_right)
            # We want to pad second last (T).
            return F.pad(x, (0, 0, 0, target_t - curr), value=pad_val)
        return x

    b_mel_padded = [pad_t(x, max_len, pad_val=-11.5) for x in b_mel]
    b_sem_padded = [pad_t(x, max_len, pad_val=0) for x in b_sem]
    b_pro_padded = [pad_t(x, max_len, pad_val=0) for x in b_pro]
    
    # Create mask
    for m in b_mel:
        l = m.shape[1]
        pad_len = max_len - l
        mask = torch.cat([torch.ones(l), torch.zeros(pad_len)])
        b_mask.append(mask)

    return {
        'mel': torch.stack(b_mel_padded), # (B, 1, T, 100) -> will likely be squeezed in loop
        'sem': torch.stack(b_sem_padded), # (B, 1, T, D)
        'pro': torch.stack(b_pro_padded),
        'spk': torch.stack(b_spk),
        'mask': torch.stack(b_mask)
    }

if __name__ == "__main__":
    main()
