
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import argparse
import torchaudio
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultra_low_bitrate_codec.models.micro_transformer import MicroTransformer
from ultra_low_bitrate_codec.training.optimizers import Lion

class DistillationDataset(Dataset):
    def __init__(self, audio_dir, features_dir, max_len=16000*3): # 3s max for VRAM safety
        self.audio_dir = audio_dir
        self.features_dir = features_dir
        self.max_len = max_len
        
        # Build map for fast lookup
        print("Building audio map...")
        self.audio_map = {}
        for root, dirs, files in os.walk(self.audio_dir):
            for f in files:
                if f.endswith(('.wav', '.flac', '.m4a', '.mp3')):
                    base = os.path.splitext(f)[0]
                    self.audio_map[base] = os.path.join(root, f)
        
        # Find matches
        self.files = []
        feat_files = glob.glob(f"{features_dir}/*.pt")
        for f in feat_files:
            name = os.path.basename(f).replace('.pt', '')
            if name in self.audio_map:
                self.files.append(name)
            
        print(f"Found {len(self.files)} paired samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        
        # Load Features
        feat_path = os.path.join(self.features_dir, f"{name}.pt")
        try:
            feat_data = torch.load(feat_path, map_location='cpu')
            if isinstance(feat_data, dict) and 'features' in feat_data: 
                features = feat_data['features']
            else:
                features = feat_data
            
            # Load Audio
            audio_path = self.audio_map[name]
            wav, sr = sf.read(audio_path)
            wav = torch.tensor(wav, dtype=torch.float32)
            if wav.dim() > 1: wav = wav.mean(dim=0)
            if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
            
            # Normalization (Zero-mean, Unit-variance for HuBERT compatibility)
            wav = (wav - wav.mean()) / (wav.std() + 1e-6)
            
            # Align (320 samples per frame)
            target_samples = features.shape[0] * 320
            if wav.shape[0] > target_samples:
                wav = wav[:target_samples]
            elif wav.shape[0] < target_samples:
                wav = torch.nn.functional.pad(wav, (0, target_samples - wav.shape[0]))
                
            # Random Crop if too long
            if wav.shape[0] > self.max_len:
                max_frames = self.max_len // 320
                start_frame = torch.randint(0, features.shape[0] - max_frames, (1,)).item()
                features = features[start_frame:start_frame+max_frames]
                wav = wav[start_frame*320 : (start_frame+max_frames)*320]
                
            return wav, features
            
        except Exception as e:
            # print(f"Error loading {name}: {e}")
            return torch.zeros(16000), torch.zeros(50, 768)

def collate(batch):
    batch = [item for item in batch if item[0].shape[0] > 0]
    if not batch: return None
    
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    wavs, feats = zip(*batch)
    
    max_len = wavs[0].shape[0]
    wav_batch = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        wav_batch[i, :w.shape[0]] = w
        
    max_frames = feats[0].shape[0]
    feat_batch = torch.zeros(len(feats), max_frames, 768)
    mask = torch.zeros(len(feats), max_frames).bool()
    
    for i, f in enumerate(feats):
        feat_batch[i, :f.shape[0]] = f
        mask[i, :f.shape[0]] = True
        
    return wav_batch, feat_batch, mask

def main():
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="data/audio")
    parser.add_argument("--features_dir", default="data/features_distilhubert")
    parser.add_argument("--output_dir", default="checkpoints/microtransformer_distill")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--use_rvq", action="store_true", help="Enable RVQ bottleneck")
    parser.add_argument("--rvq_quantizers", type=int, default=8, help="Number of RVQ quantizers")
    parser.add_argument("--rvq_dropout", type=float, default=0.0, help="Dropout probability for RVQ codebooks")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate") # Restore LR just in case
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize Student (BitNet - Upgraded Capacity)
    model = MicroTransformer(
        hidden_dim=384, 
        num_layers=8,
        use_rvq=args.use_rvq,
        rvq_num_quantizers=args.rvq_quantizers,
        rvq_dropout_p=args.rvq_dropout
    ).to(device)
    print(f"MicroTransformer Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    if args.resume and os.path.exists(args.resume):
        print(f"Loading {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}, strict=False)
    
    # Compile
    # model = torch.compile(model) # Optional, can cause issues with BitNet triton kernels if not careful
    
    dataset = DistillationDataset(args.audio_dir, args.features_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate
    )
    
    # Use Lion for stronger optimization
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Switch to ReduceLROnPlateau for adaptive LR based on loss speed
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Visualization setup
    viz_dir = os.path.join(args.output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    import imageio
    
    # Setup Pipeline for Algo-Viz
    viz_pipeline = setup_pipeline_models(device)
    
    # Load fixed sample for visualization consistency
    viz_sample = torch.zeros(16000*3) # default
    if os.path.exists("data/jakubie.wav"):
        w, sr = sf.read("data/jakubie.wav")
        if w.ndim > 1: w = w.mean(axis=1) # mix to mono
        # resample if needed
        import torchaudio.functional as AF
        if sr != 16000:
             w_t = torch.tensor(w).float()
             w_t = AF.resample(w_t, sr, 16000)
             w = w_t.numpy()
        viz_sample = torch.tensor(w).float()
        print("Loaded viz sample: data/jakubie.wav")
    else:
        print("Warning: data/jakubie.wav not found, using silence for viz.")

    global_step = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        # Try to infer epoch from filename
        try:
            start_epoch = int(args.resume.split("_ep")[-1].replace(".pt", ""))
            print(f"Resuming from Epoch {start_epoch}")
        except:
            pass

    print("\nStarting Distillation Training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            if batch is None: continue
            
            global_step += 1
            
            wav, target, mask = batch
            wav, target, mask = wav.to(device), target.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if args.use_rvq:
                    pred, rvq_loss, _ = model(wav)
                else:
                    pred = model(wav)
                    rvq_loss = 0.0
                
                # Align lengths
                min_len = min(pred.shape[1], target.shape[1])
                pred = pred[:, :min_len]
                target = target[:, :min_len]
                mask = mask[:, :min_len]
                
                # Losses
                pred_f = pred[mask]
                target_f = target[mask]
                
                l1_loss = (pred_f - target_f).abs().mean()
                
                cos_sim = F.cosine_similarity(pred_f, target_f, dim=-1).mean()
                cos_loss = 1.0 - cos_sim
                
                std_pred = pred_f.std()
                std_target = target_f.std()
                var_loss = (std_pred - std_target).abs()
                
                d_pred = pred[:, 1:] - pred[:, :-1]
                d_target = target[:, 1:] - target[:, :-1]
                d_mask = mask[:, 1:] & mask[:, :-1]
                d_loss = (d_pred - d_target).abs()[d_mask].mean()
                
                # Total
                loss = l1_loss + cos_loss + (10 * var_loss) + (1 * d_loss) + rvq_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            rvq_val = rvq_loss.item() if isinstance(rvq_loss, torch.Tensor) else rvq_loss
            pbar.set_postfix(l1=l1_loss.item(), cos=cos_loss.item(), total=loss.item(), rvq=rvq_val)
            
            # --- Pipeline Visualization Every 200 Steps ---
            if global_step % 200 == 0:
                 run_viz(model, viz_pipeline, viz_sample, device, global_step, viz_dir)
        
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            path = f"{args.output_dir}/microtransformer_ep{epoch+1}.pt"
            torch.save(model.state_dict(), path)
            print(f"Saved {path}")
            
        # --- Animation (Pipeline) ---
        try:
            all_imgs = sorted(glob.glob(os.path.join(viz_dir, "pipeline_*.png")))
            if len(all_imgs) >= 5:
                # Use frames from last 5
                last_5 = all_imgs[-5:]
                gif_path = os.path.join(viz_dir, f"progress_pipeline_ep{epoch+1}.gif")
                
                images = []
                for filename in last_5:
                    images.append(imageio.imread(filename))
                imageio.mimsave(gif_path, images, duration=0.5) 
                print(f"Saved animation: {gif_path}")
        except Exception as e:
            print(f"Animation failed: {e}")
            
# Auxiliary Models for Pipeline Viz
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
import yaml

def setup_pipeline_models(device):
    """Load frozen auxiliary models for visualization"""
    print("Loading visualization pipeline...")
    
    # Config
    try:
        with open("src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml") as f: 
            config = yaml.safe_load(f)
    except:
        # Fallback if config not found
        config = {'model': {'decoder': {'fusion_dim': 80, 'hidden_dim': 512, 'fusion_heads': 8, 'dropout': 0.1}, 
                            'fsq_levels': [8,5,5,5], 'rfsq_num_levels': 1}}

    # 1. Factorizer
    factorizer = InformationFactorizerV2(config).to(device)
    try:
        fq_path = "checkpoints/factorizer_microhubert_finetune_v4/factorizer.pt"
        if not os.path.exists(fq_path): fq_path = "checkpoints/factorizer_microhubert_finetune_v4/factorizer_best_step.pt"
        ckpt_f = torch.load(fq_path, map_location=device)
        if isinstance(ckpt_f, dict) and 'model_state_dict' in ckpt_f: ckpt_f = ckpt_f['model_state_dict']
        factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_f.items()}, strict=False)
        factorizer.eval()
    except Exception as e:
        print(f"Warning: Failed to load Factorizer for viz: {e}")
        return None

    # 2. Fuser
    try:
        fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
        fu_path = "checkpoints/checkpoints_flow_v2/fuser_epoch20.pt"
        fu_ckpt = torch.load(fu_path, map_location=device)
        if isinstance(fu_ckpt, dict) and 'model_state_dict' in fu_ckpt: fu_ckpt = fu_ckpt['model_state_dict']
        fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fu_ckpt.items()})
        fuser.eval()
    except Exception as e:
        print(f"Warning: Failed to load Fuser for viz: {e}")
        return None

    # 3. Flow
    try:
        config['model']['decoder']['fusion_dim'] = 80
        config['model']['decoder']['hidden_dim'] = 512
        flow_model = ConditionalFlowMatching(config).to(device)
        fl_path = "checkpoints/checkpoints_flow_v2/flow_epoch20.pt"
        fl_ckpt = torch.load(fl_path, map_location=device)
        if isinstance(fl_ckpt, dict) and 'model_state_dict' in fl_ckpt: fl_ckpt = fl_ckpt['model_state_dict']
        flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fl_ckpt.items()})
        flow_model.eval()
    except Exception as e:
        print(f"Warning: Failed to load Flow for viz: {e}")
        return None
        
    # 4. Quantizers (Fixed instance)
    sem_vq = ResidualFSQ(levels=[8,5,5,5], num_levels=1, input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=[8,5,5,5], num_levels=1, input_dim=8).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    return {
        'factorizer': factorizer,
        'fuser': fuser,
        'flow': flow_model,
        'sem_vq': sem_vq,
        'pro_vq': pro_vq,
        'spk_pq': spk_pq
    }

def run_viz(model, pipeline, wav_sample, device, step, save_dir):
    """Run full pipeline visualization on a single audio sample"""
    if pipeline is None: return
    
    with torch.no_grad():
        # Preprocess Audio
        wav = wav_sample.to(device)
        # Norm
        wav_norm = (wav - wav.mean()) / (wav.std() + 1e-6)
        
        # 1. Student -> Features
        # Handle tuple return if RVQ is on
        out = model(wav_norm.unsqueeze(0))
        if isinstance(out, tuple):
            features = out[0]
        else:
            features = out
            
        # 2. Factorizer -> Latents
        sem, pro, spk = pipeline['factorizer'](features)
        sem_z, _, _ = pipeline['sem_vq'](sem)
        pro_z, _, _ = pipeline['pro_vq'](pro)
        spk_z, _, _ = pipeline['spk_pq'](spk)
        
        # 3. Fuser -> Cond
        target_mel_len = wav.shape[0] // 320
        cond = pipeline['fuser'](sem_z, pro_z, spk_z, target_mel_len)
        
        # 4. Flow -> Mel
        # Use fewer steps for speed during training (e.g. 10 instead of 50)
        mel = pipeline['flow'].solve_ode(cond, steps=10, solver='euler', cfg_scale=1.0)
        
        # Plot
        mel_np = mel.squeeze().T.cpu().numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_np, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Pipeline Reconstruction (Step {step})")
        
        path = os.path.join(save_dir, f"pipeline_{step:06d}.png")
        plt.savefig(path)
        plt.close()

if __name__ == "__main__":
    main()
