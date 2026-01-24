
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class AudioDataset(Dataset):
    def __init__(self, data_dir, segment_length=16000*2): # 2 seconds
        self.files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
        self.files = [f for f in self.files if os.path.getsize(f) > 50000]
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
                
            return wav.squeeze(0)
        except:
            return torch.zeros(self.segment_length)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Full BitNet checkpoint")
    parser.add_argument("--data_dir", default="data/audio")
    parser.add_argument("--output_dir", default="checkpoints_vocoder_finetune")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    
    # Load Model
    config = load_config(args.config)
    
    print("Initializing SpeechDecoderV2 (with New Phase Head)...")
    # This automatically uses the UPDATED NeuralVocoderV2 code
    model = SpeechDecoderV2(config).to(device)
    
    # Load Checkpoint (Partial Loading)
    print(f"Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Filter out mismatched keys (Phase Head)
    model_state = model.state_dict()
    new_state_dict = {}
    ignored_keys = []
    
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                new_state_dict[k] = v
            else:
                print(f"Ignoring {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
                ignored_keys.append(k)
        else:
            # Handle prefix matching if needed
            # e.g. "module." prefix
            suffix = k.replace("_orig_mod.", "")
            if suffix in model_state and state_dict[k].shape == model_state[suffix].shape:
                new_state_dict[suffix] = state_dict[k]
                
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded {len(new_state_dict)} keys. Ignored: {len(ignored_keys)}")
    
    # We only want to train the Vocoder (or just the Phase Head?)
    # Let's train the whole Vocoder High-LR, keeping Feature Reconstructor frozen.
    
    print("Freezing Reconstructor...")
    for p in model.reconstructor.parameters():
        p.requires_grad = False
        
    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Loss
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    # Dataset
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # Safety
    
    # Training Loop
    global_step = 0
    
    # Pre-train Loop (Auto-Encoder style)
    # We iterate audio -> features (via Encoder? NO, we only have Decoder here)
    # Wait. Decoder needs Sem/Pro/Spk codes.
    # We can't just feed Audio into Decoder. 
    # We need the ENCODER to produce codes from Audio.
    
    # Need to load the rest of the pipeline!
    print("Loading Encoder/Quantizers...")
    from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
    from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    factorizer = InformationFactorizerV2(config).to(device)
    # Load factorizer weights
    # Assuming checkpoint dir structure or single file
    ckpt_dir = os.path.dirname(args.checkpoint)
    
    def load_component(name, obj):
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location=device)
            # handle keys
            new_d = {}
            for k,v in d.items(): 
                new_d[k.replace("_orig_mod.", "")] = v
            obj.load_state_dict(new_d)
            print(f"Loaded {name}")
        else:
            print(f"Warning: {name} not found, random init")

    load_component("factorizer", factorizer)
    
    fsq_levels = config['model']['fsq_levels']
    rfsq_num_levels = config['model']['rfsq_num_levels']
    sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device)
    
    load_component("sem_rfsq", sem_vq)
    load_component("pro_rfsq", pro_vq)
    load_component("spk_pq", spk_pq)
    
    # Freeze Encoder side
    factorizer.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_pq.eval()
    
    model.train() # Decoder/Vocoder in train mode (Reconstructor frozen manually above)
    model.reconstructor.eval() 
    
    print("Starting Fine-tuning...")
    
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for wav in pbar:
            wav = wav.to(device)
            # (B, T)
            
            with torch.no_grad():
                # Extract HuBERT
                inputs = hubert_processor(wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
                features = hubert_out.hidden_states[config['model'].get('hubert_layer', 9)]
                
                # Quantize
                sem, pro, spk = factorizer(features)
                sem_z, _, _ = sem_vq(sem)
                pro_z, _, _ = pro_vq(pro)
                spk_z, _, _ = spk_pq(spk)
            
            # Decoder Forward
            # Note: Decoder output length might differ slightly from input due to upsampling
            pred_wav = model(sem_z, pro_z, spk_z)
            
            # Match lengths
            min_len = min(wav.shape[1], pred_wav.shape[1])
            wav_crop = wav[:, :min_len]
            pred_crop = pred_wav[:, :min_len]
            
            # Loss types
            # 1. Multi-Res STFT (Mag + Phase proxy)
            sc_loss, mag_loss = stft_loss(pred_crop.squeeze(1), wav_crop)
            loss = sc_loss + mag_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            global_step += 1
            
            if global_step % 100 == 0:
                writer.add_scalar("loss/total", loss.item(), global_step)
        
        # Save per epoch
        path = os.path.join(args.output_dir, f"decoder_finetuned_epoch{epoch}.pt")
        torch.save(model.state_dict(), path)
        print(f"Saved {path}")

if __name__ == "__main__":
    main()
