import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
import glob
import os
import argparse
import yaml
import soundfile as sf
import random
import sys
import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.post_net import AudioEnhancer

from transformers import Wav2Vec2FeatureExtractor, HubertModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Dataset ---
class AudioDataset(Dataset):
    def __init__(self, manifest_files, sample_rate=16000, max_duration=2.0):
        self.files = []
        for mf in manifest_files:
            if not os.path.exists(mf): continue
            import json
            with open(mf, 'r') as f:
                 data = json.load(f)
                 # Expecting list of dicts with 'audio_path' or similar
                 # Checking format... usually it's {"audio_filepath": ...} or similar
                 # If it's the custom format: {"audio": "path", ...}
                 for item in data:
                     if 'audio_filepath' in item:
                         self.files.append(item['audio_filepath'])
                     elif 'audio_path' in item:
                         self.files.append(item['audio_path'])
                     elif 'audio' in item:
                         self.files.append(item['audio'])
        
        # If no manifest, try globbing typical dirs
        if not self.files:
            print("No manifest found or empty. Searching recursively in typical data dirs...")
            # Fallback to local 'data' dir or specific paths if known
            pass
            
        self.sample_rate = sample_rate
        self.segment_length = int(max_duration * sample_rate)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            # Load basic info first without loading whole file if possible? 
            # soundfile can seek.
            info = sf.info(path)
            length = info.frames
            sr = info.samplerate
            
            if length < self.segment_length:
                 # Pad
                 wav, _ = sf.read(path)
                 wav = torch.from_numpy(wav).float()
                 if len(wav.shape) > 1: wav = wav[:, 0] # mono
                 if sr != self.sample_rate:
                     wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                 
                 pad_len = self.segment_length - wav.shape[0]
                 wav = F.pad(wav, (0, pad_len))
                 return wav
            else:
                # Random crop
                start = random.randint(0, length - self.segment_length)
                wav, _ = sf.read(path, start=start, frames=self.segment_length)
                wav = torch.from_numpy(wav).float()
                if len(wav.shape) > 1: wav = wav[:, 0] # mono
                if sr != self.sample_rate:
                     wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                     # Resampling changes length
                     if wav.shape[0] > self.segment_length:
                         wav = wav[:self.segment_length]
                     elif wav.shape[0] < self.segment_length:
                         wav = F.pad(wav, (0, self.segment_length - wav.shape[0]))
                return wav
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(self.segment_length)

# --- Losses ---
class MelSpecLoss(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.l1 = nn.L1Loss()

    def forward(self, y_hat, y):
        mel_hat = self.mel_transform(y_hat)
        mel_y = self.mel_transform(y)
        # Log mel loss
        loss = self.l1(torch.log(mel_hat + 1e-5), torch.log(mel_y + 1e-5))
        return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/multispeaker_optimized.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_multispeaker")
    parser.add_argument("--step", type=int, default=57000)
    parser.add_argument("--output_dir", type=str, default="checkpoints_postnet")
    parser.add_argument("--data_json", type=str, nargs='+', default=["data/train_clean_100.json", "data/train_clean_360.json"], help="Path to data manifest jsons")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Pretrained Codec (Frozen)
    print("Loading Codec...")
    config = load_config(args.config)
    
    factorizer = InformationFactorizerV2(config).to(device).eval()
    decoder = SpeechDecoderV2(config).to(device).eval()
    sem_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=config['model']['semantic']['output_dim']
    ).to(device).eval()
    pro_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=config['model']['prosody']['output_dim']
    ).to(device).eval()
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device).eval()
    
    # Load Weights
    def load_clean_state_dict(model, path):
        state_dict = torch.load(path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."): new_state_dict[k[10:]] = v
            else: new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    ckpt_dir = args.checkpoint_dir
    step = args.step
    models_to_load = {
        'factorizer': (factorizer, 'factorizer'),
        'decoder': (decoder, 'decoder'),
        'sem_vq': (sem_vq, 'sem_rfsq'),
        'pro_vq': (pro_vq, 'pro_rfsq'),
        'spk_pq': (spk_pq, 'spk_pq')
    }
    
    for name, (model, filename_base) in models_to_load.items():
        flat_path = os.path.join(ckpt_dir, f"{filename_base}_{step}.pt")
        nested_path = os.path.join(ckpt_dir, f"step_{step}", f"{filename_base}.pt")
        if os.path.exists(flat_path): load_path = flat_path
        elif os.path.exists(nested_path): load_path = nested_path
        else:
             print(f"Skipping {name} (not found)")
             continue
        load_clean_state_dict(model, load_path)
    
    # HuBERT
    print("Loading HuBERT...")
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    # 2. Init PostNet
    print("Initializing PostNet (SnakeBeta)...")
    post_net = AudioEnhancer(use_snake_beta=True).to(device)
    optimizer = torch.optim.AdamW(post_net.parameters(), lr=args.lr, betas=(0.8, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Data
    dataset = AudioDataset(args.data_json)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 4. Loss
    criterion_mel = MelSpecLoss().to(device)
    criterion_l1 = nn.L1Loss().to(device)
    
    # Helper to run codec
    def run_codec(audio_batch):
        with torch.no_grad():
            # Features
            inputs = hubert_processor(audio_batch.squeeze(1).cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            # Need to handle variable lengths if batching? Dataset forces fixed length so ok
            outputs = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
            layer = config['model'].get('hubert_layer', 9)
            features = outputs.hidden_states[layer]
            
            # Forward
            sem, pro, spk = factorizer(features)
            sem_z, _, _ = sem_vq(sem)
            pro_z, _, _ = pro_vq(pro)
            spk_z, _, _ = spk_pq(spk)
            
            audio_hat = decoder(sem_z, pro_z, spk_z)
            
            # Fix length mismatch (HuBERT downsampling vs Decoder upsampling)
            if audio_hat.shape[-1] != audio_batch.shape[-1]:
                audio_hat = F.interpolate(audio_hat.unsqueeze(1), size=audio_batch.shape[-1], mode='linear').squeeze(1)
            
            return audio_hat

    # 5. Train Loop
    print(f"Starting training for {args.epochs} epochs...")
    global_step = 0
    
    for epoch in range(args.epochs):
        post_net.train()
        total_loss = 0
        
        # Wrap dataloader with tqdm
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, audio in enumerate(pbar):
            audio = audio.to(device)
            if audio.dim() == 2: audio = audio.unsqueeze(1) # B, 1, T
            
            # Generate degraded input
            audio_degraded = run_codec(audio).unsqueeze(1) # B, 1, T
            
            # PostNet forward
            audio_enhanced = post_net(audio_degraded)
            
            # Loss
            loss_mel = criterion_mel(audio_enhanced.squeeze(1), audio.squeeze(1))
            loss_time = criterion_l1(audio_enhanced, audio)
            loss = loss_mel + 10 * loss_time # Weight time domain?
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(post_net.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            
            # Update tqdm bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'mel': f"{loss_mel.item():.4f}"})
                
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")
        
        if epoch % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"postnet_epoch_{epoch}.pt")
            torch.save(post_net.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
