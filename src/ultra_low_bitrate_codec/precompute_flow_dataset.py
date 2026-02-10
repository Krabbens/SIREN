import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
import glob
import random
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.getcwd())

from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from transformers import Wav2Vec2FeatureExtractor, HubertModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class AudioDataset(Dataset):
    def __init__(self, data_dir, segment_length=16000*3): # 3 seconds chunks
        self.files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
        # Sort for determinism
        self.files = sorted([f for f in self.files if os.path.getsize(f) > 10000])
        self.segment_length = segment_length
        print(f"Found {len(self.files)} files")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        try:
            fname = self.files[idx]
            import soundfile as sf
            wav, sr = sf.read(fname)
            wav = torch.tensor(wav, dtype=torch.float32)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0) # (1, T)
            else:
                wav = wav.t() # (C, T)
                wav = wav.mean(0, keepdim=True) # Mix to mono
            
            if sr != 16000:
                # Use torchaudio resample if possible, but if that fails, maybe simple interpolation?
                # Ideally config guarantees 16k input.
                # But let's assume torchaudio.functional works (it should be pure python/cpp)
                wav = torchaudio.functional.resample(wav, sr, 16000)

            
            # Pad if too short
            if wav.shape[1] < self.segment_length:
                pad = self.segment_length - wav.shape[1]
                wav = F.pad(wav, (0, pad))
            
            # Return full file (or up to max length to avoid huge files)
            # Actually flow matching works best with fixed length segments usually, 
            # but for precomputing we can save 3s chunks or full files.
            # Let's simple slice into non-overlapping chunks 
            
            # For simplicity in this script, we just return one random chunk per file to avoid dataset bloat
            # OR we can slide. Let's do random chunk.
            if wav.shape[1] > self.segment_length:
                start = random.randint(0, wav.shape[1] - self.segment_length)
                wav = wav[:, start:start+self.segment_length]
                
            return wav.squeeze(0), os.path.basename(fname)
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(self.segment_length), "error.wav"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Full BitNet checkpoint (decoder)")
    parser.add_argument("--data_dir", default="data/audio")
    parser.add_argument("--output_dir", default="data/flow_dataset")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    parser.add_argument("--num_samples", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = load_config(args.config)
    
    # STFT Transform
    # 1024 FFT -> 513 bins
    n_fft = 1024
    hop_length = 320
    win_length = 1024
    
    # Load Models
    print("Loading Models...")
    model = SpeechDecoderV2(config).to(device)
    factorizer = InformationFactorizerV2(config).to(device)
    
    fsq_levels = config['model']['fsq_levels']
    rfsq_num_levels = config['model']['rfsq_num_levels']
    sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device)
    
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    # Load Weights
    ckpt_dir = os.path.dirname(args.checkpoint)
    
    def load_component(name, obj):
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location=device)
            new_d = {}
            for k,v in d.items(): 
                new_d[k.replace("_orig_mod.", "")] = v
            # Special case for decoder loaded as main
            try:
                if name == "decoder":
                    model_state = obj.state_dict()
                    filtered_d = {}
                    ignored_keys = []
                    for k, v in new_d.items():
                        if k in model_state:
                            if v.shape == model_state[k].shape:
                                filtered_d[k] = v
                            else:
                                ignored_keys.append(k)
                    
                    obj.load_state_dict(filtered_d, strict=False)
                    if ignored_keys:
                        print(f"Ignored {len(ignored_keys)} keys due to shape mismatch (e.g. {ignored_keys[0]})")
                else:
                    obj.load_state_dict(new_d, strict=False)
                print(f"Loaded {name} (strict=False)")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
    
    load_component("decoder", model)
    load_component("factorizer", factorizer)
    load_component("sem_rfsq", sem_vq)
    load_component("pro_rfsq", pro_vq)
    load_component("spk_pq", spk_pq)
    
    model.eval()
    factorizer.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_pq.eval()
    
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    print("Starting Precomputation...")
    count = 0
    
    with torch.no_grad():
        for i, (wav, fname) in enumerate(tqdm(dataloader)):
            if count >= args.num_samples: break
            if fname[0] == "error.wav": continue
            
            wav = wav.to(device)
            
            # 1. Compute Log-Mel Spectrogram (Target)
            # (B, 1, T) -> (B, n_mels, T_mel)
            
            # Define Mel Transform (move outside loop for efficiency if possible, but safe here)
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=80,
                center=True,
                power=2.0
            ).to(device)
            
            mel = mel_transform(wav) # (1, 80, T)
            
            # Log Transform (prevent -inf)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            
            target = mel # (1, 80, T)
            
            # 2. Compute Conditioning Features
            # Extract HuBERT
            inputs = hubert_processor(wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
            features = hubert_out.hidden_states[config['model'].get('hubert_layer', 9)]
            
            # Quantize
            sem, pro, spk = factorizer(features)
            sem_z, _, _ = sem_vq(sem)
            pro_z, _, _ = pro_vq(pro)
            spk_z, _, _ = spk_pq(spk)
            
            # Extract Features via Decoder Reconstructor
            cond = model.reconstructor(sem_z, pro_z, spk_z)
            
            # 3. Align Shapes
            # Mel: (1, 80, T_mel)
            # Cond: (1, T_cond, 512)
            
            cond = cond.transpose(1, 2) # (1, 512, T_cond)
            
            # Match cond to target length via interpolation
            if cond.shape[2] != target.shape[2]:
                cond = F.interpolate(cond, size=target.shape[2], mode='linear')
            
            # Save
            save_path = os.path.join(args.output_dir, f"{count:05d}.pt")
            torch.save({
                'mel': target.cpu().squeeze(0), # (80, T)
                'cond': cond.cpu().squeeze(0), # (512, T)
                'sem': sem_z.cpu().squeeze(0), # Save quantized tokens for potential debugging/fuser use
                'pro': pro_z.cpu().squeeze(0),
                'spk': spk_z.cpu().squeeze(0)
            }, save_path)
            
            count += 1

if __name__ == "__main__":
    main()
