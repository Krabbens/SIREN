
import os
import torch
import torchaudio.functional as F
import soundfile as sf
from tqdm import tqdm
import argparse
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/audio", help="Source audio directory")
    parser.add_argument("--output_dir", default="data/audio_aug_v2", help="Destination augmentation directory")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Augmentation V2: NO pitch shift (EQ + Noise + Volume only)")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect files
    files = []
    for root, dirs, filenames in os.walk(args.input_dir):
        for f in filenames:
            if f.endswith(('.wav', '.flac', '.mp3')):
                files.append(os.path.join(root, f))
    
    print(f"Found {len(files)} files.")
    
    processed_count = 0
    
    for path in tqdm(files):
        try:
            rel_path = os.path.relpath(path, args.input_dir)
            out_path = os.path.join(args.output_dir, rel_path)
            
            if os.path.exists(out_path):
                continue
                
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            wav, sr = sf.read(path)
            wav = torch.tensor(wav, dtype=torch.float32)
            
            if sr != args.sample_rate:
                wav = F.resample(wav, sr, args.sample_rate)
            
            if wav.ndim > 1:
                wav = wav.mean(dim=-1)
            
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            elif wav.ndim == 2:
                wav = wav.transpose(0, 1)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)

            wav = wav.to(device)
            
            # === Augmentation V2 (NO pitch shift) ===
            
            # 1. EQ (Random lowpass/highpass biquad)
            eq_type = random.choice(['lowpass', 'highpass', 'bandpass'])
            if eq_type == 'lowpass':
                cutoff = random.uniform(2000, 6000)
                wav = F.lowpass_biquad(wav, args.sample_rate, cutoff_freq=cutoff, Q=0.707)
            elif eq_type == 'highpass':
                cutoff = random.uniform(50, 300)
                wav = F.highpass_biquad(wav, args.sample_rate, cutoff_freq=cutoff, Q=0.707)
            else:
                center = random.uniform(300, 3000)
                wav = F.bandpass_biquad(wav, args.sample_rate, central_freq=center, Q=1.0)
            
            # 2. Additive Gaussian Noise (SNR 15-30 dB)
            snr_db = random.uniform(15, 30)
            signal_power = wav.pow(2).mean()
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(wav) * noise_power.sqrt()
            wav = wav + noise
            
            # 3. Volume perturbation
            gain = random.uniform(0.6, 1.0)
            wav = wav * gain
            
            # Normalize to avoid clipping
            peak = wav.abs().max()
            if peak > 0.95:
                wav = wav / (peak + 1e-6) * 0.95
                
            wav_cpu = wav.squeeze(0).cpu().numpy()
            sf.write(out_path, wav_cpu, args.sample_rate)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    print(f"Processed {processed_count} files.")

if __name__ == "__main__":
    main()
