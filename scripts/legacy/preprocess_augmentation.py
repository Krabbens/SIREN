
import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from tqdm import tqdm
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/audio", help="Source audio directory")
    parser.add_argument("--output_dir", default="data/audio_aug", help="Destination augmentation directory")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect files
    files = []
    for root, dirs, filenames in os.walk(args.input_dir):
        for f in filenames:
            if f.endswith(('.wav', '.flac', '.mp3')):
                files.append(os.path.join(root, f))
    
    print(f"Found {len(files)} files.")
    
    # Transforms (Instantiate on device)
    # PitchShift: fast approximate via Phase Vocoder or Time-Domain?? 
    # Torchaudio PitchShift uses Phase Vocoder.
    # We will randomly select n_steps per file.
    
    processed_count = 0
    
    for path in tqdm(files):
        try:
            rel_path = os.path.relpath(path, args.input_dir)
            out_path = os.path.join(args.output_dir, rel_path)
            
            if os.path.exists(out_path):
                continue
                
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # Load using SoundFile (bypass torchaudio backend issues)
            import soundfile as sf
            wav, sr = sf.read(path)
            wav = torch.tensor(wav, dtype=torch.float32)
            
            # Resample to target first (CPU is often safer for resample usually, but lets try GPU if needed)
            # Actually keep load/resample on CPU to avoid VRAM fragmentation, move to GPU for augment
            if sr != args.sample_rate:
                wav = F.resample(wav, sr, args.sample_rate)
            
            if wav.ndim > 1:
                wav = wav.mean(dim=-1) # Soundfile returns (T, C) usually, or (T,) if mono.
                # If (T, C), mean(dim=-1) is correct.
                # If (T,), ndim=1.
            
            # Ensure (1, T) for torchaudio functional
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            elif wav.ndim == 2:
                # SF returns (Time, Channels). Torch expects (Channels, Time)
                wav = wav.transpose(0, 1)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)

            # Move to GPU
            wav = wav.to(device)
            
            # Augment
            # 1. Random Pitch Shift (-4 to +4 semitones)
            n_steps = random.randint(-4, 4)
            if n_steps != 0:
                # pitch_shift expects (..., time)
                wav = F.pitch_shift(wav, args.sample_rate, n_steps)
            
            # 2. EQ (Three-band equalizer via BiQuad)
            # shelf filters or peaking
            # Random Low Shelf (Bass boost/cut)
            gain_db = random.uniform(-6, 6)
            wav = F.lowpass_biquad(wav, args.sample_rate, cutoff_freq=random.uniform(100, 300), Q=0.707)
            
            # 3. Volume
            gain = random.uniform(0.5, 1.0)
            wav = wav * gain
            
            # Normalize to avoid clipping after EQ
            peak = wav.abs().max()
            if peak > 0.95:
                wav = wav / (peak + 1e-6) * 0.95
                
            # Save using SoundFile
            wav_cpu = wav.squeeze(0).cpu().numpy()
            sf.write(out_path, wav_cpu, args.sample_rate)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    print(f"Processed {processed_count} files.")

if __name__ == "__main__":
    main()
