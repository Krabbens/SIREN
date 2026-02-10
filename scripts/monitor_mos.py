
#!/usr/bin/env python3
"""
Monitor MOS evolution for BitVocoder training.
Watches checkpoints/vocoder_80band_stft/ and evaluates MOS for new checkpoints.
Outputs:
- mos_history.csv
- mos_history.png
"""

import os
import time
import torch
import torchaudio
import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
from speechmos import dnsmos
import soundfile as sf
import librosa
import numpy as np
import yaml
import sys

# Append path to import project modules
sys.path.append(os.getcwd())
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder

def measure_mos(audio_path):
    print(f"  Measuring MOS for {audio_path}...")
    try:
        results = dnsmos.run(audio_path, sr=16000, verbose=False)
        return results['ovrl_mos'].mean()
    except Exception as e:
        print(f"  MOS Error: {e}")
        return 0.0

def load_vocoder(ckpt_path, device):
    # Match config from training (80 bands, 256 dim)
    vocoder = BitVocoder(
        input_dim=80, 
        dim=256, 
        n_fft=1024, 
        hop_length=320, 
        num_layers=4, 
        num_res_blocks=1
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['vocoder'] if 'vocoder' in ckpt else ckpt
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Partial load for robustness
    current = vocoder.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in current and v.shape == current[k].shape}
    vocoder.load_state_dict(filtered, strict=False)
    vocoder.eval()
    return vocoder

def main():
    CHECKPOINT_DIR = "checkpoints/vocoder_80band_stft"
    TEST_AUDIO = "data/jakubie.wav"
    HISTORY_FILE = os.path.join(CHECKPOINT_DIR, "mos_history.csv")
    PLOT_FILE = os.path.join(CHECKPOINT_DIR, "mos_history.png")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Helper to reconstruct Mel using Librosa (avoid torchaudio dependency)
    def compute_mel(wav_torch):
        # wav_torch: (B, T)
        wav_np = wav_torch.squeeze().cpu().numpy()
        mel = librosa.feature.melspectrogram(
            y=wav_np, sr=16000, n_fft=1024, hop_length=320, n_mels=80, fmin=0, fmax=8000, center=True, power=2.0
        )
        mel_torch = torch.from_numpy(mel).float().to(device)
        # Add batch dim if missing
        if mel_torch.ndim == 2: mel_torch = mel_torch.unsqueeze(0)
        return mel_torch

    # Load or init history
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
    else:
        history = pd.DataFrame(columns=['epoch', 'mos', 'ckpt_path'])

    processed_ckpts = set(history['ckpt_path'].values)
    
    print(f"Starting MOS Monitor. Watching {CHECKPOINT_DIR}...")
    
    while True:
        # Find all checkpoints
        ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "vocoder_epoch*.pt"))
        # Also check last (but handle it carefully as it overwrites)
        
        # Sort by epoch
        def get_epoch(p):
            m = re.search(r"epoch(\d+)", p)
            return int(m.group(1)) if m else -1
            
        ckpts = sorted(ckpts, key=get_epoch)
        
        updated = False
        
        for ckpt in ckpts:
            if ckpt in processed_ckpts:
                continue
                
            epoch = get_epoch(ckpt)
            print(f"Processing Epoch {epoch} ({ckpt})...")
            
            # Load Model
            try:
                vocoder = load_vocoder(ckpt, device)
            except Exception as e:
                print(f"  Load Error: {e}")
                continue
            
            # Run Inference
            # 1. Load Audio using SoundFile to avoid torchcodec issues
            wav_np, sr = sf.read(TEST_AUDIO)
            wav = torch.from_numpy(wav_np).float()
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.to(device)
            if wav.ndim > 1: wav = wav.mean(-1, keepdim=True).transpose(0, 1) # Support (T, C) -> (C, T) or just (T)
            if wav.ndim == 1: wav = wav.unsqueeze(0) # (1, T)
            
            # 2. Get Mel
            with torch.no_grad():
                mel = compute_mel(wav)
                mel = torch.log(torch.clamp(mel, min=1e-5))
                # Normalize
                mel = (mel - (-5.0)) / 3.5
                mel_in = mel.transpose(1, 2) # (B, T, C)
                
                # 3. Vocode
                audio_hat = vocoder(mel_in)
                
            # Save Temp
            out_path = os.path.join(CHECKPOINT_DIR, f"temp_mos_epoch{epoch}.wav")
            sf.write(out_path, audio_hat.squeeze().cpu().numpy(), 16000)
            
            # Measure MOS
            mos = measure_mos(out_path)
            print(f"  -> Epoch {epoch}: MOS = {mos:.3f}")
            
            # Update History
            new_row = pd.DataFrame({'epoch': [epoch], 'mos': [mos], 'ckpt_path': [ckpt]})
            history = pd.concat([history, new_row], ignore_index=True)
            history.to_csv(HISTORY_FILE, index=False)
            processed_ckpts.add(ckpt)
            updated = True
            
            # Clean temp
            # os.remove(out_path) # Keep for debug if needed
        
        if updated:
            # Plot
            plt.figure(figsize=(10, 6))
            hist_sorted = history.sort_values('epoch')
            plt.plot(hist_sorted['epoch'], hist_sorted['mos'], marker='o', linestyle='-', color='b')
            plt.title("BitVocoder MOS Evolution (Jakubie Test)")
            plt.xlabel("Epoch")
            plt.ylabel("DNSMOS (OVRL)")
            plt.grid(True)
            plt.savefig(PLOT_FILE)
            plt.close()
            print(f"Updated plot: {PLOT_FILE}")
            
            # Copy to artifacts for user visibility
            os.system(f"cp {PLOT_FILE} /home/sperm/.gemini/antigravity/brain/0eb58771-53f4-4dc3-ad59-1d3cb3abd595/mos_history.png")

        time.sleep(60) # Watch every minute

if __name__ == "__main__":
    main()
