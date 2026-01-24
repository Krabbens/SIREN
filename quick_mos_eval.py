#!/usr/bin/env python3
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio

print(f"Python executable: {sys.executable}")
print(f"NumPy version: {np.__version__}")

try:
    from speechmos import dnsmos
except ImportError as e:
    print(f"Error importing speechmos: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Other error during import: {e}")
    sys.exit(1)

def evaluate(path):
    print(f"Evaluating: {path}")
    # Use soundfile to read
    try:
        audio_np, sr = sf.read(path)
    except Exception as e:
        print(f"Error reading file with soundfile: {e}")
        return

    # Convert to standard format (mono, 16k)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1) # mix to mono
    
    if sr != 16000:
        print(f"Resampling from {sr} to 16000 Hz")
        audio_t = torch.from_numpy(audio_np).float()
        audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
        audio_np = audio_t.numpy()
    
    try:
        result = dnsmos.run(audio_np, 16000)
        # print(f"Raw Result: {result}")
        
        # Handle dict access
        ovrl = result.get('ovrl_mos', result.get('ovrl', 0))
        sig = result.get('sig_mos', result.get('sig', 0))
        bak = result.get('bak_mos', result.get('bak', 0))
        p808 = result.get('p808_mos', 0)
        
        print(f"OVRL: {ovrl:.3f}")
        print(f"SIG:  {sig:.3f}")
        print(f"BAK:  {bak:.3f}")
        print(f"P808: {p808:.3f}")
    except Exception as e:
        print(f"Error during MOS calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_mos_eval.py <wav_file>")
        sys.exit(1)
    
    evaluate(sys.argv[1])
