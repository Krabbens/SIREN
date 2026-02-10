
import os
import sys
import torch
import soundfile as sf
import librosa
import numpy as np
import argparse
from speechmos import dnsmos

def measure_mos(audio_path):
    print(f"Measuring MOS for {audio_path}...")
    # Load and resample if necessary, but dnsmos.run supports paths.
    # However, to be safe about SR, let's load and resample explicitly if needed.
    # Actually, dnsmos.run raises error if SR != 16000.
    
    # Let's verify SR first
    info = sf.info(audio_path)
    if info.samplerate != 16000:
        print(f"  Resampling from {info.samplerate} to 16000 Hz...")
        y, sr = librosa.load(audio_path, sr=16000)
        # Save to temp file
        temp_path = audio_path.replace(".wav", "").replace(".m4a", "") + "_16k_temp.wav"
        sf.write(temp_path, y, 16000)
        audio_path = temp_path
        
    results = dnsmos.run(audio_path, sr=16000, verbose=False)
    # results is a DataFrame if return_df=True (default)
    
    mos = results['ovrl_mos'].mean()
    p808 = results['p808_mos'].mean()
    sig = results['sig_mos'].mean()
    bak = results['bak_mos'].mean()
    
    # Clean up temp
    if "_16k_temp.wav" in audio_path:
        os.remove(audio_path)
        
    return mos, p808, sig, bak

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Ground Truth audio")
    parser.add_argument("--pred", required=True, help="Prediction audio")
    args = parser.parse_args()
    
    print("=== DNSMOS Evaluation ===")
    
    gt_mos, gt_p808, gt_sig, gt_bak = measure_mos(args.gt)
    print(f"[GT]   OVRL: {gt_mos:.3f} | SIG: {gt_sig:.3f} | BAK: {gt_bak:.3f} | P808: {gt_p808:.3f}")
    
    pred_mos, pred_p808, pred_sig, pred_bak = measure_mos(args.pred)
    print(f"[PRED] OVRL: {pred_mos:.3f} | SIG: {pred_sig:.3f} | BAK: {pred_bak:.3f} | P808: {pred_p808:.3f}")
    
    delta_mos = pred_mos - gt_mos
    print("-" * 30)
    print(f"DELTA MOS (OVRL): {delta_mos:+.3f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
