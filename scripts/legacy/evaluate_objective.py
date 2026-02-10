#!/usr/bin/env python3
"""
Objective Evaluation for SIREN v2
Computes PESQ, STOI, and DNSMOS.
"""

import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from speechmos import dnsmos

def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    if sr != target_sr:
        audio_t = torch.from_numpy(audio).float()
        audio_t = torchaudio.functional.resample(audio_t, sr, target_sr)
        audio = audio_t.numpy()
        
    return audio

def evaluate(ref_path, deg_path):
    print(f"Reference: {ref_path}")
    print(f"Degraded:  {deg_path}")
    print("-" * 40)
    
    ref = load_audio(ref_path)
    deg = load_audio(deg_path)
    
    # Align lengths
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]
    
    # Normalize
    ref = ref / (np.max(np.abs(ref)) + 1e-8)
    deg = deg / (np.max(np.abs(deg)) + 1e-8)
    
    results = {}
    
    # 1. PESQ (Wideband)
    try:
        results['PESQ_WB'] = pesq(16000, ref, deg, 'wb')
    except Exception as e:
        print(f"PESQ WB Error: {e}")
        results['PESQ_WB'] = 0.0
        
    # 2. STOI
    try:
        results['STOI'] = stoi(ref, deg, 16000, extended=False)
    except Exception as e:
        print(f"STOI Error: {e}")
        results['STOI'] = 0.0
        
    # 3. DNSMOS
    try:
        mos_res = dnsmos.run(deg, 16000)
        results['DNSMOS_OVRL'] = mos_res.get('ovrl_mos', mos_res.get('ovrl', 0))
        results['DNSMOS_SIG'] = mos_res.get('sig_mos', mos_res.get('sig', 0))
        results['DNSMOS_BAK'] = mos_res.get('bak_mos', mos_res.get('bak', 0))
    except Exception as e:
        print(f"DNSMOS Error: {e}")
        results['DNSMOS_OVRL'] = 0.0
        
    # Print Results
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 30)
    for k, v in results.items():
        print(f"{k:<15} | {v:.4f}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Path to reference audio (Ground Truth)")
    parser.add_argument("--deg", required=True, help="Path to degraded audio (Generated)")
    args = parser.parse_args()
    
    evaluate(args.ref, args.deg)
