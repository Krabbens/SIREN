
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import os
import glob
import numpy as np
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from ultra_low_bitrate_codec.models.micro_hubert import MicroHuBERT

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Checking Feature Stats on {device}...")
    
    # 1. Training Statistics (from stored files)
    print("\n1. Analyzing Training Data (data/features_distilhubert)...")
    files = glob.glob("data/features_distilhubert/*.pt")[:50] # Check 50 files
    if not files:
        print("ERROR: No files found in data/features_distilhubert/")
        return
        
    all_feats = []
    for f in files:
        try:
            d = torch.load(f, map_location='cpu')
            # Handle different formats just in case
            if isinstance(d, dict) and 'features' in d: feat = d['features']
            else: feat = d
            all_feats.append(feat)
        except: pass
        
    if all_feats:
        train_tensor = torch.cat(all_feats, dim=0)
        t_mean = train_tensor.mean().item()
        t_std = train_tensor.std().item()
        t_min = train_tensor.min().item()
        t_max = train_tensor.max().item()
        print(f"   Files: {len(all_feats)}")
        print(f"   Mean: {t_mean:.4f}")
        print(f"   Std:  {t_std:.4f}")
        print(f"   Min:  {t_min:.4f}")
        print(f"   Max:  {t_max:.4f}")
    
    # 2. Live DistilHuBERT (Official)
    print("\n2. Analyzing Live DistilHuBERT (ntu-spml/distilhubert)...")
    feature_extractor = AutoModel.from_pretrained("ntu-spml/distilhubert").to(device)
    feature_extractor.eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained("ntu-spml/distilhubert")
    
    wav, sr = sf.read("data/jakubie.wav")
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() > 1: wav = wav.mean(dim=0)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    wav_norm = wav / (wav.abs().max() + 1e-6)
    
    with torch.no_grad():
        inputs = processor(wav_norm, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        d_feats = feature_extractor(input_values).last_hidden_state
        
        d_mean = d_feats.mean().item()
        d_std = d_feats.std().item()
        print(f"   Mean: {d_mean:.4f}")
        print(f"   Std:  {d_std:.4f}")
        print(f"   Min:  {d_feats.min().item():.4f}")
        print(f"   Max:  {d_feats.max().item():.4f}")
        
    # 3. Live MicroHuBERT
    print("\n3. Analyzing Live MicroHuBERT...")
    micro = MicroHuBERT().to(device)
    ckpt = torch.load("checkpoints/microhubert/microhubert_ep95.pt", map_location=device)
    if isinstance(ckpt, dict): 
        if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    micro.load_state_dict(ckpt)
    micro.eval()
    
    with torch.no_grad():
        m_feats = micro(wav_norm.unsqueeze(0).to(device))
        
        m_mean = m_feats.mean().item()
        m_std = m_feats.std().item()
        print(f"   Mean: {m_mean:.4f}")
        print(f"   Std:  {m_std:.4f}")
        print(f"   Min:  {m_feats.min().item():.4f}")
        print(f"   Max:  {m_feats.max().item():.4f}")

    # Comparison logic
    print("\n" + "="*30)
    print("ANALYSIS")
    print("="*30)
    
    diff_distil_mean = abs(d_mean - t_mean)
    diff_micro_mean = abs(m_mean - t_mean)
    
    print(f"Distil Diff (Mean): {diff_distil_mean:.4f}")
    print(f"Micro Diff (Mean):  {diff_micro_mean:.4f}")
    
    if diff_distil_mean < diff_micro_mean:
        print(">> Training Data matches OFFICIAL DistilHuBERT better.")
    else:
        print(">> Training Data matches MicroHuBERT better.")
        
    if diff_distil_mean > 1.0 and diff_micro_mean > 1.0:
        print(">> WARNING: Both models deviate significantly from Training Data!")
        print(">> Possible Scaling Issue (e.g. LayerNorm missing in precompute?).")

if __name__ == "__main__":
    main()
