#!/usr/bin/env python3
import torch
import torch.nn as nn
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from ultra_low_bitrate_codec.models.micro_hubert import MicroHuBERT

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Audio
    # Use a long enough sample to get stable stats
    wav_path = "data/jakubie_16k.wav"
    wav, sr = sf.read(wav_path)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() > 1: wav = wav.mean(0) # Mono
    
    # Normalize
    wav = wav / (wav.abs().max() + 1e-6)
    wav = wav.to(device)
    
    print(f"Audio loaded: {wav.shape}, range=[{wav.min():.3f}, {wav.max():.3f}]")
    
    # 2. Official HuBERT (Reference)
    print("\n--- Official HuBERT (Layer 9) ---")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    with torch.no_grad():
        inputs = processor(wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        outputs = hubert(inputs.input_values.to(device), output_hidden_states=True)
        feat_ref = outputs.hidden_states[9] # Layer 9
        
    print(f"Ref Features: {feat_ref.shape}")
    print(f"Mean: {feat_ref.mean():.4f}")
    print(f"Std:  {feat_ref.std():.4f}")
    print(f"Min:  {feat_ref.min():.4f}")
    print(f"Max:  {feat_ref.max():.4f}")
    
    # 3. MicroHuBERT (Trained)
    print("\n--- MicroHuBERT (Epoch 95) ---")
    micro = MicroHuBERT().to(device)
    ckpt_path = "checkpoints/microhubert/microhubert_ep95.pt"
    state_dict = torch.load(ckpt_path, map_location=device)
    # Fix for torch.compile prefixes
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    micro.load_state_dict(state_dict)
    micro.eval()
    
    with torch.no_grad():
        feat_micro = micro(wav.unsqueeze(0))
        
    print(f"Micro Features: {feat_micro.shape}")
    print(f"Mean: {feat_micro.mean():.4f}")
    print(f"Std:  {feat_micro.std():.4f}")
    print(f"Min:  {feat_micro.min():.4f}")
    print(f"Max:  {feat_micro.max():.4f}")
    
    # Check Length Mismatch
    min_len = min(feat_ref.shape[1], feat_micro.shape[1])
    feat_ref = feat_ref[:, :min_len]
    feat_micro = feat_micro[:, :min_len]
    
    # 4. Correlation & L1
    l1_diff = (feat_ref - feat_micro).abs().mean().item()
    print(f"\nL1 Diff: {l1_diff:.4f}")
    
    # Cosine Sim
    cos_sim = torch.nn.functional.cosine_similarity(feat_ref, feat_micro, dim=-1).mean().item()
    print(f"Cosine Similarity: {cos_sim:.4f}")
    
    # 5. Plot Histograms
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(feat_ref.reshape(-1).cpu().numpy(), bins=100, alpha=0.5, label='Official HuBERT', density=True)
    plt.hist(feat_micro.reshape(-1).cpu().numpy(), bins=100, alpha=0.5, label='MicroHuBERT', density=True)
    plt.legend()
    plt.title("Feature Value Distribution")
    
    # Plot a few dimensions over time
    plt.subplot(1, 2, 2)
    # Pick dim 0
    plt.plot(feat_ref[0, :200, 0].cpu().numpy(), label='Official', alpha=0.8)
    plt.plot(feat_micro[0, :200, 0].cpu().numpy(), label='Micro', alpha=0.8)
    plt.legend()
    plt.title("Dim 0 over Time (First 200 frames)")
    
    plt.tight_layout()
    plt.savefig("outputs/micro_feature_analysis.png")
    print("\nSaved plot to outputs/micro_feature_analysis.png")

if __name__ == "__main__":
    main()
