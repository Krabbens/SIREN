#!/usr/bin/env python3
"""
Compare Official HuBERT vs TinyHubert features.
"""
import sys
from unittest.mock import MagicMock
# Avoid torchcodec issues
m = MagicMock()
m.__spec__ = MagicMock()
sys.modules['torchcodec'] = m
sys.modules['torchcodec.decoders'] = m
sys.modules['torchcodec._core'] = m
sys.modules['torchcodec._core.ops'] = m

import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from ultra_low_bitrate_codec.models.tiny_hubert import TinyHubert
import numpy as np

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Audio
    print("Loading Audio...")
    wav_data, sr = sf.read("data/jakubie_16k.wav")
    wav = torch.tensor(wav_data, dtype=torch.float32).to(device)
    if wav.dim() == 1: wav = wav.unsqueeze(0)
    else: wav = wav.T
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
        wav = resampler(wav)
    
    # Normalize
    wav = wav / (wav.abs().max() + 1e-6)
    
    # 1. Official HuBERT
    print("Loading Official HuBERT...")
    proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model_big = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    with torch.no_grad():
        inputs = proc(wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        out_big = model_big(inputs.input_values.to(device), output_hidden_states=True)
        feat_big = out_big.hidden_states[9] # Layer 9
    
    # 2. TinyHubert
    print("Loading TinyHubert...")
    model_tiny = TinyHubert(out_dim=768, hidden_dim=384, num_layers=12).to(device)
    model_tiny.load_state_dict(torch.load("checkpoints/tiny_hubert_best.pt", map_location=device))
    model_tiny.eval()
    
    with torch.no_grad():
        feat_tiny = model_tiny(wav)
        
    # Compare
    print(f"\nShape Big: {feat_big.shape}")
    print(f"Shape Tiny: {feat_tiny.shape}")
    
    # Align lengths
    min_len = min(feat_big.shape[1], feat_tiny.shape[1])
    f_big = feat_big[:, :min_len, :]
    f_tiny = feat_tiny[:, :min_len, :]
    
    # Metrics
    mse = torch.nn.functional.mse_loss(f_big, f_tiny)
    cos = torch.nn.functional.cosine_similarity(f_big, f_tiny, dim=-1).mean()
    
    print(f"\n--- Comparison ---")
    print(f"MSE Loss: {mse.item():.4f}")
    print(f"Cosine Similarity: {cos.item():.4f}")
    
    if cos < 0.9:
        print("\n❌ MISMATCH DETECTED! TinyHubert is NOT aligned with Official HuBERT.")
        print("This explains why Factorizer (trained on Official) fails with TinyHubert.")
    else:
        print("\n✅ Features are similar. Mismatch unlikely to be the main cause.")

if __name__ == "__main__":
    main()
