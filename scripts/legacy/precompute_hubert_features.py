#!/usr/bin/env python3
"""
Precompute Official HuBERT Features (Layer 9) for MicroHuBERT Distillation

This script:
1. Loads Official HuBERT (facebook/hubert-base-ls960)
2. Processes audio files
3. Extracts Layer 9 hidden states (the standard for content/units)
4. Saves as .pt files for training MicroHuBERT

Usage:
    python precompute_hubert_features.py \
        --audio_dir data/audio \
        --output_dir data/features_hubert_layer9
"""

import os
import argparse
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json
from transformers import HubertModel, Wav2Vec2FeatureExtractor

def precompute_features(
    audio_dir: str,
    output_dir: str,
    max_duration: float = 30.0,
    device: str = 'cuda'
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Precomputing HuBERT Layer 9 Features")
    print("=" * 60)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load HuBERT
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    
    # Scan files
    audio_files = []
    for ext in ['*.wav', '*.flac', '*.mp3', '*.m4a']:
        audio_files.extend(list(Path(audio_dir).rglob(ext)))
    audio_files = [str(f) for f in audio_files]
    print(f"Found {len(audio_files)} audio files")
    
    metadata = {
        'model': 'hubert-base-ls960',
        'layer': 9,
        'feature_dim': 768,
        'files': []
    }
    
    processed = 0
    failed = 0
    
    for audio_path in tqdm(audio_files, desc="Processing"):
        try:
            # Load audio
            wav, sr = sf.read(audio_path)
            wav = torch.tensor(wav, dtype=torch.float32)
            
            if wav.dim() > 1:
                wav = wav.mean(dim=0)
                
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
                
            # Normalize
            wav = wav / (wav.abs().max() + 1e-6)
            
            # Truncate
            max_samples = int(max_duration * 16000)
            if wav.shape[0] > max_samples:
                wav = wav[:max_samples]
                
            # Process
            wav_np = wav.unsqueeze(0).numpy()
            inputs = processor(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(inputs.input_values.to(device), output_hidden_states=True)
                # Extract Layer 9
                features = outputs.hidden_states[9] 
                
            features = features.squeeze(0).cpu() # (T, 768)
            
            # Save
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            out_path = os.path.join(output_dir, f"{basename}.pt")
            
            torch.save({
                'features': features,
                'duration': wav.shape[0]/16000
            }, out_path)
            
            metadata['files'].append(basename)
            processed += 1
            
        except Exception as e:
            # print(f"Failed {audio_path}: {e}")
            failed += 1
            
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Done. Processed: {processed}, Failed: {failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    precompute_features(args.audio_dir, args.output_dir)
