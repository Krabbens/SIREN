#!/usr/bin/env python3
"""
Precompute DistilHuBERT Features for Factorizer Training

This script:
1. Loads DistilHuBERT (optionally INT8 quantized)
2. Processes all audio files in data directory
3. Saves features as .pt files for fast training

Usage:
    # With FP32 DistilHuBERT
    python precompute_distilhubert_features.py \
        --audio_dir data/audio \
        --output_dir data/features_distilhubert \
        --manifest data/manifest_train.txt

    # With INT8 quantized
    python precompute_distilhubert_features.py \
        --audio_dir data/audio \
        --output_dir data/features_distilhubert_int8 \
        --quantized_model checkpoints/distilhubert_int8
"""

import os
import argparse
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json


import sys
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.micro_hubert import MicroHuBERT


def load_model(checkpoint_path: str = None, quantized_path: str = None):
    """Load model (MicroHuBERT or DistilHuBERT)."""
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    
    if checkpoint_path:
        print(f"Loading MicroHuBERT from {checkpoint_path}")
        model = MicroHuBERT()
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        # Handle torch.compile keys
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model, None # No processor needed for MicroHuBERT
        
    elif quantized_path and os.path.exists(f"{quantized_path}/distilhubert_int8.pt"):
        print(f"Loading INT8 quantized model from {quantized_path}")
        from scripts.quantize_distilhubert import QuantizedDistilHuBERT
        return QuantizedDistilHuBERT.load(quantized_path), None
    else:
        print("Loading FP32 DistilHuBERT from HuggingFace")
        model = AutoModel.from_pretrained("ntu-spml/distilhubert")
        processor = Wav2Vec2FeatureExtractor.from_pretrained("ntu-spml/distilhubert")
        model.eval()
        return model, processor


def precompute_features(
    audio_dir: str,
    output_dir: str,
    manifest_path: str = None,
    checkpoint_path: str = None, # Added
    quantized_model: str = None,
    sample_rate: int = 16000,
    max_duration: float = 30.0,
    device: str = 'cuda'
):
    """Precompute Features for all audio files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # Load Model
    # =========================================================================
    print("=" * 60)
    print("Precomputing Features")
    print("=" * 60)
    
    if quantized_model:
        # INT8 quantized - must run on CPU
        device = 'cpu'
    
    model, processor = load_model(checkpoint_path, quantized_model)
    model = model.to(device)
    model.eval()
    
    device = torch.device(device)
    print(f"Device: {device}")
    
    # =========================================================================
    # Get Audio Files
    # =========================================================================
    if manifest_path and os.path.exists(manifest_path):
        print(f"Loading manifest: {manifest_path}")
        with open(manifest_path) as f:
            audio_files = [line.strip().split('\t')[0] for line in f if line.strip()]
        # Make absolute paths
        audio_files = [os.path.join(audio_dir, f) if not os.path.isabs(f) else f 
                       for f in audio_files]
    else:
        print(f"Scanning directory: {audio_dir}")
        audio_files = []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            audio_files.extend(list(Path(audio_dir).rglob(ext)))
        audio_files = [str(f) for f in audio_files]
    
    print(f"Found {len(audio_files)} audio files")
    
    # =========================================================================
    # Process Files
    # =========================================================================
    max_samples = int(max_duration * sample_rate)
    processed = 0
    failed = 0
    
    metadata = {
        'model': 'distilhubert' if not quantized_model else 'distilhubert_int8',
        'sample_rate': sample_rate,
        'feature_dim': 768,
        'files': []
    }
    
    for audio_path in tqdm(audio_files, desc="Processing"):
        try:
            # Load audio
            if audio_path.endswith('.wav') or audio_path.endswith('.flac'):
                wav, sr = sf.read(audio_path)
                wav = torch.tensor(wav, dtype=torch.float32)
            else:
                wav, sr = torchaudio.load(audio_path)
                wav = wav.squeeze(0)
            
            # Convert to mono
            if wav.dim() > 1:
                wav = wav.mean(dim=0)
            
            # Resample if needed
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            
            # Truncate if too long
            if wav.shape[0] > max_samples:
                wav = wav[:max_samples]
            
            # Normalize
            wav = wav / (wav.abs().max() + 1e-6)
            
            # Extract features
            wav = wav.unsqueeze(0).to(device)
            
            with torch.no_grad():
                if processor is None:
                    # MicroHuBERT or Quantized: accepts raw waveform input directly
                    # Input: (B, T)
                    features = model(wav) 
                    # If it returns a tensor directly (MicroHuBERT), use it
                    # If it's tuple/dict, extract. MicroHuBERT returns (1, T/320, 768) tensor.
                else:
                    # HuggingFace Pretrained
                    outputs = model(wav, output_hidden_states=True)
                    features = outputs.last_hidden_state
            
            features = features.squeeze(0).cpu()  # (T, 768)
            
            # Save
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            out_path = os.path.join(output_dir, f"{basename}.pt")
            torch.save({
                'features': features,
                'audio_path': audio_path,
                'num_frames': features.shape[0],
                'duration': wav.shape[1] / sample_rate
            }, out_path)
            
            metadata['files'].append({
                'name': basename,
                'frames': features.shape[0],
                'duration': wav.shape[1] / sample_rate
            })
            
            processed += 1
            
        except Exception as e:
            print(f"Failed: {audio_path}: {e}")
            failed += 1
            continue
    
    # =========================================================================
    # Save Metadata
    # =========================================================================
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"   Processed: {processed}")
    print(f"   Failed: {failed}")
    print(f"   Output: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None, 
                        help="Path to MicroHuBERT checkpoint")
    parser.add_argument("--quantized_model", default=None, 
                        help="Path to INT8 quantized model dir")
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()
    
    precompute_features(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        quantized_model=args.quantized_model,
        max_duration=args.max_duration,
        device=args.device
    )


if __name__ == "__main__":
    main()
