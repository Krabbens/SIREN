#!/usr/bin/env python3
"""
Precompute HuBERT features for LibriTTS or any multi-speaker dataset.
This script can process a raw directory of audio files or a JSON manifest.
"""
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import os
import glob


def precompute_features(
    input_path: str,
    output_dir: str,
    hubert_layer: int = 9,
    device: str = "cuda"
):
    """
    Extract HuBERT features from audio files and save as .pt files.
    
    Args:
        input_path: Path to directory containing .wav files OR Path to JSON manifest
        output_dir: Directory to save .pt feature files
        hubert_layer: Which HuBERT layer to extract (9 is phonetic-rich)
        device: cuda or cpu
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # input_path can be a directory or a manifest file
    input_p = Path(input_path)
    data = []
    
    if input_p.is_file() and input_p.suffix == '.json':
        print(f"📄 Loading manifest: {input_path}")
        with open(input_path, 'r') as f:
            data = json.load(f)
    elif input_p.is_dir():
        print(f"📂 Scanning directory: {input_path}")
        wav_files = list(input_p.rglob("*.wav"))
        print(f"   Found {len(wav_files)} .wav files")
        # create temporary data structure
        data = [{'audio_path': str(p)} for p in wav_files]
    else:
        print(f"❌ Input path not found or invalid: {input_path}")
        return

    print(f"📄 Processing {len(data)} files")
    
    # Load HuBERT
    print("🔧 Loading HuBERT model...")
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
        model.eval()
        print("✅ HuBERT loaded")
    except Exception as e:
        print(f"❌ Error loading HuBERT: {e}")
        return
    
    # Process files
    skipped = 0
    processed = 0
    
    for item in tqdm(data, desc="Extracting features"):
        audio_path = item.get('audio_path', item.get('path', item))
        if not os.path.exists(audio_path):
            skipped += 1
            continue
        
        # Output path - mirror directory structure if input was a dir, or flat if manifest?
        # Let's verify flat vs structure. The previous script did flat basename. 
        # But for large datasets flat might have collisions. 
        # For simplicity and compatibility with existing dataset class, let's stick to flat basename for now unless collisions occur.
        # Ideally, we should use a relative path preservation if possible, but let's stick to what works for now with the existing Training Dataset loader which likely expects flattened or specific structure.
        # Actually checking FeatureDataset: it usually looks up by basename or provided path.
        # Let's use stem for now.
        
        basename = Path(audio_path).stem
        output_file = output_path / f"{basename}.pt"
        
        # Skip if already exists
        if output_file.exists():
            processed += 1
            continue
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Skip very short or long files (optional, but good for training data)
            duration = waveform.shape[1] / 16000
            if duration < 0.5 or duration > 30.0:
                skipped += 1
                continue
            
            # Extract features
            with torch.no_grad():
                inputs = processor(
                    waveform.squeeze().numpy(),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, output_hidden_states=True)
                features = outputs.hidden_states[hubert_layer]  # (1, T, 768)
                features = features.squeeze(0).cpu()  # (T, 768)
            
            # Save
            torch.save(features, output_file)
            processed += 1
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            skipped += 1
            continue
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   Processed: {processed}")
    print(f"   Skipped: {skipped}")
    print(f"   Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Precompute HuBERT features")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to directory containing .wav files OR JSON manifest")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save features")
    parser.add_argument("--hubert-layer", type=int, default=9,
                       help="HuBERT layer to extract (default: 9)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    precompute_features(
        input_path=args.input,
        output_dir=args.output_dir,
        hubert_layer=args.hubert_layer,
        device=args.device
    )

if __name__ == "__main__":
    main()
