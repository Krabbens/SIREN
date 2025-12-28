#!/usr/bin/env python3
"""
Precompute HuBERT features for LibriTTS or any multi-speaker dataset.
"""
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import os


def precompute_features(
    manifest_path: str,
    output_dir: str,
    hubert_layer: int = 9,
    device: str = "cuda"
):
    """
    Extract HuBERT features from audio files and save as .pt files.
    
    Args:
        manifest_path: Path to JSON manifest with audio_path entries
        output_dir: Directory to save .pt feature files
        hubert_layer: Which HuBERT layer to extract (9 is phonetic-rich)
        device: cuda or cpu
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    print(f"📄 Loaded manifest with {len(data)} entries")
    
    # Load HuBERT
    print("🔧 Loading HuBERT model...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()
    print("✅ HuBERT loaded")
    
    # Process files
    skipped = 0
    processed = 0
    
    for item in tqdm(data, desc="Extracting features"):
        audio_path = item.get('audio_path', item.get('path', item))
        if not os.path.exists(audio_path):
            skipped += 1
            continue
        
        # Output path
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
            
            # Skip very short or long files
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


def update_manifest_for_features(original_manifest: str, feature_dir: str, output_manifest: str):
    """
    Create a new manifest with only entries that have precomputed features.
    """
    with open(original_manifest, 'r') as f:
        data = json.load(f)
    
    feature_path = Path(feature_dir)
    filtered = []
    
    for item in data:
        audio_path = item.get('audio_path', item.get('path', item))
        basename = Path(audio_path).stem
        feat_file = feature_path / f"{basename}.pt"
        
        if feat_file.exists():
            # Update with new format
            if isinstance(item, str):
                item = {'path': item}
            item['path'] = audio_path
            filtered.append(item)
    
    with open(output_manifest, 'w') as f:
        json.dump(filtered, f, indent=2)
    
    print(f"✅ Created filtered manifest: {output_manifest}")
    print(f"   {len(filtered)}/{len(data)} samples have features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute HuBERT features")
    parser.add_argument("--manifest", type=str, required=True,
                       help="Path to JSON manifest")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save features")
    parser.add_argument("--hubert-layer", type=int, default=9,
                       help="HuBERT layer to extract (default: 9)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--update-manifest", type=str, default=None,
                       help="If set, create updated manifest at this path")
    
    args = parser.parse_args()
    
    precompute_features(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        hubert_layer=args.hubert_layer,
        device=args.device
    )
    
    if args.update_manifest:
        update_manifest_for_features(
            args.manifest, 
            args.output_dir, 
            args.update_manifest
        )
