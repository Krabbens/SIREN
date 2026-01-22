"""
Prepare multi-speaker dataset: LibriTTS + CommonVoice Polish
Creates manifests and precomputes HuBERT features
"""
import os
import sys
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultra_low_bitrate_codec.models.feature_extractor import HubertFeatureExtractor


def find_audio_files(directory: str, extensions=('.wav', '.flac', '.mp3', '.ogg')):
    """Recursively find all audio files in directory."""
    audio_files = []
    for ext in extensions:
        audio_files.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(audio_files)


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration using torchaudio.load (compatible with all versions)."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        return waveform.shape[1] / sr
    except Exception:
        return -1


def prepare_libritts_manifest(libritts_dir: str, output_path: str, max_duration: float = 15.0):
    """Create manifest for LibriTTS dataset."""
    print(f"Scanning LibriTTS directory: {libritts_dir}")
    
    audio_files = find_audio_files(libritts_dir, extensions=('.wav',))
    print(f"Found {len(audio_files)} audio files")
    
    manifest = []
    skipped = 0
    
    for audio_path in tqdm(audio_files, desc="Processing LibriTTS"):
        try:
            duration = get_audio_duration(str(audio_path))
            
            if duration < 0 or duration > max_duration or duration < 1.0:
                skipped += 1
                continue
            
            # Extract speaker ID from path: LibriTTS/train-clean-100/SPEAKER_ID/...
            parts = audio_path.parts
            speaker_id = None
            for i, part in enumerate(parts):
                if part == "train-clean-100" and i + 1 < len(parts):
                    speaker_id = parts[i + 1]
                    break
            
            manifest.append({
                "audio_path": str(audio_path),
                "duration": duration,
                "speaker_id": f"libritts_{speaker_id}" if speaker_id else "libritts_unknown",
                "dataset": "libritts",
                "language": "en"
            })
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            skipped += 1
    
    print(f"Valid samples: {len(manifest)}, Skipped: {skipped}")
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved manifest to: {output_path}")
    return manifest


def prepare_commonvoice_manifest(cv_dir: str, output_path: str, max_duration: float = 15.0):
    """Create manifest for CommonVoice Polish dataset."""
    print(f"Scanning CommonVoice directory: {cv_dir}")
    
    # CommonVoice structure: cv-corpus-XX/pl/clips/*.mp3
    clips_dir = Path(cv_dir)
    
    # Try to find clips directory
    possible_paths = [
        clips_dir / "clips",
        clips_dir / "pl" / "clips",
        clips_dir
    ]
    
    audio_files = []
    for path in possible_paths:
        if path.exists():
            audio_files = find_audio_files(str(path), extensions=('.mp3', '.wav', '.ogg'))
            if audio_files:
                print(f"Found clips in: {path}")
                break
    
    if not audio_files:
        print(f"No audio files found in {cv_dir}")
        return []
    
    print(f"Found {len(audio_files)} audio files")
    
    manifest = []
    skipped = 0
    
    for audio_path in tqdm(audio_files, desc="Processing CommonVoice PL"):
        try:
            duration = get_audio_duration(str(audio_path))
            
            if duration < 0 or duration > max_duration or duration < 1.0:
                skipped += 1
                continue
            
            # Use filename as pseudo-speaker (CommonVoice has client_id in TSV, but we simplify)
            manifest.append({
                "audio_path": str(audio_path),
                "duration": duration,
                "speaker_id": f"cv_pl_{audio_path.stem[:8]}",
                "dataset": "commonvoice_pl",
                "language": "pl"
            })
        except Exception as e:
            skipped += 1
    
    print(f"Valid samples: {len(manifest)}, Skipped: {skipped}")
    
    if manifest:
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to: {output_path}")
    
    return manifest


def download_commonvoice_pl(output_dir: str, max_samples: int = 10000):
    """Download CommonVoice Polish using HuggingFace datasets."""
    print("Downloading CommonVoice Polish from HuggingFace...")
    
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        print("Installing required packages...")
        os.system("pip install datasets soundfile")
        from datasets import load_dataset
        import soundfile as sf
    
    clips_dir = Path(output_dir) / "commonvoice_pl" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = Path(output_dir) / "commonvoice_pl_manifest.json"
    
    if manifest_path.exists():
        print(f"Manifest already exists: {manifest_path}")
        with open(manifest_path) as f:
            return json.load(f)
    
    print(f"Loading Polish speech dataset (max {max_samples} samples)...")
    
    # Try multiple Polish datasets in order of preference
    dataset = None
    dataset_name = None
    
    # Option 1: VoxPopuli (European Parliament speeches)
    try:
        print("Trying VoxPopuli Polish...")
        dataset = load_dataset(
            "facebook/voxpopuli", 
            "pl",
            split=f"train[:{max_samples}]"
        )
        dataset_name = "voxpopuli_pl"
        print(f"Loaded VoxPopuli Polish: {len(dataset)} samples")
    except Exception as e:
        print(f"VoxPopuli failed: {e}")
    
    # Option 2: MLS Polish (Multilingual LibriSpeech)
    if dataset is None:
        try:
            print("Trying MLS Polish...")
            dataset = load_dataset(
                "facebook/multilingual_librispeech",
                "polish",
                split=f"train[:{max_samples}]"
            )
            dataset_name = "mls_pl"
            print(f"Loaded MLS Polish: {len(dataset)} samples")
        except Exception as e:
            print(f"MLS failed: {e}")
    
    # Option 3: Just use more LibriTTS (fallback to English multi-speaker)
    if dataset is None:
        print("No Polish dataset available. Will use LibriTTS only.")
        return []
    
    manifest = []
    
    for i, sample in enumerate(tqdm(dataset, desc="Saving CommonVoice PL")):
        audio = sample["audio"]
        audio_array = audio["array"]
        sample_rate = audio["sampling_rate"]
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio_tensor = torch.tensor(audio_array).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)
            audio_array = audio_tensor.squeeze(0).numpy()
            sample_rate = 16000
        
        # Save audio
        audio_path = clips_dir / f"cv_pl_{i:06d}.wav"
        sf.write(str(audio_path), audio_array, sample_rate)
        
        duration = len(audio_array) / sample_rate
        
        if 1.0 <= duration <= 15.0:
            manifest.append({
                "audio_path": str(audio_path),
                "duration": duration,
                "speaker_id": f"cv_pl_{sample.get('client_id', 'unknown')[:8]}",
                "dataset": "commonvoice_pl",
                "language": "pl"
            })
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Downloaded {len(manifest)} samples to {clips_dir}")
    return manifest


def precompute_hubert_features(manifest: list, output_dir: str, device: str = "cuda"):
    """Precompute HuBERT features for all audio in manifest."""
    print(f"Precomputing HuBERT features for {len(manifest)} samples...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load HuBERT
    hubert = HubertFeatureExtractor(
        model_name="facebook/hubert-base-ls960",
        target_layer=9,
        freeze=True
    ).to(device)
    hubert.eval()
    
    updated_manifest = []
    
    for item in tqdm(manifest, desc="Extracting HuBERT features"):
        audio_path = item["audio_path"]
        
        # Create feature filename
        audio_name = Path(audio_path).stem
        dataset = item.get("dataset", "unknown")
        feature_path = output_path / f"{dataset}_{audio_name}.pt"
        
        if feature_path.exists():
            item["feature_path"] = str(feature_path)
            updated_manifest.append(item)
            continue
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Extract features
            waveform = waveform.squeeze(0).to(device)
            
            with torch.no_grad():
                features = hubert(waveform.unsqueeze(0))
            
            # Save features
            torch.save({
                "features": features.cpu(),
                "audio_path": audio_path,
                "duration": item["duration"]
            }, feature_path)
            
            item["feature_path"] = str(feature_path)
            updated_manifest.append(item)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    return updated_manifest


def combine_manifests(manifests: list, output_path: str, train_ratio: float = 0.95):
    """Combine multiple manifests and split into train/val."""
    import random
    
    combined = []
    for m in manifests:
        combined.extend(m)
    
    random.shuffle(combined)
    
    split_idx = int(len(combined) * train_ratio)
    train_manifest = combined[:split_idx]
    val_manifest = combined[split_idx:]
    
    # Save
    train_path = output_path.replace(".json", "_train.json")
    val_path = output_path.replace(".json", "_val.json")
    
    with open(train_path, 'w') as f:
        json.dump(train_manifest, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_manifest, f, indent=2)
    
    print(f"Train samples: {len(train_manifest)}")
    print(f"Val samples: {len(val_manifest)}")
    print(f"Saved to: {train_path}, {val_path}")
    
    return train_manifest, val_manifest


def main():
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    DEFAULT_DATA = str(PROJECT_ROOT / "data")
    DEFAULT_LIBRITTS = str(PROJECT_ROOT / "data" / "LibriTTS" / "train-clean-100")
    
    parser = argparse.ArgumentParser(description="Prepare multi-speaker dataset")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA,
                        help="Base data directory")
    parser.add_argument("--libritts-dir", type=str, 
                        default=DEFAULT_LIBRITTS,
                        help="LibriTTS directory")
    parser.add_argument("--cv-samples", type=int, default=5000,
                        help="Number of CommonVoice Polish samples to download")
    parser.add_argument("--skip-libritts", action="store_true",
                        help="Skip LibriTTS processing")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip CommonVoice processing")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip HuBERT feature extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for feature extraction")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    manifests = []
    
    # 1. LibriTTS
    if not args.skip_libritts:
        libritts_manifest_path = data_dir / "libritts_manifest.json"
        
        if libritts_manifest_path.exists():
            print(f"Loading existing LibriTTS manifest...")
            with open(libritts_manifest_path) as f:
                libritts_manifest = json.load(f)
        else:
            libritts_manifest = prepare_libritts_manifest(
                args.libritts_dir,
                str(libritts_manifest_path)
            )
        
        if libritts_manifest:
            manifests.append(libritts_manifest)
            print(f"LibriTTS: {len(libritts_manifest)} samples")
    
    # 2. CommonVoice Polish
    if not args.skip_cv:
        cv_manifest = download_commonvoice_pl(
            str(data_dir),
            max_samples=args.cv_samples
        )
        
        if cv_manifest:
            manifests.append(cv_manifest)
            print(f"CommonVoice PL: {len(cv_manifest)} samples")
    
    # 3. Combine manifests
    if manifests:
        all_samples = []
        for m in manifests:
            all_samples.extend(m)
        
        print(f"\nTotal samples: {len(all_samples)}")
        
        # 4. Precompute HuBERT features
        if not args.skip_features:
            features_dir = data_dir / "features_multispeaker"
            all_samples = precompute_hubert_features(
                all_samples,
                str(features_dir),
                device=args.device
            )
        
        # 5. Create train/val split
        combined_path = str(data_dir / "multispeaker_manifest.json")
        train_manifest, val_manifest = combine_manifests(
            [all_samples],
            combined_path
        )
        
        print("\n=== Dataset Preparation Complete ===")
        print(f"Train: {len(train_manifest)} samples")
        print(f"Val: {len(val_manifest)} samples")
        print(f"Features: {data_dir / 'features_multispeaker'}")


if __name__ == "__main__":
    main()
