#!/usr/bin/env python3
"""
Download and prepare LibriTTS dataset for multi-speaker training.
Uses HuggingFace datasets for easy access.
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def download_libritts(output_dir: str, subset: str = "train-clean-100", 
                      max_hours: float = None, cache_dir: str = None):
    """
    Download LibriTTS from HuggingFace and prepare manifest files.
    
    Args:
        output_dir: Directory to save audio files and manifests
        subset: One of 'train-clean-100', 'train-clean-360', 'train-other-500', etc.
        max_hours: Optional limit on total hours to download
        cache_dir: Optional HF cache directory
    """
    from datasets import load_dataset
    import soundfile as sf
    import numpy as np
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    print(f"📥 Downloading LibriTTS '{subset}' from HuggingFace...")
    
    # Map subset to config and split
    # Available configs: ['dev', 'clean', 'other', 'all']
    # Example splits: 'train.clean.100', 'train.clean.360', 'train.other.500', 'dev.clean'
    
    config_name = "clean"
    split_name = "train.clean.100"
    
    if "clean" in subset:
        config_name = "clean"
    elif "other" in subset:
        config_name = "other"
    else:
        config_name = "all"
        
    # Replace hyphens with dots for split name (e.g. train-clean-100 -> train.clean.100)
    split_name = subset.replace("-", ".")
    
    print(f"🔍 Mapped request '{subset}' to config='{config_name}', split='{split_name}'")
    
    # Load dataset
    try:
        if cache_dir:
            ds = load_dataset("mythicinfinity/libritts", config_name, 
                             split=split_name, cache_dir=cache_dir)
        else:
            ds = load_dataset("mythicinfinity/libritts", config_name, split=split_name)
    except ValueError as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"Attempting fallback to 'all' config matching...")
        # Fallback: try 'all' config if specific one fails
        ds = load_dataset("mythicinfinity/libritts", "all", split=split_name)
    
    print(f"📊 Dataset size: {len(ds)} samples")
    
    # Collect speaker info
    speaker_ids = set()
    train_manifest = []
    val_manifest = []
    
    total_duration = 0.0
    max_duration_seconds = max_hours * 3600 if max_hours else float('inf')
    
    # Process samples
    for i, sample in enumerate(tqdm(ds, desc="Processing audio")):
        # Check duration limit
        duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']
        
        if total_duration >= max_duration_seconds:
            print(f"⏱️ Reached max duration limit ({max_hours}h), stopping.")
            break
        
        # Filter by duration (1-15 seconds)
        if duration < 1.0 or duration > 15.0:
            continue
        
        speaker_id = sample['speaker_id']
        speaker_ids.add(speaker_id)
        
        # Save audio file
        filename = f"{speaker_id}_{sample['id']}.wav"
        filepath = audio_dir / filename
        
        audio_array = sample['audio']['array']
        sr = sample['audio']['sampling_rate']
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Save
        sf.write(str(filepath), audio_array, sr)
        
        # Create manifest entry
        entry = {
            "audio_path": str(filepath),
            "duration": duration,
            "speaker_id": str(speaker_id),
            "text": sample.get('text_normalized', sample.get('text', ''))
        }
        
        # 95/5 train/val split
        if i % 20 == 0:
            val_manifest.append(entry)
        else:
            train_manifest.append(entry)
        
        total_duration += duration
    
    # Save manifests
    train_path = output_path / "libritts_train.json"
    val_path = output_path / "libritts_val.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_manifest, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_manifest, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("✅ DOWNLOAD COMPLETE")
    print("="*60)
    print(f"📁 Audio directory: {audio_dir}")
    print(f"📄 Train manifest: {train_path} ({len(train_manifest)} samples)")
    print(f"📄 Val manifest: {val_path} ({len(val_manifest)} samples)")
    print(f"🎤 Unique speakers: {len(speaker_ids)}")
    print(f"⏱️ Total duration: {total_duration/3600:.2f} hours")
    
    # Save speaker list
    speaker_path = output_path / "speakers.json"
    with open(speaker_path, 'w') as f:
        json.dump(sorted(list(speaker_ids)), f)
    print(f"🎤 Speaker list saved to: {speaker_path}")
    
    return train_path, val_path


def prepare_from_existing_libritts(libritts_dir: str, output_dir: str):
    """
    Prepare manifests from existing LibriTTS download (OpenSLR format).
    """
    libritts_path = Path(libritts_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_manifest = []
    val_manifest = []
    speaker_ids = set()
    total_duration = 0.0
    
    # Find all WAV files
    wav_files = list(libritts_path.rglob("*.wav"))
    print(f"📊 Found {len(wav_files)} audio files")
    
    import torchaudio
    
    for i, wav_path in enumerate(tqdm(wav_files, desc="Processing")):
        try:
            info = torchaudio.info(str(wav_path))
            duration = info.num_frames / info.sample_rate
            
            # Filter
            if duration < 1.0 or duration > 15.0:
                continue
            
            # Parse speaker ID from path (format: speaker/chapter/speaker_chapter_uttid.wav)
            parts = wav_path.stem.split('_')
            if len(parts) >= 3:
                speaker_id = parts[0]
            else:
                speaker_id = wav_path.parent.parent.name
            
            speaker_ids.add(speaker_id)
            
            entry = {
                "audio_path": str(wav_path.absolute()),
                "duration": duration,
                "speaker_id": speaker_id,
            }
            
            if i % 20 == 0:
                val_manifest.append(entry)
            else:
                train_manifest.append(entry)
            
            total_duration += duration
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue
    
    # Save
    train_path = output_path / "libritts_train.json"
    val_path = output_path / "libritts_val.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_manifest, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_manifest, f, indent=2)
    
    print(f"\n✅ Prepared {len(train_manifest)} train, {len(val_manifest)} val samples")
    print(f"🎤 {len(speaker_ids)} unique speakers, {total_duration/3600:.2f} hours total")
    
    return train_path, val_path


def download_common_voice_pl(output_dir: str, max_hours: float = None, cache_dir: str = None):
    """
    Download Mozilla Common Voice 11.0 (Polish) and prepare manifest files.
    """
    from datasets import load_dataset, Audio
    import soundfile as sf
    import numpy as np
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    audio_dir = output_path / "audio_cv_pl"
    audio_dir.mkdir(exist_ok=True)
    
    print(f"📥 Downloading Common Voice 11.0 'pl' from HuggingFace...")
    
    # Authenticated access might be required for newer CV, assuming user has access or using older open version if issues arise
    # CV 11.0 usually requires auth. If fails, user needs to login via `huggingface-cli login`.
    
    try:
        # trust_remote_code is not needed/supported for canonical datasets usually
        ds = load_dataset("mozilla-foundation/common_voice_11_0", "pl", split="train", cache_dir=cache_dir)
        # Cast audio to 16kHz
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    except Exception as e:
        print(f"❌ Error loading Common Voice 11.0: {e}")
        print("🔄 Trying older version 'common_voice'...")
        try:
             ds = load_dataset("common_voice", "pl", split="train", cache_dir=cache_dir)
             ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        except Exception as e2:
             print(f"❌ Error loading fallback 'common_voice': {e2}")
             return None, None

    print(f"📊 Dataset size: {len(ds)} samples")

    speaker_ids = set()
    train_manifest = []
    val_manifest = []
    
    total_duration = 0.0
    max_duration_seconds = max_hours * 3600 if max_hours else float('inf')
    
    # Process
    for i, sample in enumerate(tqdm(ds, desc="Processing Common Voice PL")):
        audio_array = sample['audio']['array']
        sr = sample['audio']['sampling_rate'] # Should be 16000 now
        
        duration = len(audio_array) / sr
        
        if total_duration >= max_duration_seconds:
            print(f"⏱️ Reached max duration limit, stopping.")
            break
            
        if duration < 1.0 or duration > 15.0:
            continue
            
        client_id = sample['client_id']
        # Shorten client_id for speaker_id
        speaker_id = f"cv_{client_id[:12]}"
        speaker_ids.add(speaker_id)
        
        filename = f"{speaker_id}_{i}.wav"
        filepath = audio_dir / filename
        
        sf.write(str(filepath), audio_array, sr)
        
        entry = {
            "audio_path": str(filepath),
            "duration": duration,
            "speaker_id": speaker_id,
            "text": sample.get('sentence', '')
        }
        
        # 95/5 split
        if i % 20 == 0:
            val_manifest.append(entry)
        else:
            train_manifest.append(entry)
            
        total_duration += duration
        
    # Save manifests
    train_path = output_path / "cv_pl_train.json"
    val_path = output_path / "cv_pl_val.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_manifest, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_manifest, f, indent=2)
        
    print(f"\n✅ Prepared {len(train_manifest)} train, {len(val_manifest)} val samples from Common Voice PL")
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Download/prepare datasets")
    parser.add_argument("--output-dir", type=str, default="/home/sperm/diff/data",
                       help="Output directory for audio and manifests")
    parser.add_argument("--dataset", type=str, default="libritts", choices=["libritts", "common_voice_pl", "all"],
                       help="Dataset to download")
    parser.add_argument("--subset", type=str, default="train-clean-100",
                       help="LibriTTS subset to download")
    parser.add_argument("--max-hours", type=float, default=None,
                       help="Maximum hours to download (None = all)")
    parser.add_argument("--existing-dir", type=str, default=None,
                       help="Path to existing LibriTTS download (skip HF download)")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="HuggingFace cache directory")
    
    args = parser.parse_args()
    
    if args.dataset in ["libritts", "all"]:
        if args.existing_dir:
            prepare_from_existing_libritts(args.existing_dir, args.output_dir)
        else:
            download_libritts(args.output_dir, args.subset, args.max_hours, args.cache_dir)
            
    if args.dataset in ["common_voice_pl", "all"]:
        download_common_voice_pl(args.output_dir, args.max_hours, args.cache_dir)

if __name__ == "__main__":
    main()
