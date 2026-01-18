#!/usr/bin/env python3
"""
Preprocessing script to optimize dataset loading.
Scans feature directories, finds matching audio files, and generates a static manifest.
Eliminates the need for runtime file existence checks.
"""
import os
import json
import glob
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import argparse
import sys

# Audio extensions to look for
EXTENSIONS = {'.wav', '.flac', '.mp3', '.m4a', '.opus', '.ogg'}

def scan_audio_files(root_dir):
    """
    Recursively find all audio files and build a map: stem -> path
    """
    print(f"Indexing audio files in {root_dir}...")
    audio_map = {}
    
    # Use glob for speed
    files = glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
    
    for f in tqdm(files, desc="Indexing Audio"):
        path = Path(f)
        if path.suffix.lower() in EXTENSIONS:
            audio_map[path.stem] = str(path.absolute())
            
    return audio_map

def process_file(args):
    """
    Process a single feature file using the audio map.
    """
    feature_path_str, audio_map = args
    try:
        feature_path = Path(feature_path_str)
        feature_stem = feature_path.stem
        
        # Matching logic
        # 1. Exact match
        audio_path = audio_map.get(feature_stem)
        
        # 2. Prefix removal (commonvoice_pl_cv_pl_... -> cv_pl_...)
        if not audio_path:
            # Try removing common prefixes derived from directory names or known datasets
            # Heuristic: split by '_' and try suffixes
            parts = feature_stem.split('_')
            for i in range(1, len(parts)):
                sub_stem = "_".join(parts[i:])
                if sub_stem in audio_map:
                    audio_path = audio_map[sub_stem]
                    break
        
        # 3. LibriTTS mapping heuristic (libritts_103_1241... -> 103_1241...)
        if not audio_path and feature_stem.startswith('libritts_'):
            sub_stem = feature_stem.replace('libritts_', '')
            if sub_stem in audio_map:
                audio_path = audio_map[sub_stem]
        
        if not audio_path:
            return None, f"Audio not found for {feature_stem}"
            
        # Get metadata
        try:
            info = sf.info(audio_path)
            duration = info.duration
            samplerate = info.samplerate
            channels = info.channels
        except Exception as e:
            return None, f"Corrupt audio {audio_path}: {e}"
        
        entry = {
            "feature_path": str(feature_path.absolute()),
            "audio_path": audio_path,
            "duration": duration,
            "samplerate": samplerate,
            "channels": channels
        }
        return entry, None
        
    except Exception as e:
        return None, str(e)

def main():
    parser = argparse.ArgumentParser(description="Generate optimized dataset manifest")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory containing .pt features")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory searching for audio")
    parser.add_argument("--output", type=str, default="optimized_manifest.json", help="Output JSON path")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of worker processes")
    args = parser.parse_args()
    
    # 1. Index Audio
    audio_map = scan_audio_files(args.data_root)
    print(f"Indexed {len(audio_map)} unique audio files.")
    
    # 2. Scan Features
    print(f"Scanning features in {args.feature_dir}...")
    feature_files = glob.glob(os.path.join(args.feature_dir, "**", "*.pt"), recursive=True)
    
    if not feature_files:
        print("No feature files found.")
        return

    print(f"Found {len(feature_files)} features. Matching...")
    
    # Prepare args for workers (audio_map is read-only shared)
    worker_args = [(f, audio_map) for f in feature_files]
    
    valid_entries = []
    errors = []
    
    # Use ProcessPoolExecutor to speed up metadata reading (sf.info)
    # Note: audio_map is pickled to workers. It might be large but efficient enough.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_file, worker_args), total=len(worker_args)))
        
    for entry, error in results:
        if entry:
            valid_entries.append(entry)
        else:
            errors.append(error)
            
    print(f"\nProcessing complete.")
    print(f"Valid entries: {len(valid_entries)}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("Sample errors:")
        for e in errors[:5]:
            print(f" - {e}")
            
    with open(args.output, 'w') as f:
        json.dump(valid_entries, f, indent=2)
        
    print(f"\nManifest saved to {args.output}")

if __name__ == "__main__":
    main()
