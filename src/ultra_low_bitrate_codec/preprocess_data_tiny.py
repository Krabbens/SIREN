#!/usr/bin/env python3
"""
Precompute TinyHubert features for training the Factorizer Adapter.
"""
import torch
import torchaudio
import argparse
import os
import glob
from pathlib import Path
from tqdm import tqdm
from ultra_low_bitrate_codec.models.tiny_hubert import TinyHubert
import soundfile as sf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input directory (wavs)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tiny_hubert_best.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load TinyHubert
    print(f"Loading TinyHubert from {args.checkpoint}...")
    model = TinyHubert(out_dim=768, hidden_dim=384, num_layers=12).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # 2. Find Files
    if args.input.endswith('.json'):
        import json
        with open(args.input, 'r') as f:
            data = json.load(f)
            # data is list of dicts with 'audio_path'
            files = [d['audio_path'] for d in data]
        print(f"Loaded {len(files)} files from manifest.")
    else:
        files = glob.glob(os.path.join(args.input, "**/*.wav"), recursive=True)
        print(f"Found {len(files)} files in directory.")
    
    # Check for existing to skip
    if os.path.exists(args.output_dir):
        existing = set(os.listdir(args.output_dir))
        files = [f for f in files if Path(f).stem + ".pt" not in existing]
        print(f"Processing {len(files)} new files (skipping {len(data)-len(files) if args.input.endswith('.json') else 'some'}).")
    
    # 3. Process
    batch = []
    batch_paths = []
    
    with torch.no_grad():
        for i, fpath in enumerate(tqdm(files)):
            try:
                # Load Audio
                # Check file exists
                if not os.path.exists(fpath):
                    continue
                    
                wav, sr = sf.read(fpath)
                wav = torch.tensor(wav, dtype=torch.float32)
                if wav.dim() > 1: wav = wav.mean(1) # Mix to mono if stereo check
                
                # Resample
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    wav = resampler(wav)
                    
                # Normalize
                wav = wav / (wav.abs().max() + 1e-6)
                
                batch.append(wav)
                batch_paths.append(fpath)
                
                if len(batch) >= args.batch_size or i == len(files) - 1:
                    # Pad
                    max_len = max([w.shape[0] for w in batch])
                    padded = torch.zeros(len(batch), max_len).to(device)
                    for j, w in enumerate(batch):
                        padded[j, :w.shape[0]] = w.to(device)
                        
                    # Forward
                    features = model(padded) # (B, T, 768)
                    
                    # Save
                    for j, feat in enumerate(features):
                        orig_len = batch[j].shape[0]
                        out_name = Path(batch_paths[j]).stem + ".pt"
                        
                        # Save
                        torch.save(feat.cpu().clone(), os.path.join(args.output_dir, out_name))
                        
                    batch = []
                    batch_paths = []
                    
            except Exception as e:
                print(f"Error {fpath}: {e}")
                batch = []
                batch_paths = []

if __name__ == "__main__":
    main()
