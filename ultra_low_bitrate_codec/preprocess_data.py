#!/usr/bin/env python3
"""
Precompute HuBERT features for LibriTTS or any multi-speaker dataset.
OPTIMIZED VERSION: Uses batched inference, parallel loading, and mixed precision.
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
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Disable tokenizer warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class AudioDataset(Dataset):
    def __init__(self, file_path_list, output_dir):
        self.files = file_path_list
        self.output_dir = Path(output_dir)
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000) # cached, but freq might vary
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        try:
            # We don't know the SR ahead of time, so we load and check
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != 16000:
                # Use functional resample for simpler handling of varying input SR
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            
            # Mix to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.squeeze() # (T,)
            
            # Calculate output path
            basename = Path(audio_path).stem
            output_file = self.output_dir / f"{basename}.pt"
            
            return {
                "waveform": waveform,
                "output_file": str(output_file),
                "valid": True
            }
        except Exception as e:
            print(f"Error reading {audio_path}: {e}")
            return {"valid": False}

def collate_fn(batch):
    # Filter invalid
    batch = [b for b in batch if b["valid"]]
    if not batch:
        return None
    
    waveforms = [b["waveform"].numpy() for b in batch]
    output_files = [b["output_file"] for b in batch]
    
    return waveforms, output_files

def precompute_features(
    input_path: str,
    output_dir: str,
    hubert_layer: int = 9,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Gather files
    input_p = Path(input_path)
    audio_files = []
    
    if input_p.is_file() and input_p.suffix == '.json':
        print(f"📄 Loading manifest: {input_path}")
        with open(input_path, 'r') as f:
            data = json.load(f)
            # Support both straight paths list and dicts with 'audio_path'
            for item in data:
                if isinstance(item, dict):
                    audio_files.append(item.get('audio_path', item.get('path')))
                else:
                    audio_files.append(item)
    elif input_p.is_dir():
        print(f"📂 Scanning directory: {input_path}")
        audio_files = [str(p) for p in list(input_p.rglob("*.wav"))]
        print(f"   Found {len(audio_files)} .wav files")
    else:
        print(f"❌ Input path not found or invalid: {input_path}")
        return

    # Filter out existing (optional, but good)
    # To do this efficiently, we check existence.
    # But since we use simple filename mapping, we can check.
    # For now, let's skip this check in the main gathering to save time if directory is huge,
    # OR do it if requested. Let's rely on the dataset length, maybe check existence in Dataset?
    # No, better to filter list first to get accurate progress bar.
    
    print("🔍 Checking existing files...")
    files_to_process = []
    existing_count = 0
    for f in audio_files:
        basename = Path(f).stem
        out_f = output_path / f"{basename}.pt"
        if not out_f.exists():
            files_to_process.append(f)
        else:
            existing_count += 1
            
    print(f"   Skipping {existing_count} existing files.")
    print(f"   Processing {len(files_to_process)} files.")
    
    if not files_to_process:
        print("✅ Nothing to do!")
        return

    # 2. Setup Loader
    ds = AudioDataset(files_to_process, output_path)
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 3. Load Model
    print(f"🔧 Loading HuBERT model on {device}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    model.to(device)
    model.eval()
    
    # Compile for speedup if available (PyTorch 2.0+)
    # Note: Compilation might take a minute at first run but worth it for large datasets
    try:
        model = torch.compile(model)
        print("🚀 Model compiled with torch.compile")
    except Exception as e:
        print(f"⚠️ Could not compile model: {e}")

    print(f"🚀 Starting extraction (Batch size: {batch_size}, Workers: {num_workers})")
    
    processed = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            if batch is None: continue
            
            waveforms, out_paths = batch
            
            # Process with transformers
            # This handles padding and creates attention masks
            inputs = processor(
                waveforms, 
                sampling_rate=16000, 
                padding=True, 
                return_tensors="pt"
            )
            
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device):
                outputs = model(input_values, attention_mask=attention_mask, output_hidden_states=True)
                # (B, T, D)
                features = outputs.hidden_states[hubert_layer]
            
            # Post-process and save
            features = features.cpu() # Move to CPU once
            
            # We need to unpad. 
            # The model output length is different from input length due to striding (320x).
            # The attention_mask is for input. transformers doesn't return output mask.
            # But the output features correspond to the input duration.
            # We can calculate valid output length based on input attention mask.
            # Formula: L_out = floor((L_in - 1) / 320) + 1  roughly, simpler is sum(mask) / 320
            
            # Precise way: use the attention mask to find the last 1.
            # Or just use the processor's capability to return valid lengths? No.
            
            input_lengths = attention_mask.sum(dim=1).cpu().long()
            
            for i, feat in enumerate(features):
                # Calculate valid length
                # HuBERT/wav2vec2 stride is 320, receptive field (kernel) is 400
                # conv formula: out = (in - kernel) / stride + 1
                # = (in - 400) / 320 + 1
                # correct way used in w2v2:
                
                valid_in_len = input_lengths[i].item()
                valid_out_len = int((valid_in_len - 400) / 320) + 1
                if valid_out_len < 1: valid_out_len = 1
                
                # Slice
                # Ensure we don't go out of bounds if formula is slightly off for edge cases
                valid_out_len = min(valid_out_len, feat.shape[0])
                
                feat_sliced = feat[:valid_out_len]
                
                torch.save(feat_sliced.clone(), out_paths[i])
                processed += 1

    print(f"\n✅ Extraction complete!")
    print(f"   Processed: {processed} files")


def main():
    parser = argparse.ArgumentParser(description="Precompute HuBERT features (Optimized)")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to directory containing .wav files OR JSON manifest")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save features")
    parser.add_argument("--hubert-layer", type=int, default=9,
                       help="HuBERT layer to extract (9)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size (default: 32)")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count()), 
                       help="Data loader workers (default: auto)")
    
    args = parser.parse_args()
    
    precompute_features(
        input_path=args.input,
        output_dir=args.output_dir,
        hubert_layer=args.hubert_layer,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
