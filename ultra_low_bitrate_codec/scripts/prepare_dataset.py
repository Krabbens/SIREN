import os
import json
import random
import glob

def prepare_ljspeech(data_dir, output_dir):
    wav_files = glob.glob(os.path.join(data_dir, 'wavs', '*.wav'))
    print(f"Found {len(wav_files)} wav files in {data_dir}")
    
    random.shuffle(wav_files)
    
    # 95/5 split
    split_idx = int(len(wav_files) * 0.95)
    train_files = wav_files[:split_idx]
    val_files = wav_files[split_idx:]
    
    # Format for dataset: list of dicts or just paths?
    # Dataset loader supports specific json: "files = [d['path'] if isinstance(d, dict) else d for d in data]"
    # So list of dicts with 'path' is good.
    
    train_manifest = [{'path': os.path.abspath(f)} for f in train_files]
    val_manifest = [{'path': os.path.abspath(f)} for f in val_files]
    
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_manifest, f, indent=4)
        
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_manifest, f, indent=4)
        
    print(f"Created train.json ({len(train_files)}) and val.json ({len(val_files)})")

if __name__ == '__main__':
    # Assuming data is in ./data/LJSpeech-1.1
    prepare_ljspeech('data/LJSpeech-1.1', 'data')
