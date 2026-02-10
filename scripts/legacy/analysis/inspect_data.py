import torch
import glob
import os
import sys

def check_dir(d):
    print(f"--- Checking {d} ---")
    files = glob.glob(os.path.join(d, "*.pt"))
    if not files:
        print("No .pt files found.")
        return

    print(f"Found {len(files)} files.")
    f = files[0]
    try:
        data = torch.load(f, map_location='cpu')
        print(f"Keys: {data.keys()}")
        if 'mel' in data:
            mel = data['mel']
            print(f"Mel Shape: {mel.shape}")
            print(f"Mel Min: {mel.min()}, Max: {mel.max()}")
            print(f"Mel Mean: {mel.mean()}, Std: {mel.std()}")
            
            # Check for infinites
            if torch.isinf(mel).any():
                print("WARNING: Mel contains Inf!")
            if torch.isnan(mel).any():
                print("WARNING: Mel contains NaN!")
                
        if 'sem' in data:
            print(f"Sem Shape: {data['sem'].shape}")
    except Exception as e:
        print(f"Error loading {f}: {e}")

check_dir("data/flow_dataset")
check_dir("data/flow_dataset_24k")
