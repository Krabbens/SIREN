import torch
import glob
import os
from tqdm import tqdm

def main():
    files = glob.glob("data/flow_dataset_24k/*.pt")
    print(f"Checking {len(files)} files...")
    counts = {}
    for f in tqdm(files):
        try:
            d = torch.load(f, map_location='cpu')
            shape = d['mel'].shape
            counts[shape] = counts.get(shape, 0) + 1
            if shape[-1] != 100:
                print(f"BAD FILE: {f} Shape: {shape}")
        except Exception as e:
            print(f"ERROR {f}: {e}")
    
    print("\nSummary of shapes found:")
    for s, c in counts.items():
        print(f"{s}: {c}")

if __name__ == "__main__":
    main()
