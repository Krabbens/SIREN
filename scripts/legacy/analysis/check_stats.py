import torch
import glob
import os
from tqdm import tqdm

def check_stats(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.pt"))
    print(f"Checking statistics for {len(files)} files...")
    
    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []
    
    # Check random subset of 500 files
    import random
    random.shuffle(files)
    files = files[:500]
    
    for f in tqdm(files):
        try:
            data = torch.load(f, map_location='cpu')
            target = data['target'] # (1026, T)
            
            all_means.append(target.mean().item())
            all_stds.append(target.std().item())
            all_mins.append(target.min().item())
            all_maxs.append(target.max().item())
        except:
            pass
            
    import numpy as np
    print(f"\nGlobal Stats (N={len(files)}):")
    print(f"Mean: {np.mean(all_means):.4f}")
    print(f"Std:  {np.mean(all_stds):.4f}")
    print(f"Min:  {np.min(all_mins):.4f}")
    print(f"Max:  {np.max(all_maxs):.4f}")
    
    # Check percentile to see if min/max are outliers
    print("\nRange Analysis:")
    print(f"Avg Min: {np.mean(all_mins):.4f}")
    print(f"Avg Max: {np.mean(all_maxs):.4f}")

if __name__ == "__main__":
    check_stats("data/flow_dataset")
