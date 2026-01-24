import os
import torch
import glob
import argparse
from tqdm import tqdm
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/flow_dataset")
    parser.add_argument("--check_n", type=int, default=1000)
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.data_dir, "*.pt"))
    print(f"Found {len(files)} files")
    
    random.shuffle(files)
    files = files[:args.check_n]
    
    valid_count = 0
    zeros_count = 0
    error_count = 0
    
    print(f"Checking {len(files)} samples...")
    
    for f in tqdm(files):
        try:
            data = torch.load(f, map_location='cpu')
            target = data['target']
            cond = data['cond']
            
            # Check shapes
            if target.shape[0] != 1026:
                print(f"Spread shape: {target.shape} in {f}")
                error_count += 1
                continue
                
            if cond.shape[0] != 512:
                print(f"Bad cond shape: {cond.shape} in {f}")
                error_count += 1
                continue
                
            # Check content
            if torch.all(target == 0):
                print(f"ZEROS TARGET: {f}")
                zeros_count += 1
                continue
                
            if torch.all(cond == 0):
                print(f"ZEROS COND: {f}")
                zeros_count += 1
                continue
                
            # Check stats
            if torch.isnan(target).any():
                print(f"NAN TARGET: {f}")
                error_count += 1
                continue
                
            valid_count += 1
            
        except Exception as e:
            print(f"Error loading {f}: {e}")
            error_count += 1
            
    print("\nResults:")
    print(f"Valid: {valid_count}")
    print(f"Zeros: {zeros_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
