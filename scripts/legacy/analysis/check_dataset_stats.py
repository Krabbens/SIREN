import os
import torch
import glob
from tqdm import tqdm

def main():
    data_dir = "data/flow_dataset_24k"
    files = glob.glob(os.path.join(data_dir, "*.pt"))
    print(f"Found {len(files)} files")
    
    if len(files) == 0:
        print("No files found!")
        return

    # Sample a subset to save time
    import random
    random.shuffle(files)
    files = files[:500]
    
    all_mels = []
    
    print("Loading data...")
    for f in tqdm(files):
        try:
            d = torch.load(f, map_location='cpu')
            if 'mel' in d:
                all_mels.append(d['mel'])
        except:
            pass
            
    if not all_mels:
        print("No Mel data found in files")
        return

    # Stack
    # mels are likely (1, T, 100) or similar
    all_mels = torch.cat(all_mels, dim=1) # Concat along time
    
    print(f"Shape: {all_mels.shape}")
    print(f"Min: {all_mels.min()}")
    print(f"Max: {all_mels.max()}")
    print(f"Mean: {all_mels.mean()}")
    print(f"Std: {all_mels.std()}")

if __name__ == "__main__":
    main()
