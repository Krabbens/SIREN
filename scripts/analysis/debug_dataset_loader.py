
import os
import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_bitnet_mel import MelFeatureDataset

def main():
    feature_dir = "data/flow_dataset_24k"
    audio_dir = "data/audio"
    
    if not os.path.exists(feature_dir) or not os.path.exists(audio_dir):
        print(f"Dirs not found: {feature_dir} {audio_dir}")
        return
        
    dataset = MelFeatureDataset(feature_dir, audio_dir, segment_frames=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Dataset Stats:")
    print(f"Len: {len(dataset)}")
    
    # Check one batch
    features, audio = next(iter(dataloader))
    
    print(f"Features (Mel) Shape: {features.shape}") # Expect (B, 100, T) or (B, T, 100)?
    # train_bitnet_mel.py: features = features.transpose(0, 1) # (100, T) inside dataset?
    # No, dataset returns (100, T)
    # So batch should be (B, 100, T)
    
    print(f"Audio Shape: {audio.shape}")
    
    print(f"Features - Mean: {features.mean():.4f} | Std: {features.std():.4f} | Min: {features.min():.4f} | Max: {features.max():.4f}")
    print(f"Audio    - Mean: {audio.mean():.4f} | Std: {audio.std():.4f} | Min: {audio.min():.4f} | Max: {audio.max():.4f}")
    
    # Check if Audio is silent
    if audio.std() < 0.01:
        print("WARNING: Audio seems silent!")
        
    # Check alignment (visual check)
    # We can't easily check alignment without listening or phase correlation, but valid stats are a good start.

if __name__ == "__main__":
    main()
