
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_alphas(ckpt_path):
    print(f"Loading {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Handle state dict wrapper
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint # Raw state dict
        
    alphas = []
    layer_names = []
    
    # print all keys to debug
    # for k in state_dict.keys():
    #     print(k)

    for k, v in state_dict.items():
        if 'alpha' in k: # Relaxed check
            # print(f"Found alpha in {k}")
            # Flatten and convert to numpy
            alpha_vals = v.float().flatten().numpy()
            alphas.append(alpha_vals)
            layer_names.append(k)
            
    if not alphas:
        print("No SnakeBeta alphas found!")
        return
        
    all_alphas = np.concatenate(alphas)
    
    print(f"Found {len(alphas)} SnakeBeta layers.")
    print(f"Total alpha parameters: {len(all_alphas)}")
    print(f"Range: [{all_alphas.min():.4f}, {all_alphas.max():.4f}]")
    print(f"Mean: {all_alphas.mean():.4f}")
    
    # Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_alphas, bins=100, log=True)
    plt.title(f"SnakeBeta Alpha Distribution\n{os.path.basename(ckpt_path)}")
    plt.xlabel("Alpha (Frequency)")
    plt.ylabel("Count (Log Scale)")
    plt.grid(True, alpha=0.3)
    
    out_path = "snake_alpha_dist.png"
    plt.savefig(out_path)
    print(f"Saved distribution plot to {out_path}")
    
    # Check for clustering (Potential Banding)
    # Bin alphas and see if there are spikes
    hist, bin_edges = np.histogram(all_alphas, bins=200)
    
    # Simple peak detection
    peaks = []
    threshold = hist.mean() + 2 * hist.std()
    for i in range(len(hist)):
        if hist[i] > threshold:
            center = (bin_edges[i] + bin_edges[i+1]) / 2
            peaks.append((center, hist[i]))
            
    if peaks:
        print("\nPotential Frequency Clusters (Banding Sources):")
        # Sort by count desc
        peaks.sort(key=lambda x: x[1], reverse=True)
        for p in peaks[:10]:
            print(f"  Freq ~ {p[0]:.2f}: count {p[1]}")
    else:
        print("\nNo significant clustering detected.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint")
    args = parser.parse_args()
    
    analyze_alphas(args.checkpoint)
