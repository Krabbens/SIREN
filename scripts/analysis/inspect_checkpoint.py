import torch
import sys
import os

def inspect_ckpt(path):
    print(f"Inspecting: {path}")
    try:
        data = torch.load(path, map_location='cpu')
        keys = list(data.keys())
        print(f"Total keys: {len(keys)}")
        print("First 20 keys:")
        for k in keys[:20]:
            print(f"  {k}: {data[k].shape}")
            
        # Check for specific patterns
        has_bias = any("bias" in k for k in keys)
        has_norm = any("norm.weight" in k and "linear" not in k for k in keys) # rough check
        print(f"\nHas bias: {has_bias}")
        
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <path>")
    else:
        inspect_ckpt(sys.argv[1])
