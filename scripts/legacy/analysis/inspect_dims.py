import torch
import sys

def inspect(path):
    print(f"Inspecting {path}...")
    try:
        ckpt = torch.load(path, map_location='cpu')
        
        # Factorizer
        if 'semantic_proj.3.weight' in ckpt:
            print(f"semantic_proj.3.weight: {ckpt['semantic_proj.3.weight'].shape}")
        
        if 'prosody_proj.3.weight' in ckpt:
            print(f"prosody_proj.3.weight: {ckpt['prosody_proj.3.weight'].shape}")
            
        # Check FSQ if possible (might be in separate file)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect("checkpoints/checkpoints_stable/step_87000/factorizer.pt")
