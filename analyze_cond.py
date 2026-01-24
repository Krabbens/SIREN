import torch

def analyze_cond():
    try:
        cond = torch.load("debug_cond.pt", map_location='cpu')
        print(f"Cond shape: {cond.shape}")
        print(f"Min: {cond.min():.4f}")
        print(f"Max: {cond.max():.4f}")
        print(f"Mean: {cond.mean():.4f}")
        print(f"Std: {cond.std():.4f}")
        
        # Check for constant values
        if cond.std() < 1e-4:
            print("WARNING: Cond has near-zero variance (constant)!")
            
        # Check for zeros
        if (cond == 0).all():
            print("WARNING: Cond is all zeros!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_cond()
