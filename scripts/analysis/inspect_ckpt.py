import torch
import sys

def main():
    path = "checkpoints/checkpoints_ultra200bps_gan/bitvocoder_epoch2.pt"
    try:
        sd = torch.load(path, map_location='cpu')
        print("Keys in checkpoint (first 20):")
        for k in list(sd.keys())[:20]:
            print(k)
            
        print("\nKeys related to vocoder/decoder:")
        for k in sd.keys():
            if "vocoder" in k or "decoder" in k:
                print(k)
                if len(k) > 100: break # Just a few
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
