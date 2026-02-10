
import torch
import sys

def main():
    path = "checkpoints/checkpoints_bitnet_mel_v3/mel_vocoder_epoch3.pt"
    try:
        sd = torch.load(path, map_location='cpu')
        print(f"Loaded {path}")
        if isinstance(sd, dict):
            print(f"Keys: {list(sd.keys())[:5]}")
            if 'model_state_dict' in sd:
                keys = list(sd['model_state_dict'].keys())
                print(f"Model Keys: {keys[:5]} ...")
            elif 'input_conv.weight' in sd:
                print("Found input_conv.weight at root")
            else:
                # keys of the state dict itself
                print(f"State Dict Keys: {list(sd.keys())[:5]} ...")
        else:
            print("Loaded object is not a dict")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
