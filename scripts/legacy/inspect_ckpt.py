
import torch
import sys

def inspect(path):
    try:
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                keys = list(ckpt['model_state_dict'].keys())
                for k in keys:
                    print(k)
            else:
                keys = list(ckpt.keys())
                for k in keys:
                    print(k)
        else:
            print(f"Checkpoint is not a dict, but {type(ckpt)}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
