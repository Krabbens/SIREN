import torch
ckpt = torch.load('checkpoints/microhubert/microhubert_ep45.pt', map_location='cpu')
print(f"Type: {type(ckpt)}")
if isinstance(ckpt, dict):
    print(f"Keys: {list(ckpt.keys())}")
    if 'state_dict' in ckpt:
        print("Found 'state_dict' key")
    if 'model' in ckpt:
        print("Found 'model' key")
        sd = ckpt['model']
        keys = list(sd.keys())
        print(f"Total keys: {len(keys)}")
        print(f"First 10 keys: {keys[:10]}")
    else:
        print("No 'model' key, assuming root state dict")
        keys = list(ckpt.keys())
        print(f"Total keys: {len(keys)}")
        print(f"First 10 keys: {keys[:10]}")
