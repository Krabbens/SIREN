
import torch
import sys
import os

def main():
    ckpt_path = "checkpoints/checkpoints_bitnet_mel_v2/mel_vocoder_epoch73.pt"
    if not os.path.exists(ckpt_path):
        print(f"File not found: {ckpt_path}")
        return

    print(f"Inspecting {ckpt_path}...")
    try:
        d = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print(f"Keys: {list(d.keys())[:5]} ...")
    
    # Check for state_dict or if d IS the state_dict
    if 'state_dict' in d:
        sd = d['state_dict']
    else:
        sd = d
        
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                print(f"[!] {k} has NaNs!")
            if torch.isinf(v).any():
                print(f"[!] {k} has Infs!")
            
            v_abs = v.abs()
            mean = v.mean().item()
            std = v.std().item()
            min_val = v.min().item()
            max_val = v.max().item()
            
            # Check for dead weights
            if v_abs.max() < 1e-6:
                print(f"[!] {k} is effectively ZERO (max abs < 1e-6)")
            else:
                pass
                # print(f"{k}: Mean={mean:.4f} Std={std:.4f} Range=[{min_val:.4f}, {max_val:.4f}]")
                
    # Look specifically at a BitLinear layer
    # model.mag_head.0.weight
    specific_keys = ["mag_head.0.weight", "conv_in.weight", "backbone.0.dwconv.weight"]
    for k in specific_keys:
        if k in sd:
            v = sd[k]
            print(f"\n--- {k} Stats ---")
            print(f"Mean: {v.mean()}")
            print(f"Std: {v.std()}")
            print(f"Min: {v.min()}")
            print(f"Max: {v.max()}")
            
            # Simulate quantization stats
            scale = v.abs().mean()
            w_norm = v / (scale + 1e-9)
            w_quant = torch.round(w_norm.clamp(-1, 1))
            zeros = (w_quant == 0).sum().item()
            total = w_quant.numel()
            print(f"Ternary Zero Ratio: {zeros/total:.2%}")

if __name__ == "__main__":
    main()
