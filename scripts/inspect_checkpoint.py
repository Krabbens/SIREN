import torch
import sys
import os

def inspect(path):
    print(f"Inspecting {path}...")
    try:
        ckpt = torch.load(path, map_location='cpu')
        if 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        
        print(f"Total keys: {len(ckpt)}")
        
        # Group by module to deduce structure
        shapes = {}
        for k, v in ckpt.items():
            k = k.replace("_orig_mod.", "")
            if 'weight' in k or 'bias' in k:
               shapes[k] = list(v.shape)
        
        # Print Key Layers to deduce dims
        relevant_keys = [
            "reconstructor.sem_upsampler.upsample.weight",
            "reconstructor.pro_upsampler.upsample.weight",
            "reconstructor.spk_proj.0.weight",
            "reconstructor.fusion_proj.weight",
            "reconstructor.cross_fusion.norm_sem.weight",
            "reconstructor.cross_fusion.norm_pro.weight",
            "reconstructor.cross_fusion.cross_attn.in_proj_weight",
            "vocoder.model.conv_in.weight",
            "vocoder.model.phase_head.cumsum_weight"
        ]
        
        for k in relevant_keys:
            if k in shapes:
                print(f"{k}: {shapes[k]}")
            else:
                print(f"{k}: NOT FOUND")
                
        # Also print all keys in reconstructor.fusion_proj to see if name matches
        print("\nFusion Proj Keys:")
        for k in shapes:
            if "fusion_proj" in k:
                print(f"{k}: {shapes[k]}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        inspect("checkpoints/checkpoints_stable/step_87000/decoder.pt")
