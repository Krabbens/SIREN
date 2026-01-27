import torch
import os
import sys

def print_checkpoint_info(name, path):
    print(f"\n{'='*20} {name} {'='*20}")
    print(f"Path: {path}")
    
    if not os.path.exists(path):
        print("❌ File NOT FOUND")
        return

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"File Size: {size_mb:.2f} MB")

    try:
        state_dict = torch.load(path, map_location='cpu')
        
        # Handle cases where checkpoint might be a dict with 'model_state_dict' or similar
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            print("(Loaded from 'state_dict' key)")
        elif isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
            print("(Loaded from 'model' key)")

        total_params = 0
        print("\n--- Parameter Shapes ---")
        
        # Sort keys for cleaner output
        keys = sorted(state_dict.keys())
        
        # To avoid too much output, we can group or just list them.
        # Let's list everything but truncated if too long? 
        # Actually user wants to see weights, so listing shapes is best.
        
        for k in keys:
            t = state_dict[k]
            if isinstance(t, torch.Tensor):
                shape = tuple(t.shape)
                params = t.numel()
                total_params += params
                print(f"{k}: {shape} | {t.dtype}")
            else:
                print(f"{k}: {type(t)}")

        print(f"\nTotal Parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

def main():
    checkpoints = [
        ("Factorizer", "checkpoints/checkpoints_stable/step_87000/factorizer.pt"),
        ("Semantic RFSQ", "checkpoints/checkpoints_stable/step_87000/sem_rfsq.pt"),
        ("Prosody RFSQ", "checkpoints/checkpoints_stable/step_87000/pro_rfsq.pt"),
        ("Speaker PQ", "checkpoints/checkpoints_stable/step_87000/spk_pq.pt"),
        ("Flow Model", "checks/flow_dryrun/flow_epoch6.pt"),
        ("Condition Fuser", "checks/flow_dryrun/fuser_epoch6.pt"),
        ("Flow Model (Epoch 13)", "checks/flow_dryrun/flow_epoch13.pt"),
        ("Condition Fuser (Epoch 13)", "checks/flow_dryrun/fuser_epoch13.pt"),
        ("BitVocoder", "checkpoints/checkpoints_bitnet_mel_v2/mel_vocoder_epoch82.pt")
    ]

    for name, path in checkpoints:
        print_checkpoint_info(name, path)

if __name__ == "__main__":
    main()
