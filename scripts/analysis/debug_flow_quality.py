import torch
import matplotlib.pyplot as plt
import glob
import os
import sys
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from train_flow_matching import ConditionFuser

def save_spectrogram(data, path, title="Spectrogram"):
    viz = data.T.cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(viz, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Inspect Data
    files = glob.glob("data/flow_dataset_24k/*.pt")
    if not files:
        print("No files found in data/flow_dataset_24k")
        return
    
    # Pick a random file
    fpath = files[0]
    print(f"Inspecting {fpath}...")
    data = torch.load(fpath, map_location=device)
    mel = data['mel']
    print(f"Mel stats: Min={mel.min():.2f}, Max={mel.max():.2f}, Mean={mel.mean():.2f}, Std={mel.std():.2f}")
    
    # Save GT
    save_spectrogram(mel.squeeze(), "debug_gt_mel.png", "Ground Truth Mel (Raw)")

    # 2. Inspect Checkpoint
    ckpt_path = "checkpoints/checkpoints_flow_v6/flow_epoch4.pt" # Adjust if user has different one
    # Find latest specific epoch if needed, or specific path
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        # Try finding latest in checkpoints/checkpoints_flow_new
        ckpts = glob.glob("checkpoints/checkpoints_flow_new/*.pt")
        if ckpts:
            ckpt_path = max(ckpts, key=os.path.getmtime)
            print(f"Using latest checkpoint: {ckpt_path}")
        else:
            print("No checkpoints found.")
            return

    # Load Model (Minimal Config)
    config = {
        'model': {
            'decoder': {
                'fusion_dim': 100,
                'fusion_heads': 8,
                'dropout': 0.1
            },
            'flow_matching_layers': 8
        }
    }
    
    model = ConditionalFlowMatching(config).to(device).eval()
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Quick Inference on this sample
    # Need cond
    sem = data['sem'].to(device).float().unsqueeze(0)
    pro = data['pro'].to(device).float().unsqueeze(0)
    spk = data['spk'].to(device).float().unsqueeze(0)
    target_len = mel.shape[1]
    
    # Load Fuser
    fuser = ConditionFuser(8, 8, 256, 512).to(device).eval()
    # Try finding fuser ckpt
    fuser_ckpt = ckpt_path.replace("flow_", "fuser_")
    if os.path.exists(fuser_ckpt):
         fuser.load_state_dict(torch.load(fuser_ckpt, map_location=device))
         print(f"Fuser loaded from {fuser_ckpt}")
    else:
        print("Fuser checkpoint not found! Using random fuser (Bad results expected).")

    with torch.no_grad():
        if sem.dim() == 4: sem = sem.squeeze(1)
        if pro.dim() == 4: pro = pro.squeeze(1)
        if spk.dim() == 3: spk = spk.squeeze(1)
        
        cond = fuser(sem, pro, spk, target_len)
        
        # ODE Solve
        # Normalize/Denormalize manual check
        # Prediction uses: x_t_v * MEL_STD + MEL_MEAN
        # Where MEL_MEAN = -5.0, MEL_STD = 3.5
        
        mel_pred_normalized = model.solve_ode(cond, steps=50) 
        mel_pred = mel_pred_normalized * 3.5 - 5.0
        
        save_spectrogram(mel_pred[0], "debug_pred_mel.png", "Predicted Mel")

    print("Done.")

if __name__ == "__main__":
    main()
