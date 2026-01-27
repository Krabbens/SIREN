
import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_flow_matching import ConditionFuser

def plot_heatmap(data, title, path):
    plt.figure(figsize=(10, 4))
    plt.imshow(data.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mock Inputs
    # 50Hz semantic tokens -> 2 sec audio -> 100 frames
    # Target 24kHz Mel -> 2 sec -> 188 frames (approx 2x upsample)
    
    T_src = 100
    T_tgt = 188
    D = 8 # Sem Dim
    
    # Create "Step" data to simulate quantized codes
    # [0, 0, 1, 1, 0, 0...]
    s = torch.zeros(1, T_src, D).to(device)
    for i in range(T_src):
        if (i // 10) % 2 == 0:
            s[:, i, :] = 1.0
            
    p = torch.zeros(1, T_src, D).to(device)
    spk = torch.randn(1, 256).to(device)
    
    # Initialize Fuser
    fuser = ConditionFuser(D, D, 256, 512).to(device).eval()
    
    # 1. Standard Linear
    with torch.no_grad():
        cond_linear = fuser(s, p, spk, T_tgt)
        
    plot_heatmap(cond_linear[0].cpu().numpy(), "Conditioning (Linear Interp)", "debug_fuser_linear.png")
    
    # 2. Bicubic attempt
    # Torch interpolate requires (B, C, L)
    def smooth_upsample(x, size, mode='linear'):
        x = x.transpose(1, 2) # (B, D, T)
        if mode == 'bicubic':
             # bicubic is 2D only usually.
             # Fake 2D: (B, C, 1, W)
             x = x.unsqueeze(2)
             x = F.interpolate(x, size=(1, size), mode='bicubic', align_corners=False)
             x = x.squeeze(2)
        else:
             x = F.interpolate(x, size=size, mode=mode, align_corners=False)
        return x.transpose(1, 2)

    # Monkey patch fuser (or simulate)
    # We can just run the logic manually since we know it
    s_cubic = smooth_upsample(s, T_tgt, 'bicubic')
    p_cubic = smooth_upsample(p, T_tgt, 'bicubic')
    spk_exp = spk.unsqueeze(1).expand(-1, T_tgt, -1)
    cat = torch.cat([s_cubic, p_cubic, spk_exp], dim=-1)
    cond_cubic = fuser.proj(cat)
    
    plot_heatmap(cond_cubic[0].detach().cpu().numpy(), "Conditioning (Cubic Interp)", "debug_fuser_cubic.png")
    
    # 3. Gaussian Blur
    # Apply 1D conv smoothing after linear upsample
    def gaussian_blur(x, kernel_size=5, sigma=1.0):
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        # Create kernel
        k = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        k = torch.exp(-0.5 * (k / sigma)**2)
        k = k / k.sum()
        k = k.view(1, 1, -1).to(x.device)
        k = k.expand(x.shape[1], 1, -1) # Depthwise
        
        pad = kernel_size // 2
        x_blur = F.conv1d(x, k, padding=pad, groups=x.shape[1])
        return x_blur.transpose(1, 2)
        
    cond_blur = gaussian_blur(cond_linear, kernel_size=11, sigma=2.0)
    plot_heatmap(cond_blur[0].cpu().numpy(), "Conditioning (Gaussian Blur)", "debug_fuser_blur.png")
    
    print("Saved visualizations to debug_fuser_*.png")

if __name__ == "__main__":
    main()
