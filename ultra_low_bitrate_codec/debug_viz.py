import torch
import matplotlib.pyplot as plt
import glob
import os

def debug_viz():
    files = glob.glob("data/flow_dataset/*.pt")
    if not files:
        print("No files found!")
        return

    f = files[0]
    data = torch.load(f, map_location='cpu')
    target = data['target'] # (1026, T)
    
    print(f"Loaded {f}, shape: {target.shape}")
    print(f"Min: {target.min()}, Max: {target.max()}, Mean: {target.mean()}")
    
    # Split into Real/Imag
    # (1026, T) -> (2, 513, T)
    real = target[:513, :]
    imag = target[513:, :]
    
    # Compute Magnitude
    mag = torch.sqrt(real**2 + imag**2 + 1e-6)
    
    # Log Magnitude
    log_mag = torch.log10(mag)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(log_mag.numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.title("Log-Magnitude Spectrogram (Reconstructed from Complex)")
    plt.colorbar()
    plt.savefig("debug_spectrogram_magnitude.png")
    print("Saved debug_spectrogram_magnitude.png")

    # Also save raw real channel 0 just to compare
    plt.figure(figsize=(10, 4))
    plt.imshow(real.numpy(), aspect='auto', origin='lower', cmap='viridis', vmin=-1, vmax=1)
    plt.title("Raw Real Component (What user saw)")
    plt.savefig("debug_spectrogram_raw.png")

if __name__ == "__main__":
    debug_viz()
