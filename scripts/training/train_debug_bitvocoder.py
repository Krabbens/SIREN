import torch
import torch.nn.functional as F
import yaml
import soundfile as sf
import os
import sys
import matplotlib.pyplot as plt
from torch.optim import AdamW

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.models.decoder import FeatureReconstructorV2

def stft_loss(y_hat, y):
    """Simple functional STFT loss for debugging."""
    # Magnitudes
    y_mag = torch.abs(torch.stft(y.squeeze(), n_fft=1024, hop_length=320, win_length=1024, return_complex=True))
    y_hat_mag = torch.abs(torch.stft(y_hat.squeeze(), n_fft=1024, hop_length=320, win_length=1024, return_complex=True))
    
    sc_loss = torch.norm(y_mag - y_hat_mag, p="fro") / torch.norm(y_mag, p="fro")
    mag_loss = F.l1_loss(torch.log(y_mag + 1e-5), torch.log(y_hat_mag + 1e-5))
    
    return sc_loss + mag_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # Config
    config_path = "ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Model
    print("Initializing BitVocoder...")
    vocoder = BitVocoder(
        input_dim=config['model']['decoder']['fusion_dim'],
        dim=256,
        n_fft=1024,
        hop_length=320,
        num_layers=4,
        num_res_blocks=1
    ).to(device)
    
    optimizer = AdamW(vocoder.parameters(), lr=2e-4, betas=(0.8, 0.99))
    
    # Dummy Features (simulate FeatureReconstructor output)
    # Target shape: (B, T, D)
    B, T, D = 1, 200, config['model']['decoder']['fusion_dim']
    dummy_feats = torch.randn(B, T, D).to(device)
    dummy_audio_len = (T - 1) * 320
    dummy_audio = torch.randn(B, dummy_audio_len).to(device) # Random noise target for now, just to see loss drop
    
    print("Starting Overfit Test (Random Target)...")
    losses = []
    
    vocoder.train()
    for i in range(100):
        optimizer.zero_grad()
        
        audio_hat = vocoder(dummy_feats)
        
        # Crop to min length
        min_len = min(audio_hat.shape[-1], dummy_audio.shape[-1])
        audio_hat = audio_hat[..., :min_len]
        target = dummy_audio[..., :min_len]
        
        loss = stft_loss(audio_hat, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if i % 10 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}")
            
    print(f"Final Loss: {losses[-1]:.4f}")
    
    # Check if loss dropped meaningfully
    if losses[0] - losses[-1] > 0.5:
        print("SUCCESS: Model can learn (loss dropped).")
    else:
        print("FAILURE: Model is not learning.")
        
    plt.plot(losses)
    plt.title("BitVocoder Overfit Curve")
    plt.savefig("debug_overfit.png")
    print("Saved debug_overfit.png")

if __name__ == "__main__":
    main()
