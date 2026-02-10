
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss, SnakeBetaDiversityLoss

def save_plot(data, path, title="Plot"):
    plt.figure(figsize=(10, 4))
    if data.ndim == 2:
        plt.imshow(data.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
    elif data.ndim == 1:
        plt.plot(data)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load ONE sample
    # We'll generate a dummy sample or load one depending on what's available
    # Let's use `jakubie_16k.wav` resampled to 24k as the SINGLE GT
    audio_path = "data/jakubie_16k.wav"
    if not os.path.exists(audio_path):
        print("Audio not found, generating sine wave")
        sr = 24000
        t = torch.linspace(0, 5, 5*sr)
        audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    else:
        # Use soundfile instead of torchaudio.load to avoid backend issues
        audio_np, sr = sf.read(audio_path)
        audio = torch.from_numpy(audio_np).float()
        if audio.dim() == 1: audio = audio.unsqueeze(0)
        
        if sr != 24000:
            audio = torchaudio.functional.resample(audio, sr, 24000)
        audio = audio.mean(0, keepdim=True)
        # Crop to 2 seconds for speed
        if audio.shape[1] > 24000*2:
             audio = audio[:, :24000*2]
    
    audio = audio.to(device)
    # Normalize audio roughly
    audio = audio / (audio.abs().max() + 1e-6) * 0.95
    
    # Extract Mel (Target Input)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, 
        n_fft=1024, 
        win_length=1024, 
        hop_length=256, 
        n_mels=100,
        power=1.0 # Standard
    ).to(device)
    
    with torch.no_grad():
        mel = mel_transform(audio)
        mel_log = torch.log(mel + 1e-5).transpose(1, 2) # (1, T, 100)
    
    print(f"Audio Shape: {audio.shape}")
    print(f"Mel Shape: {mel_log.shape}")
    
    # 2. Setup Model
    model = BitVocoder(
        input_dim=100, 
        dim=512, # Large enough
        num_layers=4,
        num_res_blocks=1,
        hop_length=256 
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Lower LR for stability
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    
    # 3. Overfit Loop
    print("Starting Overfit Loop...")
    losses = []
    
    for i in tqdm(range(500)):
        optimizer.zero_grad()
        
        # Monitor Weights every 100 steps
        if i % 100 == 0:
            w = model.conv_in.weight
            print(f"\n[Step {i}] ConvIn W Mean: {w.mean().item():.5f} | Std: {w.std().item():.5f} | Grad: {w.grad.norm().item() if w.grad is not None else 'None'}")
        
        # Forward
        audio_pred = model(mel_log)
        
        # Trim to match length
        min_len = min(audio.shape[1], audio_pred.shape[1])
        a_gt = audio[:, :min_len]
        a_pred = audio_pred[:, :min_len] # (1, T)
        
        # Loss
        # STFT loss expects (B, T)
        sc, mag = stft_loss_fn(a_pred, a_gt)
        loss = sc + mag
        
        loss.backward()
        
        # Check Gradient Norm
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        losses.append(loss.item())
        
        if i % 50 == 0:
            print(f"[{i}] Loss: {loss.item():.4f} | Audio Pred Std: {a_pred.std().item():.4f} | Grad Norm: {total_norm.item():.4f}")
            
    # 4. Save Results
    os.makedirs("debug_overfit", exist_ok=True)
    save_plot(torch.tensor(losses).numpy(), "debug_overfit/loss_curve.png", "Loss Curve")
    
    sf.write("debug_overfit/overfit_target.wav", a_gt.cpu().squeeze().numpy(), 24000)
    sf.write("debug_overfit/overfit_pred.wav", a_pred.detach().cpu().squeeze().numpy(), 24000)
    
    save_plot(a_gt.cpu().squeeze().numpy(), "debug_overfit/waveform_gt.png", "GT Waveform")
    save_plot(a_pred.detach().cpu().squeeze().numpy(), "debug_overfit/waveform_pred.png", "Pred Waveform")
    print("Saved overfit results to debug_overfit/")

if __name__ == "__main__":
    main()
