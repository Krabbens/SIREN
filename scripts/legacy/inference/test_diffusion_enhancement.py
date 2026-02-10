import torch
import torchaudio
import soundfile as sf
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.tiny_diffusion import TinyDiffusionEnhancer

def save_spectrogram(spec, path, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.cpu().numpy(), aspect='auto', origin='lower', vmin=-11.5, vmax=2.0)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input audio file (BitNet output)")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoints_diffusion/best_model.pt")
    parser.add_argument("--output_dir", default="diffusion_results")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--strength", type=float, default=0.5)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = TinyDiffusionEnhancer(n_mels=80, hidden_dim=64).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    
    # Load Audio
    print(f"Processing {args.input}...")
    audio_np, sr = sf.read(args.input)
    # Convert to tensor (1, T)
    if audio_np.ndim == 1:
        audio = torch.from_numpy(audio_np).unsqueeze(0).float()
    else:
        audio = torch.from_numpy(audio_np).t().float()
        audio = audio.mean(0, keepdim=True)
        
    if sr != 16000:
        print(f"Resampling from {sr} to 16000")
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Mel Transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80, power=1.0
    ).to(device)
    
    with torch.no_grad():
        # Get Mel
        mel = mel_transform(audio.to(device))
        log_mel = torch.log(mel + 1e-5)
        
        # Enhance
        # Mode 1: Sample (Iterative)
        print("Running Iterative Enhancement (Sampling)...")
        enhanced_mel = model.sample(log_mel, num_steps=args.steps)
        
        # Mode 2: Simple (Single Step)
        # print("Running Simple Enhancement...")
        # enhanced_mel_simple = model.enhance_simple(log_mel, strength=args.strength)
        
        # Reconstruct Audio (Griffin-Lim)
        print("Reconstructing Audio (InverseMel + Griffin-Lim)...")
        
        # 1. Mel -> Linear
        inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=1024 // 2 + 1, n_mels=80, sample_rate=16000
        ).to(device)
        
        # 2. Linear -> Audio
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=1024, hop_length=320, power=1.0, n_iter=32
        ).to(device)
        
        # Denormalize log mel
        lin_mel = torch.exp(enhanced_mel)
        
        # Convert to linear spec
        linear_spec = inv_mel(lin_mel)
        
        # Griffin-Lim
        enhanced_audio = griffin_lim(linear_spec)
        
        # Save results
        basename = os.path.splitext(os.path.basename(args.input))[0]
        out_wav = os.path.join(args.output_dir, f"{basename}_enhanced_step{args.steps}.wav")
        out_spec_orig = os.path.join(args.output_dir, f"{basename}_original_spec.png")
        out_spec_enh = os.path.join(args.output_dir, f"{basename}_enhanced_spec.png")
        
        sf.write(out_wav, enhanced_audio.cpu().squeeze().numpy(), 16000)
        
        save_spectrogram(log_mel.squeeze(), out_spec_orig, "Original Input Mel")
        save_spectrogram(enhanced_mel.squeeze(), out_spec_enh, "Enhanced Mel")
        
        print(f"Saved: {out_wav}")

        # Calculate Mel Loss difference
        loss_orig = torch.nn.functional.l1_loss(log_mel, log_mel).item() # 0
        # Wait, we want loss vs something? 
        # Ideally we compare to ground truth, but we don't have it here easily (unless input was ground truth).
        # We just want to see if it CHANGED.
        diff = torch.nn.functional.mse_loss(log_mel, enhanced_mel).item()
        print(f"Change (MSE): {diff:.4f}")

if __name__ == "__main__":
    main()
