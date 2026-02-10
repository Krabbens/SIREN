
import os
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import yaml

# Imports
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoderTiny
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input wav file")
    parser.add_argument("--ckpt_dir", default="checkpoints/microencoder_aug_siamese", help="Checkpoint directory")
    parser.add_argument("--config", default="src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml", help="Config file")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    print(f"Loading checkpoints from {args.ckpt_dir}...")
    
    # Auto-detect latest
    import glob
    import re
    files = glob.glob(os.path.join(args.ckpt_dir, "encoder_ep*.pt"))
    if not files: raise FileNotFoundError("No checkpoints found")
    epochs = [int(re.search(r"encoder_ep(\d+).pt", f).group(1)) for f in files if re.search(r"encoder_ep(\d+).pt", f)]
    latest = max(epochs)
    tag = f"ep{latest}"
    print(f"Using Epoch {latest}")
    
    # Load Models
    encoder = MicroEncoderTiny(hidden_dim=128, output_dim=768, num_layers=2).to(device)
    encoder.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"encoder_{tag}.pt"), map_location=device))
    
    factorizer = InformationFactorizerV2(config).to(device)
    factorizer.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"factorizer_{tag}.pt"), map_location=device))
    
    # Process Audio
    wav, sr = sf.read(args.input)
    wav = torch.tensor(wav, dtype=torch.float32)
    if sr != 16000:
        import torchaudio.functional as F_audio
        wav = F_audio.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    wav = wav / (wav.abs().max() + 1e-6)
    wav = wav.unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat = encoder(wav)
        s, p, spk = factorizer(feat)
        
    # Stats
    s_np = s.squeeze(0).cpu().numpy() # (T_s, D)
    p_np = p.squeeze(0).cpu().numpy() # (T_p, D)
    
    # Variance per dimension
    s_var = np.var(s_np, axis=0)
    p_var = np.var(p_np, axis=0)
    
    print(f"Semantic Variance (Mean): {np.mean(s_var):.4f}")
    print(f"Prosody Variance (Mean): {np.mean(p_var):.4f}")
    
    plt.figure(figsize=(12, 10))
    
    # 1. Semantic Features
    plt.subplot(3, 1, 1)
    plt.imshow(s_np.T, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Semantic Latents (Mean Var: {np.mean(s_var):.4f})")
    plt.colorbar()
    
    # 2. Prosody Features
    plt.subplot(3, 1, 2)
    plt.imshow(p_np.T, aspect='auto', origin='lower', cmap='plasma')
    plt.title(f"Prosody Latents (Mean Var: {np.mean(p_var):.4f})")
    plt.colorbar()
    
    # 3. Waveform
    plt.subplot(3, 1, 3)
    plt.plot(wav.squeeze().cpu().numpy())
    plt.title("Waveform")
    plt.xlim(0, len(wav.squeeze()))
    
    plt.tight_layout()
    plt.savefig("outputs/latent_analysis.png")
    print("Saved outputs/latent_analysis.png")

if __name__ == "__main__":
    main()
