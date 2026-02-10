
import os
import sys
import torch
import torchaudio
import soundfile as sf
import yaml
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder

def save_spectrogram(data, path, title="Spectrogram"):
    viz = data.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(viz, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Config & Checkpoint
    vocoder_ckpt_path = "checkpoints/checkpoints_bitnet_mel_v2/mel_vocoder_epoch82.pt"
    input_wav = "data/jakubie.wav"
    output_wav = "checks/vocoder_isolation_test.wav"
    os.makedirs("checks", exist_ok=True)
    
    # 2. Init Vocoder
    print("Initializing Vocoder...")
    # These parameters MUST match what train_bitnet_mel.py used.
    # Typically input_dim=100 for Mel. dim=512 was used in previous scripts? 
    # train_bitnet_mel.py doesn't show hardcoded args, it implies looking at config or defaults.
    # Let's try to infer from state dict or assume 512 as per run_inference_pipeline.
    
    # Try loading state dict first to check shapes?
    sd = torch.load(vocoder_ckpt_path, map_location=device)
    
    # Inspect first layer weight shape to deduce input dim and hidden dim
    # input_conv.weight shape: (dim, input_dim, kernel)
    first_weight = sd['conv_in.weight']
    hidden_dim = first_weight.shape[0]
    input_dim = first_weight.shape[1]
    print(f"Detected from ckpt: Input Dim: {input_dim}, Hidden Dim: {hidden_dim}")
    
    vocoder = BitVocoder(
        input_dim=input_dim, 
        dim=hidden_dim, 
        num_layers=12, 
        num_res_blocks=2, 
        hop_length=256
    ).to(device)
    
    vocoder.load_state_dict(sd)
    vocoder.eval()
    print("Vocoder loaded.")
    
    # 3. Process Audio -> Mel
    print(f"Processing {input_wav}...")
    # audio, sr = torchaudio.load(input_wav) -> Crashes due to backend
    audio_np, sr = sf.read(input_wav)
    audio = torch.from_numpy(audio_np).float()
    
    # Handle Mono/Stereo
    if audio.dim() > 1:
        audio = audio.mean(1, keepdim=True).t() # (1, T)
    else:
        audio = audio.unsqueeze(0) # (1, T)
    
    # Resample to 24k if needed
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)
    
    audio = audio.to(device)
    
    # Normalize Volume (Match precompute logic)
    audio = audio / (audio.abs().max() + 1e-6) * 0.95
    
    # Mel Transform (Match precompute_flow_dataset.py)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, 
        n_fft=1024, 
        win_length=1024, 
        hop_length=256, 
        n_mels=100
    ).to(device)
    
    with torch.no_grad():
        mel = mel_transform(audio) # (1, 100, T)
        mel_log = torch.log(mel + 1e-5)
        
        print(f"Mel Stats: Max={mel_log.max():.2f}, Min={mel_log.min():.2f}, Mean={mel_log.mean():.2f}")
        
        save_spectrogram(mel_log, "checks/vocoder_isolation_input_mel.png")
        
        # 4. Vocode
        # Input to vocoder should be (B, 100, T)
        audio_hat = vocoder(mel_log)
        
    # Save
    path = output_wav
    sf.write(path, audio_hat.squeeze().cpu().numpy(), 24000)
    print(f"Saved to {path}")

if __name__ == "__main__":
    main()
