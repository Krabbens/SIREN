import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import sys
import yaml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder

def plot_spectrogram(audio, title, save_path):
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80
    )(audio)
    plt.figure(figsize=(10, 4))
    plt.imshow(torch.log(spec[0] + 1e-5).cpu().numpy(), origin='lower', aspect='auto')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    feat_path = "data/vocoder_features/1034_1034_121119_000010_000004.pt"
    audio_path = "data/audio/1034_1034_121119_000010_000004.wav"
    checkpoint_path = "checkpoints_bitnet/checkpoint_epoch60.pt"
    output_dir = "bitnet_test_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = BitVocoder(input_dim=512, dim=256, num_layers=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded BitVocoder from {checkpoint_path} (epoch {checkpoint['epoch']})")

    # Load features
    features = torch.load(feat_path).to(device)
    if features.dim() == 2:
        features = features.unsqueeze(0)
    print(f"Features shape: {features.shape}")

    # Generate audio
    with torch.no_grad():
        pred_audio = model(features)
        # BitVocoder might return (B, 1, T) or (B, T) depending on iSTFT implementation
        if pred_audio.dim() == 3:
            pred_audio = pred_audio.squeeze(1)
    
    # Save audio
    torchaudio.save(os.path.join(output_dir, "bitnet_reconstructed.wav"), pred_audio.cpu(), 16000)
    print(f"Saved reconstructed audio to {output_dir}/bitnet_reconstructed.wav")

    # Load and save original for comparison
    if os.path.exists(audio_path):
        orig_audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            orig_audio = torchaudio.functional.resample(orig_audio, sr, 16000)
        torchaudio.save(os.path.join(output_dir, "original.wav"), orig_audio, 16000)
        plot_spectrogram(orig_audio, "Original Spectrogram", os.path.join(output_dir, "spectrogram_original.png"))
        print(f"Saved original audio and spectrogram to {output_dir}")

    # Plot bitnet spectrogram
    plot_spectrogram(pred_audio.cpu(), f"BitNet Spectrogram (Epoch {checkpoint['epoch']})", 
                     os.path.join(output_dir, "spectrogram_bitnet.png"))
    print(f"Saved BitNet spectrogram to {output_dir}/spectrogram_bitnet.png")

if __name__ == "__main__":
    main()
