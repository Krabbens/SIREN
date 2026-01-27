"""
Inference and Spectrogram Visualization
Runs codec on a sample audio file and plots original vs reconstructed spectrograms.
"""
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(PROJECT_ROOT)

from ultra_low_bitrate_codec.models.encoder import SpeechEncoder
from ultra_low_bitrate_codec.models.quantizers import VectorQuantizer, ProductQuantizer
from ultra_low_bitrate_codec.models.decoder import SpeechDecoder
import yaml

def load_audio(path, target_sr=16000, max_duration=3.0):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    max_samples = int(max_duration * target_sr)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]
    return audio.squeeze(0)

def compute_spectrogram(audio, sr=16000, n_fft=1024, hop_length=256):
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=80
    )(audio.unsqueeze(0))
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    return spec_db.squeeze().numpy()

def main(audio_path, config_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading models...")
    # Load models
    encoder = SpeechEncoder(config).to(device)
    
    sem_vq = VectorQuantizer(
        dim=config['model']['semantic']['output_dim'],
        vocab_size=config['model']['semantic']['vocab_size']
    ).to(device)
    
    pro_vq = VectorQuantizer(
        dim=config['model']['prosody']['output_dim'],
        vocab_size=config['model']['prosody']['vocab_size']
    ).to(device)
    
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    
    decoder = SpeechDecoder(config).to(device)
    
    # Set to eval mode
    encoder.eval()
    decoder.eval()
    
    print(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path).to(device)
    
    print("Running inference...")
    with torch.no_grad():
        # Encode
        sem, pro, spk = encoder(audio.unsqueeze(0))
        
        # Quantize
        sem_z, _, sem_idx = sem_vq(sem)
        pro_z, _, pro_idx = pro_vq(pro)
        spk_z, _, spk_idx = spk_pq(spk)
        
        # Decode
        audio_hat = decoder(sem_z, pro_z, spk_z)
    
    # Match lengths
    min_len = min(audio.shape[0], audio_hat.shape[1])
    audio = audio[:min_len].cpu()
    audio_hat = audio_hat[0, :min_len].cpu()
    
    print("Computing spectrograms...")
    orig_spec = compute_spectrogram(audio)
    recon_spec = compute_spectrogram(audio_hat)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    im1 = axes[0].imshow(orig_spec, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title('Original Audio Spectrogram')
    axes[0].set_ylabel('Mel Bins')
    plt.colorbar(im1, ax=axes[0], label='dB')
    
    im2 = axes[1].imshow(recon_spec, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title(f'Reconstructed Audio Spectrogram (~{65} bps)')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Mel Bins')
    plt.colorbar(im2, ax=axes[1], label='dB')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved spectrogram to: {output_path}")
    
    # Also save audio files
    torchaudio.save(output_path.replace('.png', '_original.wav'), audio.unsqueeze(0), 16000)
    torchaudio.save(output_path.replace('.png', '_reconstructed.wav'), audio_hat.unsqueeze(0), 16000)
    print("Saved audio files too!")

if __name__ == "__main__":
    DEFAULT_AUDIO = os.path.join(PROJECT_ROOT, 'data/LJSpeech-1.1/wavs/LJ001-0001.wav')
    DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, 'ultra_low_bitrate_codec/configs/default.yaml')
    DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, 'spectrogram_comparison.png')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, default=DEFAULT_AUDIO)
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    
    main(args.audio, args.config, args.output)
