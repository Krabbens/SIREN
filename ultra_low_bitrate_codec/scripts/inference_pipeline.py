"""
Full Inference Pipeline for Ultra-Low Bitrate Codec V2
Creates spectrogram comparison between original and reconstructed audio.
"""
import torch
import torchaudio
import yaml
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.feature_extractor import HubertFeatureExtractor


def load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load and resample audio to target sample rate."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0)  # (T,)


def compute_mel_spectrogram(audio: torch.Tensor, sr: int = 16000, 
                            n_fft: int = 1024, hop_length: int = 256,
                            n_mels: int = 80) -> np.ndarray:
    """Compute mel spectrogram for visualization."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=8000
    )
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    mel = mel_transform(audio)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    
    return mel_db.squeeze().numpy()


def plot_comparison(original_audio: torch.Tensor, 
                    reconstructed_audio: torch.Tensor,
                    sr: int = 16000,
                    output_path: str = "comparison.png",
                    title: str = "Ultra-Low Bitrate Codec (~68 bps)"):
    """Create a comprehensive comparison plot."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Align lengths
    min_len = min(len(original_audio), len(reconstructed_audio))
    orig = original_audio[:min_len].cpu()
    recon = reconstructed_audio[:min_len].cpu()
    
    time = np.arange(min_len) / sr
    
    # Row 1: Waveforms
    axes[0, 0].plot(time, orig.numpy(), color='#2ecc71', alpha=0.8, linewidth=0.5)
    axes[0, 0].set_title('Original Waveform', fontsize=12)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_xlim([0, time[-1]])
    
    axes[0, 1].plot(time, recon.numpy(), color='#e74c3c', alpha=0.8, linewidth=0.5)
    axes[0, 1].set_title('Reconstructed Waveform', fontsize=12)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_xlim([0, time[-1]])
    
    # Row 2: Mel Spectrograms
    mel_orig = compute_mel_spectrogram(orig)
    mel_recon = compute_mel_spectrogram(recon)
    
    im1 = axes[1, 0].imshow(mel_orig, aspect='auto', origin='lower', 
                            cmap='magma', extent=[0, time[-1], 0, 80])
    axes[1, 0].set_title('Original Mel Spectrogram', fontsize=12)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Mel Bin')
    plt.colorbar(im1, ax=axes[1, 0], label='dB')
    
    im2 = axes[1, 1].imshow(mel_recon, aspect='auto', origin='lower', 
                            cmap='magma', extent=[0, time[-1], 0, 80])
    axes[1, 1].set_title('Reconstructed Mel Spectrogram', fontsize=12)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Mel Bin')
    plt.colorbar(im2, ax=axes[1, 1], label='dB')
    
    # Row 3: Difference and Stats
    # Spectrogram difference
    diff = mel_orig - mel_recon
    vmax = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2, 0].imshow(diff, aspect='auto', origin='lower', 
                            cmap='coolwarm', extent=[0, time[-1], 0, 80],
                            vmin=-vmax, vmax=vmax)
    axes[2, 0].set_title('Difference (Original - Reconstructed)', fontsize=12)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Mel Bin')
    plt.colorbar(im3, ax=axes[2, 0], label='dB Difference')
    
    # Statistics text
    axes[2, 1].axis('off')
    
    # Calculate metrics
    mse = np.mean((mel_orig - mel_recon) ** 2)
    mae = np.mean(np.abs(mel_orig - mel_recon))
    
    # Waveform correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(orig.numpy().flatten(), recon.numpy().flatten())
    
    stats_text = f"""
    === Audio Quality Metrics ===
    
    Duration: {time[-1]:.2f} seconds
    Sample Rate: {sr} Hz
    
    Mel Spectrogram:
    • MSE: {mse:.4f}
    • MAE: {mae:.4f}
    
    Waveform:
    • Pearson Correlation: {corr:.4f}
    • RMS (Original): {torch.sqrt(torch.mean(orig**2)).item():.4f}
    • RMS (Reconstructed): {torch.sqrt(torch.mean(recon**2)).item():.4f}
    
    === Bitrate Estimation ===
    
    Semantic: ~12.5 Hz × 18.6 bits ≈ 232 bps
    Prosody: ~6.25 Hz × 18.6 bits ≈ 116 bps
    Speaker: 32 bits (one-time)
    
    Total (raw): ~350 bps
    Est. with entropy: ~68 bps
    """
    
    axes[2, 1].text(0.1, 0.9, stats_text, transform=axes[2, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to: {output_path}")
    return mse, corr


# Get project root for default paths
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_DEFAULT_CHECKPOINT_DIR = str(_PROJECT_ROOT / "checkpoints_v2")
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "ultra_low_bitrate_codec" / "configs" / "improved.yaml")
_DEFAULT_OUTPUT = str(_PROJECT_ROOT / "ultra_low_bitrate_codec" / "inference_output")

def run_inference(audio_path: str, 
                  checkpoint_dir: str = _DEFAULT_CHECKPOINT_DIR,
                  config_path: str = _DEFAULT_CONFIG,
                  step: int = 16500,
                  output_dir: str = _DEFAULT_OUTPUT):
    """Run full inference pipeline."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading models...")
    
    # Initialize models
    hubert = HubertFeatureExtractor(
        model_name=config['model']['hubert_model'],
        target_layer=config['model']['hubert_layer'],
        freeze=True
    ).to(device)
    
    factorizer = InformationFactorizerV2(config).to(device)
    decoder = SpeechDecoderV2(config).to(device)
    
    # Quantizers
    fsq_levels = config['model']['fsq_levels']
    sem_dim = config['model']['semantic']['output_dim']
    pro_dim = config['model']['prosody']['output_dim']
    
    sem_rfsq = ResidualFSQ(
        levels=fsq_levels, 
        num_levels=4, 
        input_dim=sem_dim
    ).to(device)
    
    pro_rfsq = ResidualFSQ(
        levels=fsq_levels, 
        num_levels=4, 
        input_dim=pro_dim
    ).to(device)
    
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    
    # Load checkpoints
    print(f"Loading checkpoints from step {step}...")
    
    factorizer.load_state_dict(
        torch.load(f"{checkpoint_dir}/factorizer_{step}.pt", map_location=device, weights_only=True)
    )
    decoder.load_state_dict(
        torch.load(f"{checkpoint_dir}/decoder_{step}.pt", map_location=device, weights_only=True)
    )
    sem_rfsq.load_state_dict(
        torch.load(f"{checkpoint_dir}/sem_rfsq_{step}.pt", map_location=device, weights_only=True)
    )
    pro_rfsq.load_state_dict(
        torch.load(f"{checkpoint_dir}/pro_rfsq_{step}.pt", map_location=device, weights_only=True)
    )
    spk_pq.load_state_dict(
        torch.load(f"{checkpoint_dir}/spk_pq_{step}.pt", map_location=device, weights_only=True)
    )
    
    print("All models loaded successfully!")
    
    # Set to eval mode
    hubert.eval()
    factorizer.eval()
    decoder.eval()
    sem_rfsq.eval()
    pro_rfsq.eval()
    spk_pq.eval()
    
    # Load audio
    print(f"Loading audio from: {audio_path}")
    audio = load_audio(audio_path, target_sr=16000)
    print(f"Audio shape: {audio.shape}, Duration: {len(audio)/16000:.2f}s")
    
    # Process
    audio_input = audio.unsqueeze(0).to(device)  # (1, T)
    
    print("Running inference...")
    with torch.no_grad():
        # Extract HuBERT features
        hubert_features = hubert(audio_input)  # (1, T/320, 768)
        print(f"HuBERT features shape: {hubert_features.shape}")
        
        # Factorize
        sem, pro, spk = factorizer(hubert_features)
        print(f"Semantic: {sem.shape}, Prosody: {pro.shape}, Speaker: {spk.shape}")
        
        # Quantize
        sem_q, sem_loss, sem_indices = sem_rfsq(sem)
        pro_q, pro_loss, pro_indices = pro_rfsq(pro)
        spk_q, spk_loss, spk_indices = spk_pq(spk)
        print(f"Quantized - Sem: {sem_q.shape}, Pro: {pro_q.shape}, Spk: {spk_q.shape}")
        
        # Decode
        audio_recon = decoder(sem_q, pro_q, spk_q)
        print(f"Reconstructed audio shape: {audio_recon.shape}")
    
    # Save audio files
    audio_name = Path(audio_path).stem
    
    # Ensure correct dimensions for saving
    orig_save = audio.unsqueeze(0) if audio.dim() == 1 else audio
    recon_save = audio_recon.squeeze().cpu()
    if recon_save.dim() == 1:
        recon_save = recon_save.unsqueeze(0)
    
    orig_path = f"{output_dir}/{audio_name}_original.wav"
    recon_path = f"{output_dir}/{audio_name}_reconstructed.wav"
    
    torchaudio.save(orig_path, orig_save.cpu(), 16000)
    torchaudio.save(recon_path, recon_save, 16000)
    
    print(f"Saved original to: {orig_path}")
    print(f"Saved reconstructed to: {recon_path}")
    
    # Create comparison plot
    plot_path = f"{output_dir}/{audio_name}_comparison.png"
    mse, corr = plot_comparison(
        audio, 
        audio_recon.squeeze().cpu(),
        output_path=plot_path,
        title=f"Ultra-Low Bitrate Codec - {audio_name} (Step {step})"
    )
    
    print(f"\n=== Results ===")
    print(f"Mel MSE: {mse:.4f}")
    print(f"Waveform Correlation: {corr:.4f}")
    print(f"Files saved to: {output_dir}")
    
    return {
        'mse': mse,
        'correlation': corr,
        'original_path': orig_path,
        'reconstructed_path': recon_path,
        'plot_path': plot_path
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Low Bitrate Codec Inference")
    parser.add_argument('--audio', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--step', type=int, default=16500, help='Checkpoint step')
    parser.add_argument('--output', type=str, default=_DEFAULT_OUTPUT,
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_inference(
        audio_path=args.audio,
        step=args.step,
        output_dir=args.output
    )
