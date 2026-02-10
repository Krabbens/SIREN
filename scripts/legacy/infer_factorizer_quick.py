import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import argparse
import sys
# import torchaudio # CAUSES CRASH DUE TO TORCHCODEC

# Add src to path
sys.path.append(os.path.abspath("src"))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2, InformationFactorizer
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2, FeatureReconstructorV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ
from ultra_low_bitrate_codec.models.bitlinear import BitLinear, BitConv1d

import yaml

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs_monitoring')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # Load Models
    print("Using InformationFactorizerV2 (Target Model)...")
    factorizer = InformationFactorizerV2(config).to(device)
    decoder = SpeechDecoderV2(config).to(device)
    
    sem_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=config['model']['semantic']['output_dim']
    ).to(device)
    
    pro_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=config['model']['prosody']['output_dim']
    ).to(device)
    
    # Load Checkpoint (Split files)
    print(f"Loading checkpoint from dir: {args.checkpoint}")
    
    # helper
    def load_part(model, name):
        p = os.path.join(args.checkpoint, name)
        print(f"Loading {name}...")
        try:
            model.load_state_dict(torch.load(p, map_location=device), strict=False)
        except Exception as e:
            print(f"WARNING: Failed to load {name}: {e}")

    load_part(factorizer, 'factorizer.pt')
    load_part(decoder, 'decoder.pt')
    load_part(sem_vq, 'sem_rfsq.pt')
    load_part(pro_vq, 'pro_rfsq.pt')
    
    # spk_pq is not used in this inference script (we use factorizer output directly)
    
    factorizer.eval()
    decoder.eval()
    
    import librosa
    import numpy as np
    
    # Load Audio using Librosa (more robust than torchaudio on some systems)
    # wav, sr = torchaudio.load(args.input_file)
    wav_np, sr = librosa.load(args.input_file, sr=16000)
    wav = torch.from_numpy(wav_np).unsqueeze(0) # (1, T)
    
    # if sr != 16000:
    #     resampler = torchaudio.transforms.Resample(sr, 16000)
    #     wav = resampler(wav)
    
    # Compute Mel for GT comparison (using decoder's mel transform logic if possible, or torchaudio)
    # Reusing the decoder's mel transform would be best but it's internal.
    # Let's trust Factorizer to take Hubert features.
    # WAIT: Factorizer takes HuBERT features, not Waveform!
    # We need to extract features first.
    
    # Correction: The Factorizer takes HuBERT features.
    
    # We use the real FeatureExtractor.
    # For a quick check, we can try to find the precomputed features for 'jakubie' if they exist.
    # Or simplified: Just check the Decoder's ability to reconstruct from "something".
    
    # Actually, we can use the `FeatureExtractor` if we import it.
    # Use TinyHubert (Correct Model)
    from ultra_low_bitrate_codec.models.tiny_hubert import TinyHubert
    
    print("Loading TinyHubert...")
    extractor = TinyHubert(out_dim=768, hidden_dim=384, num_layers=12).to(device)
    # Load checkpoint
    tiny_ckpt = "checkpoints/tiny_hubert_best.pt"
    if os.path.exists(tiny_ckpt):
        extractor.load_state_dict(torch.load(tiny_ckpt, map_location=device))
        print(f"Loaded TinyHubert from {tiny_ckpt}")
    else:
        print(f"WARNING: {tiny_ckpt} not found! Using random initialization (GARBAGE OUTPUT EXPECTED)")
    
    extractor.eval()
    
    wav_16k = wav.to(device)
    if wav_16k.dim() == 1: wav_16k = wav_16k.unsqueeze(0)
    
    print("Extracting features...")
    with torch.no_grad():
        features = extractor(wav_16k) # (B, T, 768)
        
        # Inference
        sem, pro, spk = factorizer(features)
        
        # Quantize
        sem_q, _, sem_codes = sem_vq(sem)
        pro_q, _, pro_codes = pro_vq(pro)
        
        # Decode
        out = decoder(sem_q, pro_q, spk)
        print(f"DEBUG: Decoder output type: {type(out)}")
        if isinstance(out, torch.Tensor):
            print(f"DEBUG: Decoder output shape: {out.shape}")
            wave_pred = out
        elif isinstance(out, tuple):
            print(f"DEBUG: Decoder output tuple len: {len(out)}")
            wave_pred = out[0]
        else:
            print(f"DEBUG: Decoder output strange: {out}")
            wave_pred = out # hope for best
        
        # if isinstance(wave_pred_tuple, tuple):
        #      wave_pred = wave_pred_tuple[0] # Often (wave, other features)
        # else:
        #      wave_pred = wave_pred_tuple
             
    # Save Outputs
    output_base = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input_file))[0])
    
    # Save Wave
    import soundfile as sf
    sf.write(f"{output_base}_recon.wav", wave_pred.squeeze().detach().cpu().numpy(), 16000)
    
    # Save Spectrogram
    # use librosa for mel
    wave_np_recon = wave_pred.squeeze().detach().cpu().numpy()
    
    def get_mel(y):
        return librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=256, n_mels=80)
        
    reconst_mel = np.log10(get_mel(wave_np_recon) + 1e-9)
    # orig_mel = mel_transform(wav_16k).squeeze(0).cpu().log10()
    orig_mel = np.log10(get_mel(wav_np) + 1e-9)
    
    # Calculate Energy Envelopes (Mean over frequency)
    orig_energy = orig_mel.mean(axis=0)
    recon_energy = reconst_mel.mean(axis=0)
    
    # Normalize for easier visual comparison
    orig_energy = (orig_energy - orig_energy.min()) / (orig_energy.max() - orig_energy.min())
    recon_energy = (recon_energy - recon_energy.min()) / (recon_energy.max() - recon_energy.min())
    
    plt.figure(figsize=(10, 12))
    
    plt.subplot(3, 1, 1)
    plt.imshow(orig_mel, origin='lower', aspect='auto')
    plt.title("Original Mel")
    plt.colorbar()
    
    plt.subplot(3, 1, 2)
    plt.imshow(reconst_mel, origin='lower', aspect='auto')
    plt.title("Reconstructed Mel (Factorizer Tiny)")
    plt.colorbar()
    
    plt.subplot(3, 1, 3)
    plt.plot(orig_energy, label="Original Energy", alpha=0.7)
    plt.plot(recon_energy, label="Recon Energy", alpha=0.7)
    plt.title("Energy Envelope Alignment (Proof of Encoder Sync)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_base}_comparison.png")
    print(f"Saved to {output_base}_comparison.png")

if __name__ == "__main__":
    main()
