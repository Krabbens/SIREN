#!/usr/bin/env python3
"""
Prosty skrypt inferencji z obliczaniem rzeczywistego BPS (bez kodowania entropijnego).
"""
import torch
import torchaudio
import numpy as np
import yaml
import argparse
import os
import sys
import math
import soundfile as sf
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_clean_state_dict(model, path, device):
    """Load state dict, removing _orig_mod. prefix if present."""
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)


def plot_spectrogram(y, y_hat, title, save_path):
    """Plot original vs reconstructed spectrogram."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
    db_transform = torchaudio.transforms.AmplitudeToDB()
    
    # Original
    if y.dim() == 1:
        y = y.unsqueeze(0)
    spec = db_transform(mel_transform(y.cpu())).squeeze().numpy()
    im1 = axs[0].imshow(spec, aspect='auto', origin='lower', vmin=-80, vmax=20)
    axs[0].set_title(f"Original: {title}")
    plt.colorbar(im1, ax=axs[0], format='%+2.0f dB')
    
    # Reconstructed
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(0)
    spec_hat = db_transform(mel_transform(y_hat.cpu())).squeeze().numpy()
    im2 = axs[1].imshow(spec_hat, aspect='auto', origin='lower', vmin=-80, vmax=20)
    axs[1].set_title("Reconstructed")
    plt.colorbar(im2, ax=axs[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SIREN Inference - Raw BPS Calculation")
    parser.add_argument("--input", type=str, required=True, help="Input audio file (WAV)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_v2")
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step to load")
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--output_dir", type=str, default="inference_results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Extract key parameters for BPS calculation
    fsq_levels = config['model']['fsq_levels']  # e.g. [8,8,8,8,8,8,8,8]
    rfsq_num_levels = config['model']['rfsq_num_levels']  # e.g. 3
    sem_temporal_compression = config['model']['semantic']['temporal_compression']  # e.g. 2
    pro_temporal_compression = config['model']['prosody']['temporal_compression']  # e.g. 8
    spk_num_groups = config['model']['speaker']['num_groups']  # e.g. 8
    spk_codes_per_group = config['model']['speaker']['codes_per_group']  # e.g. 256
    
    # Calculate bits per FSQ code
    # FSQ with levels [8,8,8,8,8,8,8,8] has vocab_size = 8^8 = 16,777,216
    fsq_vocab_size = 1
    for level in fsq_levels:
        fsq_vocab_size *= level
    bits_per_fsq_code = math.log2(fsq_vocab_size)
    
    # Speaker bits
    bits_per_spk_code = math.log2(spk_codes_per_group)
    
    print(f"\n{'='*60}")
    print("CODEC CONFIGURATION")
    print(f"{'='*60}")
    print(f"FSQ Levels: {fsq_levels} → vocab_size = {fsq_vocab_size}")
    print(f"Bits per FSQ code: {bits_per_fsq_code:.2f}")
    print(f"RFSQ num levels (residual): {rfsq_num_levels}")
    print(f"Semantic temporal compression: {sem_temporal_compression}x (→ {50/sem_temporal_compression:.1f} Hz)")
    print(f"Prosody temporal compression: {pro_temporal_compression}x (→ {50/pro_temporal_compression:.2f} Hz)")
    print(f"Speaker: {spk_num_groups} groups × {spk_codes_per_group} codes = {bits_per_spk_code:.0f} bits/group")
    print(f"{'='*60}\n")
    
    # Load models
    print("Loading models...")
    factorizer = InformationFactorizerV2(config).to(device)
    decoder = SpeechDecoderV2(config).to(device)
    sem_vq = ResidualFSQ(
        levels=fsq_levels,
        num_levels=rfsq_num_levels,
        input_dim=config['model']['semantic']['output_dim']
    ).to(device)
    pro_vq = ResidualFSQ(
        levels=fsq_levels,
        num_levels=rfsq_num_levels,
        input_dim=config['model']['prosody']['output_dim']
    ).to(device)
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=spk_num_groups,
        codes_per_group=spk_codes_per_group
    ).to(device)
    
    # Load weights
    ckpt_dir = os.path.join(args.checkpoint_dir, f"step_{args.step}")
    if not os.path.exists(ckpt_dir):
        print(f"ERROR: Checkpoint directory not found: {ckpt_dir}")
        return
    
    load_clean_state_dict(factorizer, os.path.join(ckpt_dir, "factorizer.pt"), device)
    load_clean_state_dict(decoder, os.path.join(ckpt_dir, "decoder.pt"), device)
    load_clean_state_dict(sem_vq, os.path.join(ckpt_dir, "sem_rfsq.pt"), device)
    load_clean_state_dict(pro_vq, os.path.join(ckpt_dir, "pro_rfsq.pt"), device)
    load_clean_state_dict(spk_pq, os.path.join(ckpt_dir, "spk_pq.pt"), device)
    print(f"Loaded checkpoint: step {args.step}")
    
    factorizer.eval()
    decoder.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_pq.eval()
    
    # Load HuBERT
    print("Loading HuBERT...")
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    hubert_model.eval()
    
    # Load audio
    print(f"\nProcessing: {args.input}")
    audio_np, sr = sf.read(args.input)
    audio = torch.from_numpy(audio_np).float()
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    duration_seconds = audio.shape[1] / 16000
    print(f"Duration: {duration_seconds:.2f}s ({audio.shape[1]} samples)")
    
    # Extract HuBERT features
    with torch.no_grad():
        inputs = hubert_processor(
            audio.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        outputs = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        layer = config['model'].get('hubert_layer', 9)
        features = outputs.hidden_states[layer]
    
    num_hubert_frames = features.shape[1]  # HuBERT @ 50Hz
    print(f"HuBERT frames: {num_hubert_frames} (@ 50 Hz)")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        # Factorize
        sem, pro, spk = factorizer(features)
        
        # Quantize
        sem_z, _, sem_indices = sem_vq(sem)
        pro_z, _, pro_indices = pro_vq(pro)
        spk_z, _, spk_indices = spk_pq(spk)
        
        # Decode
        audio_hat = decoder(sem_z, pro_z, spk_z)
    
    # Get actual frame counts after temporal compression
    num_sem_frames = sem_indices.shape[1]  # After compression
    num_pro_frames = pro_indices.shape[1]  # After compression
    
    print(f"\n{'='*60}")
    print("TOKEN COUNTS")
    print(f"{'='*60}")
    print(f"Semantic frames: {num_sem_frames} (@ {50/sem_temporal_compression:.1f} Hz)")
    print(f"Prosody frames:  {num_pro_frames} (@ {50/pro_temporal_compression:.2f} Hz)")
    print(f"Speaker groups:  {spk_num_groups} (one-shot per utterance)")
    print(f"{'='*60}\n")
    
    # Calculate RAW BPS (without entropy coding)
    # Semantic: num_frames * rfsq_levels * bits_per_code
    sem_bits = num_sem_frames * rfsq_num_levels * bits_per_fsq_code
    
    # Prosody: num_frames * rfsq_levels * bits_per_code
    pro_bits = num_pro_frames * rfsq_num_levels * bits_per_fsq_code
    
    # Speaker: num_groups * bits_per_group (sent once per utterance)
    spk_bits = spk_num_groups * bits_per_spk_code
    
    total_bits = sem_bits + pro_bits + spk_bits
    total_bps = total_bits / duration_seconds
    
    sem_bps = sem_bits / duration_seconds
    pro_bps = pro_bits / duration_seconds
    spk_bps = spk_bits / duration_seconds
    
    print(f"{'='*60}")
    print("RAW BITRATE (bez kodowania entropijnego)")
    print(f"{'='*60}")
    print(f"Semantic:  {sem_bits:>8.0f} bits → {sem_bps:>8.1f} bps")
    print(f"Prosody:   {pro_bits:>8.0f} bits → {pro_bps:>8.1f} bps")
    print(f"Speaker:   {spk_bits:>8.0f} bits → {spk_bps:>8.1f} bps")
    print(f"{'-'*60}")
    print(f"TOTAL:     {total_bits:>8.0f} bits → {total_bps:>8.1f} bps")
    print(f"{'='*60}\n")
    
    # Theoretical minimum with ideal entropy coding (for reference)
    print(f"{'='*60}")
    print("ANALIZA POZIOMÓW RFSQ")
    print(f"{'='*60}")
    
    for l in range(1, rfsq_num_levels + 1):
        sem_bits_l = num_sem_frames * l * bits_per_fsq_code
        pro_bits_l = num_pro_frames * l * bits_per_fsq_code
        total_bits_l = sem_bits_l + pro_bits_l + spk_bits
        bps_l = total_bits_l / duration_seconds
        print(f"Level {l}: {bps_l:.0f} bps (sem={sem_bits_l/duration_seconds:.0f}, pro={pro_bits_l/duration_seconds:.0f}, spk={spk_bps:.0f})")
    
    print(f"{'='*60}\n")
    
    # Save output
    if audio_hat.dim() == 3:
        audio_hat = audio_hat.squeeze(0)
    if audio_hat.dim() == 2:
        audio_hat = audio_hat.squeeze(0)
    

    # Calculate Mel Loss for filename
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80).to(device)
    orig_mel = mel_transform(audio.to(device))
    rec_mel = mel_transform(audio_hat.to(device))
    
    # Fix length mismatch
    min_len = min(orig_mel.shape[-1], rec_mel.shape[-1])
    orig_mel = orig_mel[..., :min_len]
    rec_mel = rec_mel[..., :min_len]
    
    mel_loss = torch.nn.functional.l1_loss(orig_mel, rec_mel).item()
    print(f"Mel Loss: {mel_loss:.4f}")
    
    basename = os.path.splitext(os.path.basename(args.input))[0]
    out_wav = os.path.join(args.output_dir, f"{basename}_step{args.step}_loss{mel_loss:.3f}.wav")
    out_spec = os.path.join(args.output_dir, f"{basename}_step{args.step}_loss{mel_loss:.3f}_spec.png")
    
    sf.write(out_wav, audio_hat.cpu().numpy(), 16000)
    plot_spectrogram(audio.squeeze(), audio_hat.cpu(), basename, out_spec)
    
    print(f"Saved: {out_wav}")
    print(f"Saved: {out_spec}")
    
    print(f"\n{'='*60}")
    print(f"PODSUMOWANIE: {total_bps:.0f} bps (raw)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
