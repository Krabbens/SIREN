#!/usr/bin/env python3
"""
Quality Evaluation Script for SIREN Ultra-Low Bitrate Codec

Evaluates:
1. Bitrate (target: 190-230 bps)
2. MOS quality (target: > 3.0)
3. Spectral quality metrics

Used for overnight autonomous training monitoring.
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoder
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.models.adapter import FeatureAdapter
import yaml


def load_config(config_path="ultra_low_bitrate_codec/configs/sub100bps.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def calculate_bitrate(features, audio_duration, quantizer_levels=None):
    """
    Calculate bitrate in bits per second (bps).
    
    For FSQ quantization:
    - N levels requires log2(N) bits
    - Total bits = num_frames * num_dims * bits_per_dim
    - BPS = total_bits / duration
    
    For continuous features (no quantization):
    - Estimate based on feature dimensions and frame rate
    """
    # Default FSQ levels from sub100bps.yaml: [6, 6, 6, 6]
    if quantizer_levels is None:
        quantizer_levels = [6, 6, 6, 6]
    
    # Features: (B, T, C) or (B, C, T)
    if features.dim() == 3:
        if features.shape[-1] == 768 or features.shape[-1] == 512:
            # (B, T, C)
            num_frames = features.shape[1]
        else:
            # (B, C, T)
            num_frames = features.shape[2]
    else:
        num_frames = features.shape[0] if features.dim() == 2 else 1
    
    # Each FSQ level: log2(level) bits
    bits_per_quantizer = [np.log2(level) for level in quantizer_levels]
    total_bits_per_frame = sum(bits_per_quantizer)
    
    total_bits = num_frames * total_bits_per_frame
    bps = total_bits / audio_duration
    
    return bps, {
        'num_frames': num_frames,
        'bits_per_frame': total_bits_per_frame,
        'total_bits': total_bits,
        'duration_sec': audio_duration,
        'bps': bps
    }


def evaluate_mos(audio_path_or_tensor, sample_rate=16000):
    """
    Evaluate MOS using speechmos library.
    Returns MOS score (1-5 scale).
    """
    try:
        from speechmos import dnsmos
        
        if isinstance(audio_path_or_tensor, (str, Path)):
            audio, sr = torchaudio.load(audio_path_or_tensor)
            if sr != sample_rate:
                audio = torchaudio.functional.resample(audio, sr, sample_rate)
            audio = audio.mean(0).numpy()
        else:
            audio = audio_path_or_tensor.cpu().numpy()
            if audio.ndim > 1:
                audio = audio.mean(0) if audio.shape[0] < audio.shape[-1] else audio.flatten()
        
        # DNSMOS evaluation
        result = dnsmos.run(audio, sample_rate)
        return {
            'ovrl': result.ovrl,  # Overall MOS
            'sig': result.sig,    # Signal quality
            'bak': result.bak,    # Background quality
            'p808_mos': result.p808_mos if hasattr(result, 'p808_mos') else result.ovrl
        }
    except Exception as e:
        print(f"MOS evaluation error: {e}")
        return {'ovrl': 0.0, 'sig': 0.0, 'bak': 0.0, 'error': str(e)}


def evaluate_spectral_quality(original, reconstructed, sample_rate=16000):
    """
    Evaluate spectral quality metrics.
    """
    # Ensure same length
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]
    
    # Mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80
    )
    
    orig_mel = mel_transform(original)
    rec_mel = mel_transform(reconstructed)
    
    # Log mel
    orig_log = torch.log(orig_mel.clamp(min=1e-5))
    rec_log = torch.log(rec_mel.clamp(min=1e-5))
    
    # Mel Cepstral Distortion
    mcd = torch.sqrt(2 * torch.mean((orig_log - rec_log) ** 2))
    
    # Spectral Convergence
    sc = torch.norm(orig_mel - rec_mel) / torch.norm(orig_mel)
    
    # Log Spectral Distance
    lsd = torch.mean(torch.sqrt(torch.mean((orig_log - rec_log) ** 2, dim=-1)))
    
    return {
        'mcd': mcd.item(),
        'spectral_convergence': sc.item(),
        'log_spectral_distance': lsd.item()
    }


def run_inference_pipeline(audio_path, encoder, adapter, vocoder, device='cuda'):
    """
    Run full inference pipeline:
    Audio -> MicroEncoder -> Adapter -> BitVocoder -> Audio
    
    Returns reconstructed audio and bitrate info.
    """
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio.mean(0)  # Mono
    
    audio_duration = len(audio) / 16000
    audio_batch = audio.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encode
        features = encoder(audio_batch)  # (B, T, 768)
        
        # Adapt to vocoder dim
        adapted = adapter(features)  # (B, T, 512)
        
        # Decode
        audio_rec = vocoder(adapted)
    
    # Calculate bitrate
    bps, bitrate_info = calculate_bitrate(features, audio_duration)
    
    return {
        'original': audio_batch.cpu(),
        'reconstructed': audio_rec.cpu(),
        'features': features.cpu(),
        'duration': audio_duration,
        'bps': bps,
        'bitrate_info': bitrate_info
    }


def load_models(vocoder_checkpoint, device='cuda'):
    """
    Load all models for inference.
    """
    config = load_config()
    
    # Encoder
    encoder = MicroEncoder().to(device)
    encoder_ckpt = torch.load('checkpoints/checkpoints_micro_encoder/best_model.pt', map_location=device)
    encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    encoder.eval()
    
    # Adapter
    adapter = FeatureAdapter(in_dim=768, out_dim=512, upsample_factor=1).to(device)
    adapter_ckpt = torch.load('checkpoints/checkpoints_adapter/checkpoint_epoch9.pt', map_location=device)
    adapter.load_state_dict(adapter_ckpt, strict=False)
    adapter.eval()
    
    # Vocoder
    voc_conf = config['model'].get('vocoder', {})
    vocoder = BitVocoder(
        input_dim=512,
        dim=256,
        n_fft=1024,
        hop_length=320,
        num_layers=voc_conf.get('num_convnext_layers', 8),
        num_res_blocks=voc_conf.get('num_res_blocks', 3)
    ).to(device)
    
    voc_ckpt = torch.load(vocoder_checkpoint, map_location=device)
    state = voc_ckpt.get('model_state_dict', voc_ckpt)
    # Remove model. prefix if present
    state = {k[6:] if k.startswith('model.') else k: v for k, v in state.items()}
    vocoder.load_state_dict(state, strict=False)
    vocoder.eval()
    
    return encoder, adapter, vocoder


def full_evaluation(vocoder_checkpoint, test_audio_path, device='cuda'):
    """
    Run full evaluation: bitrate + MOS + spectral quality
    """
    print(f"Loading models with vocoder: {vocoder_checkpoint}")
    encoder, adapter, vocoder = load_models(vocoder_checkpoint, device)
    
    print(f"Running inference on: {test_audio_path}")
    result = run_inference_pipeline(test_audio_path, encoder, adapter, vocoder, device)
    
    print(f"Calculating MOS...")
    # Save reconstructed for MOS evaluation
    rec_path = "/tmp/reconstructed_eval.wav"
    torchaudio.save(rec_path, result['reconstructed'].squeeze(0).unsqueeze(0), 16000)
    mos = evaluate_mos(rec_path)
    
    print(f"Calculating spectral quality...")
    spectral = evaluate_spectral_quality(result['original'], result['reconstructed'])
    
    evaluation = {
        'timestamp': datetime.now().isoformat(),
        'vocoder_checkpoint': vocoder_checkpoint,
        'test_audio': test_audio_path,
        'duration_sec': result['duration'],
        'bitrate': {
            'bps': result['bps'],
            'info': result['bitrate_info']
        },
        'mos': mos,
        'spectral': spectral,
        'targets': {
            'bps_target': '190-230',
            'mos_target': '> 3.0',
            'bps_ok': 190 <= result['bps'] <= 230,
            'mos_ok': mos.get('ovrl', 0) > 3.0
        }
    }
    
    return evaluation


def print_evaluation(eval_result):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Timestamp: {eval_result['timestamp']}")
    print(f"Vocoder: {eval_result['vocoder_checkpoint']}")
    print(f"Audio duration: {eval_result['duration_sec']:.2f}s")
    print()
    print(f"BITRATE:")
    print(f"  BPS: {eval_result['bitrate']['bps']:.2f}")
    print(f"  Target: {eval_result['targets']['bps_target']}")
    print(f"  Status: {'✓ OK' if eval_result['targets']['bps_ok'] else '✗ OUT OF RANGE'}")
    print()
    print(f"MOS QUALITY:")
    print(f"  Overall: {eval_result['mos'].get('ovrl', 0):.3f}")
    print(f"  Signal: {eval_result['mos'].get('sig', 0):.3f}")
    print(f"  Background: {eval_result['mos'].get('bak', 0):.3f}")
    print(f"  Target: {eval_result['targets']['mos_target']}")
    print(f"  Status: {'✓ OK' if eval_result['targets']['mos_ok'] else '✗ NEEDS IMPROVEMENT'}")
    print()
    print(f"SPECTRAL QUALITY:")
    print(f"  MCD: {eval_result['spectral']['mcd']:.4f}")
    print(f"  Spectral Convergence: {eval_result['spectral']['spectral_convergence']:.4f}")
    print(f"  Log Spectral Distance: {eval_result['spectral']['log_spectral_distance']:.4f}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocoder", type=str, default="checkpoints/checkpoints_bitnet_v2_antibanding/best_model.pt")
    parser.add_argument("--audio", type=str, default="data/jakubie_16k.wav")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.vocoder):
        print(f"Vocoder checkpoint not found: {args.vocoder}")
        print("Looking for alternatives...")
        for ckpt in ["checkpoints/checkpoints_bitnet/best_model.pt", "checkpoints/checkpoints_bitnet_v2_antibanding/checkpoint_epoch10.pt"]:
            if os.path.exists(ckpt):
                args.vocoder = ckpt
                print(f"Found: {ckpt}")
                break
    
    result = full_evaluation(args.vocoder, args.audio)
    print_evaluation(result)
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")
