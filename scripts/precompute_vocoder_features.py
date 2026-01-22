#!/usr/bin/env python3
"""
Precompute Decoder Features for BitNet Vocoder Training

Extract features from the full pipeline (HuBERT → Factorizer → VQ → Decoder Input)
and save them for efficient vocoder training.

Usage:
    python scripts/precompute_vocoder_features.py \
        --checkpoint_dir checkpoints_stable/step_87000 \
        --audio_dir data/audio \
        --output_dir data/vocoder_features
"""

import os
import sys
import argparse
import yaml
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import glob

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer


def load_hubert(device):
    """Load HuBERT model for feature extraction."""
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    
    print("Loading HuBERT...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()
    
    return processor, model


@torch.no_grad()
def extract_features(audio, hubert_processor, hubert_model, factorizer, 
                     sem_vq, pro_vq, spk_pq, decoder, device, use_l1=True):
    """
    Extract decoder input features from audio.
    
    Pipeline:
        Audio → HuBERT → Factorizer → VQ → Reconstructor → Features
    
    Returns:
        Fused features ready for vocoder input (B, C, T)
    """
    # Ensure 16kHz mono
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Process with HuBERT
    inputs = hubert_processor(
        audio.squeeze().cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(device)
    
    # Extract HuBERT features
    hubert_out = hubert_model(input_values)
    hubert_features = hubert_out.last_hidden_state  # (B, T, 768)
    
    # Factorize
    # factorizer expects (B, T, C)
    semantic, prosody, speaker = factorizer(hubert_features)
    
    # Quantize
    if use_l1:
        # L1: only first level of RFSQ
        sem_q, _, _ = sem_vq(semantic, num_levels=1)
        pro_q, _, _ = pro_vq(prosody, num_levels=1)
    else:
        sem_q, _, _ = sem_vq(semantic)
        pro_q, _, _ = pro_vq(prosody)
    
    spk_q, _, _ = spk_pq(speaker)
    
    # Fuse features using pretrained Reconstructor
    # reconstructor expects (B, T, C) and returns (B, T, C)
    fused = decoder.reconstructor(sem_q, pro_q, spk_q)
    
    # Vocoder expects (B, C, T)
    fused = fused.transpose(1, 2)
    
    return fused


def load_models(checkpoint_dir, config, device):
    """Load factorizer, quantizers, and decoder from checkpoint."""
    print(f"Loading models from {checkpoint_dir}...")
    
    # Initialize models
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
    
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    
    # Load weights
    def load_state_dict(model, path):
        if os.path.exists(path):
            state = torch.load(path, map_location=device)
            # Handle DDP wrapped models
            state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            print(f"Loaded {path}")
        else:
            print(f"Warning: {path} not found")
    
    load_state_dict(factorizer, os.path.join(checkpoint_dir, "factorizer.pt"))
    load_state_dict(decoder, os.path.join(checkpoint_dir, "decoder.pt"))
    load_state_dict(sem_vq, os.path.join(checkpoint_dir, "sem_rfsq.pt"))
    load_state_dict(pro_vq, os.path.join(checkpoint_dir, "pro_rfsq.pt"))
    load_state_dict(spk_pq, os.path.join(checkpoint_dir, "spk_pq.pt"))
    
    factorizer.eval()
    decoder.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_pq.eval()
    
    return factorizer, sem_vq, pro_vq, spk_pq, decoder


def process_audio_file(audio_path, hubert_processor, hubert_model, factorizer,
                       sem_vq, pro_vq, spk_pq, decoder, device, sample_rate=16000):
    """Process a single audio file and extract features."""
    try:
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        
        # Convert to mono
        audio = audio.mean(0)
        
        # Skip very short files
        if len(audio) < sample_rate:  # Less than 1 second
            return None
        
        # Move to device
        audio = audio.to(device)
        
        # Extract features
        features = extract_features(
            audio, hubert_processor, hubert_model, factorizer,
            sem_vq, pro_vq, spk_pq, decoder, device
        )
        
        return features.cpu()
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Precompute vocoder features")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--config", type=str, 
                        default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory with training audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for features")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process")
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load models
    hubert_processor, hubert_model = load_hubert(device)
    factorizer, sem_vq, pro_vq, spk_pq, decoder = load_models(
        args.checkpoint_dir, config, device
    )
    
    # Find audio files
    audio_files = []
    for ext in ['*.wav', '*.flac', '*.mp3']:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, '**', ext), recursive=True))
    
    print(f"Found {len(audio_files)} audio files")
    
    if args.max_files:
        audio_files = audio_files[:args.max_files]
        print(f"Processing first {len(audio_files)} files")
    
    # Process files
    processed = 0
    failed = 0
    
    for audio_path in tqdm(audio_files, desc="Extracting features"):
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(args.output_dir, f"{basename}.pt")
        
        # Skip if already processed
        if os.path.exists(output_path):
            processed += 1
            continue
        
        features = process_audio_file(
            audio_path, hubert_processor, hubert_model, factorizer,
            sem_vq, pro_vq, spk_pq, decoder, device
        )
        
        if features is not None:
            torch.save(features, output_path)
            processed += 1
        else:
            failed += 1
    
    print(f"\nDone! Processed: {processed}, Failed: {failed}")
    print(f"Features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
