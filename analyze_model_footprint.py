import torch
import yaml
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from ultra_low_bitrate_codec.data.feature_dataset import PrecomputedFeatureDataset

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024

def analyze_footprint(config_path, checkpoint_dir):
    print(f"📊 Analyzing Model Footprint for {config_path}")
    print("=" * 60)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Initialize Models
    factorizer = InformationFactorizerV2(config).to(device)
    decoder = SpeechDecoderV2(config).to(device)
    entropy_model = EntropyModel(config).to(device)
    
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
    
    # Load Checkpoints
    print("Loading checkpoints...") 
    try:
        factorizer.load_state_dict(torch.load(f"{checkpoint_dir}/factorizer.pt", map_location=device), strict=False)
        decoder.load_state_dict(torch.load(f"{checkpoint_dir}/decoder.pt", map_location=device), strict=False)
        entropy_model.load_state_dict(torch.load(f"{checkpoint_dir}/entropy.pt", map_location=device), strict=False)
        sem_vq.load_state_dict(torch.load(f"{checkpoint_dir}/sem_rfsq.pt", map_location=device), strict=False)
        pro_vq.load_state_dict(torch.load(f"{checkpoint_dir}/pro_rfsq.pt", map_location=device), strict=False)
        spk_pq.load_state_dict(torch.load(f"{checkpoint_dir}/spk_pq.pt", map_location=device), strict=False)
    except FileNotFoundError as e:
        print(f"⚠️ Error loading checkpoints: {e}")
        return

    # 2. Model Size Analysis
    msg = ""
    total_size = 0
    
    components = [
        ("Encoder (Factorizer)", factorizer),
        ("Decoder", decoder),
        ("Semantic Quantizer", sem_vq),
        ("Prosody Quantizer", pro_vq),
        ("Speaker Quantizer", spk_pq),
        ("Entropy Model", entropy_model)
    ]
    
    print("\n📦 Model Weights Sizes:")
    print("-" * 60)
    print(f"{'Component':<25} | {'Size (MB)':<10} | {'Params':<12}")
    print("-" * 60)
    
    for name, model in components:
        size_mb = get_model_size_mb(model)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:<25} | {size_mb:>10.2f} | {params:>12,}")
        total_size += size_mb
        
    print("-" * 60)
    print(f"{'TOTAL':<25} | {total_size:>10.2f}")
    print("=" * 60)

    # 3. Bitrate & Intermediate File Analysis
    print("\n💾 Intermediate Compressed Stream Analysis:")
    
    # Load validation sample
    val_ds = PrecomputedFeatureDataset(
        features_dir=config['data']['feature_dir'],
        manifest_path=config['data']['val_manifest'],
        max_frames=500
    )
    sample = val_ds[0]
    features = sample['features'].unsqueeze(0).to(device)
    audio = sample['audio'].unsqueeze(0).to(device)
    duration = audio.shape[-1] / 16000.0
    
    with torch.no_grad():
        sem, pro, spk = factorizer(features)
        sem_z, _, sem_indices = sem_vq(sem)
        pro_z, _, pro_indices = pro_vq(pro)
        spk_z, _, spk_indices = spk_pq(spk)
        
        # Entropy estimation
        sem_bits, pro_bits = entropy_model.estimate_bits(
            sem_indices.view(sem_indices.size(0), -1),
            pro_indices.view(pro_indices.size(0), -1)
        )
        total_bits_entropy = sem_bits.sum() + pro_bits.sum()

    # Calculate Raw Bits (simulating tight packing)
    # FSQ levels: [6, 6, 6, 6, 6, 6] -> ceil(log2(6)) = 3 bits per index
    bits_per_index = 3 
    
    sem_idx_flat = sem_indices.view(-1)
    pro_idx_flat = pro_indices.view(-1)
    
    num_sem_indices = sem_idx_flat.numel()
    num_pro_indices = pro_idx_flat.numel()
    
    raw_sem_bits = num_sem_indices * bits_per_index
    raw_pro_bits = num_pro_indices * bits_per_index
    
    # Speaker bits (one-time header)
    spk_bits = spk_indices.numel() * 8 # Assuming 8-bit index for speaker PQ
    
    total_raw_bits_stream = raw_sem_bits + raw_pro_bits
    
    print("-" * 60)
    print(f"Audio Duration: {duration:.2f} sec")
    print(f"Semantic Tokens: {num_sem_indices} ({num_sem_indices/duration:.1f} Hz)")
    print(f"Prosody Tokens:  {num_pro_indices} ({num_pro_indices/duration:.1f} Hz)")
    print("-" * 60)
    
    print(f"\n1. Raw Packed Bits (No Entropy Coding):")
    print(f"   Semantic: {raw_sem_bits} bits")
    print(f"   Prosody:  {raw_pro_bits} bits")
    print(f"   Total Stream: {total_raw_bits_stream} bits")
    print(f"   👉 BPS: {total_raw_bits_stream / duration:.1f} bps")
    
    print(f"\n2. Entropy Coded Size (Theoretical Minimum):")
    print(f"   Total Bits: {total_bits_entropy.item():.1f} bits")
    print(f"   👉 BPS: {total_bits_entropy.item() / duration:.1f} bps")
    
    print(f"\n3. Speaker Header (One-time):")
    print(f"   Size: {spk_bits} bits ({spk_bits/8:.0f} bytes)")
    
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_model_footprint.py <config> <checkpoint_dir>")
        sys.exit(1)
        
    analyze_footprint(sys.argv[1], sys.argv[2])
