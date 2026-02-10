#!/usr/bin/env python3
"""
Check Model Sizes
Prints parameter count and estimated size for all SIREN components.
"""
import torch
import torch.nn as nn
import yaml
import os
import sys

# Add src
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ultra_low_bitrate_codec.models.micro_hubert import MicroHuBERT
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def get_params(model):
    return sum(p.numel() for p in model.parameters())

def format_size(params, bit_width=32):
    size_mb = (params * bit_width / 8) / (1024 * 1024)
    return f"{params/1e6:.2f}M params | {size_mb:.2f} MB (FP{bit_width})"

def main():
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    print("=" * 60)
    print("SIREN v2 - Model Component Analysis")
    print("=" * 60)
    
    total_params = 0
    inference_params = 0
    
    # 1. MicroHuBERT
    micro = MicroHuBERT()
    p = get_params(micro)
    total_params += p
    inference_params += p
    print(f"1. MicroHuBERT (Feature Extractor): {format_size(p)}")
    
    # 2. Factorizer
    factorizer = InformationFactorizerV2(config)
    p = get_params(factorizer)
    total_params += p
    inference_params += p
    print(f"2. Factorizer (Encoder):            {format_size(p)}")
    
    # 3. Quantizers
    # Sem
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8)
    # Pro
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8)
    # Spk
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group'])
    
    p_q = get_params(sem_vq) + get_params(pro_vq) + get_params(spk_pq)
    total_params += p_q
    inference_params += p_q
    print(f"3. Quantizers (FSQ/PQ):             {format_size(p_q)}")
    
    print("-" * 60)
    print("   Bitstream Transmission (~200 bps)")
    print("-" * 60)
    
    # 4. Fuser
    fuser = ConditionFuserV2(
        sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512,
        sem_upsample=4, pro_upsample=8
    )
    p = get_params(fuser)
    total_params += p
    inference_params += p
    print(f"4. Fuser (Conditioning):            {format_size(p)}")
    
    # 5. Flow Model
    # Override config for Flow (80 bands)
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow = ConditionalFlowMatching(config)
    p = get_params(flow)
    total_params += p
    inference_params += p
    print(f"5. Flow Matching (Generator):       {format_size(p)}")
    
    # 6. Vocoder
    vocoder = MelVocoderBitNet()
    p = get_params(vocoder)
    total_params += p
    inference_params += p
    print(f"6. MelVocoder (Audio Synth):        {format_size(p)}")
    
    print("=" * 60)
    print(f"TOTAL INFERENCE PIPELINE:           {format_size(inference_params)}")
    print(f"TOTAL (FP16/BF16 Optimized):        {inference_params/1e6:.2f}M | {(inference_params*2/8)/(1024*1024):.2f} MB")
    print(f"TOTAL (INT8 Quantized ~):           {inference_params/1e6:.2f}M | {(inference_params*1/8)/(1024*1024):.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
