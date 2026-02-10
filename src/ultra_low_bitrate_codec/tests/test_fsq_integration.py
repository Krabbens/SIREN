import torch
import yaml
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from ultra_low_bitrate_codec.models.quantizers import FSQ
from ultra_low_bitrate_codec.models.encoder import SpeechEncoder
from ultra_low_bitrate_codec.models.decoder import SpeechDecoder

def test_fsq_integration():
    print("Testing FSQ Integration...")
    
    # Mock Config
    config = {
        'model': {
            'hubert_model': 'facebook/hubert-base-ls960',
            'hubert_layer': 9,
            'freeze_hubert': True,
            'quantizer_type': 'fsq',
            'fsq_levels': [8, 5, 5, 5],
            'semantic': {
                'hidden_dim': 128,
                'output_dim': 4, # Matches len(fsq_levels)
                'temporal_compression': 4,
                'vocab_size': 1000
            },
            'prosody': {
                'hidden_dim': 64,
                'output_dim': 4,
                'temporal_compression': 8,
                'vocab_size': 1000
            },
            'speaker': {
                'embedding_dim': 256,
                'num_groups': 8,
                'codes_per_group': 256
            },
            'decoder': {
                'fusion_layers': 2,
                'fusion_dim': 256,
                'fusion_heads': 4
            },
            'vocoder': {
                'type': 'hifigan',
                'pretrained': False
            },
            'entropy': {
                'lm_dim': 64,
                'lm_layers': 2,
                'lm_heads': 2 # Check if used
            }
        }
    }
    
    # 1. Instantiate FSQ
    levels = config['model']['fsq_levels']
    fsq = FSQ(levels)
    print(f"FSQ initialized with levels {levels}, dim={fsq.dim}, vocab={fsq.vocab_size}")
    
    # 2. Test Encoder Output -> FSQ
    B, T, D = 2, 100, 768 # Dummy HuBERT feats
    # Note: Encoder expects raw audio usually, but we can test factorizer directly?
    # Let's test full encoder
    # Mock Feature Extractor to return dummy
    
    # 3. Test FSQ Forward
    # Fake input (B, T, 4)
    z = torch.randn(2, 50, 4)
    z_q, loss, indices = fsq(z)
    
    print(f"Input: {z.shape}")
    print(f"Output: {z_q.shape}, Indices: {indices.shape}")
    
    assert z_q.shape == z.shape
    assert indices.shape == (2, 50)
    assert indices.max() < 1000
    
    # 4. Test Decoder Input (from FSQ)
    # Decoder expects (sem_z, pro_z, spk_z)
    decoder = SpeechDecoder(config)
    
    sem_z = torch.randn(2, 25, 4) # 100/4
    pro_z = torch.randn(2, 12, 4) # 100/8 approx
    spk_z = torch.randn(2, 256)
    
    out = decoder(sem_z, pro_z, spk_z)
    print(f"Decoder Output: {out.shape}")
    
    # Check if forward pass works
    assert out is not None
    print("Integration Test Passed!")

if __name__ == "__main__":
    test_fsq_integration()
