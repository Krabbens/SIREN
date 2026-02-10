import torch
import unittest
import sys
import os
import yaml
import json

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultra_low_bitrate_codec.models.encoder import SpeechEncoder
from ultra_low_bitrate_codec.models.decoder import SpeechDecoder
from ultra_low_bitrate_codec.training.trainer import Trainer

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        # Create a dummy config
        self.config = {
            'model': {
                'hubert_model': 'facebook/hubert-base-ls960',
                'hubert_layer': 9,
                'freeze_hubert': True,
                'semantic': {
                    'input_dim': 768, 'hidden_dim': 128, 'output_dim': 32,
                    'temporal_compression': 4, 'vocab_size': 64
                },
                'prosody': {
                    'input_dim': 768, 'hidden_dim': 64, 'output_dim': 16,
                    'temporal_compression': 8, 'vocab_size': 64
                },
                'speaker': {
                    'embedding_dim': 256, 'num_groups': 8, 'codes_per_group': 256
                },
                'entropy': {
                    'enabled': True, 'lm_layers': 2, 'lm_dim': 128, 'lm_heads': 4, 'context_length': 64
                },
                'decoder': {
                    'fusion_layers': 2, 'fusion_dim': 256, 'fusion_heads': 4
                },
                'vocoder': {
                    'type': 'hifigan', 'pretrained': False
                }
            },
            'training': {
                'batch_size': 2, 'learning_rate': 1e-4, 'warmup_steps': 100,
                'max_steps': 10, 'commitment_weight': 0.25,
                'codebook_ema_decay': 0.99, 'reconstruction_weight': 1.0,
                'perceptual_weight': 0.1, 'speaker_weight': 0.5,
                'log_every': 1, 'eval_every': 10, 'save_every': 10
            },
            'data': {
                'train_manifest': 'dummy_manifest.json',
                'val_manifest': 'dummy_manifest.json',
                'sample_rate': 16000, 'max_duration': 1.0, 'min_duration': 0.5
            },
            'audio': {
                'sample_rate': 16000, 'hop_length': 320
            }
        }
        
        # Save dummy config
        config_path = '/tmp/test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
            
    def test_pipeline_forward(self):
        print("Initializing Trainer...")
        # We need to mock AudioDataset loading or just instantiate models manually?
        # Trainer init tries to load dataset. Let's create a dummy manifest.
        with open('dummy_manifest.json', 'w') as f:
            json.dump([{'path': '/tmp/dummy.wav'}], f) # Path doesn't matter if we mock dl or don't run loop
            
        # Mock dataset loading to avoid file IO error
        # We can just instantiate models directly for this test
        device = 'cpu'
        
        encoder = SpeechEncoder(self.config).to(device)
        decoder = SpeechDecoder(self.config).to(device)
        
        # Helper to bypass HuBERT loading for speed? 
        # No, we need to verify shapes. But HuBERT model download might be slow.
        # Assuming we have internet access or cached model.
        # If not, we should mock feature extractor output.
        
        # Mock Feature Extractor to return random tensors
        encoder.feature_extractor = torch.nn.Identity() # Hack
        # But `feature_extractor` expects audio and returns features.
        # Let's override the forward method of feature_extractor
        encoder.feature_extractor.forward = lambda x: torch.randn(x.shape[0], 50, 768) # 50 frames
        
        dummy_audio = torch.randn(2, 16000) # 1 sec batch
        
        print("Running Forward Pass...")
        sem, pro, spk = encoder(dummy_audio)
        
        print(f"Semantic: {sem.shape}, Prosody: {pro.shape}, Speaker: {spk.shape}")
        
        # Assert Shapes
        # 50 frames / 4 = 12.5 -> 12 frames
        # 50 frames / 8 = 6.25 -> 6 frames
        self.assertEqual(sem.shape[1], 12)
        self.assertEqual(pro.shape[1], 6)
        self.assertEqual(spk.shape[1], 256)
        
        # Decoder
        print("Running Decoder...")
        # Simulate quantization (pass thru)
        out_audio = decoder(sem, pro, spk)
        print(f"Output Audio: {out_audio.shape}")
        
        # Check Output Length
        # Decoder upsamples:
        # Semantic 12 * 4 = 48
        # Prosody 6 * 8 = 48
        # Vocoder: 48 * 320 = 15360
        self.assertEqual(out_audio.shape[1], 48 * 320)
        
if __name__ == '__main__':
    unittest.main()
