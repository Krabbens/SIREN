"""
Test inference of MicroEncoder vs HuBERT.
Loads best_model.pt and compares feature cosine similarity on a random test file.
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import glob
import random
import matplotlib.pyplot as plt
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoder

def load_hubert(device):
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    print("Loading HuBERT...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()
    return processor, model

@torch.no_grad()
def extract_hubert_features(audio, processor, hubert, device, layer=9):
    inputs = processor(
        audio.cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(device)
    outputs = hubert(input_values, output_hidden_states=True)
    return outputs.hidden_states[layer]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load MicroEncoder
    print("Loading MicroEncoder...")
    model = MicroEncoder().to(device)
    checkpoint = torch.load("checkpoints_micro_encoder/checkpoint_epoch5.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Load HuBERT
    processor, hubert = load_hubert(device)

    # Find audio
    audio_files = glob.glob("data/audio/**/*.wav", recursive=True)
    if not audio_files:
        print("No audio files found in data/audio")
        return

    test_file = random.choice(audio_files)
    print(f"Testing on: {test_file}")

    # Load audio
    audio, sr = torchaudio.load(test_file)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    audio = audio.squeeze(0)

    # Cut to 3 seconds max to match training somewhat
    if len(audio) > 16000 * 3:
        audio = audio[:16000 * 3]

    audio = audio.to(device)
    
    # Run Inference
    # MicroEncoder requires (B, T)
    audio_batch = audio.unsqueeze(0)
    
    # MicroEncoder
    micro_feats = model(audio_batch)
    
    # HuBERT
    hubert_feats = extract_hubert_features(audio_batch.squeeze(0), processor, hubert, device)
    
    # Align lengths
    min_len = min(micro_feats.shape[1], hubert_feats.shape[1])
    micro_feats = micro_feats[:, :min_len]
    hubert_feats = hubert_feats[:, :min_len]
    
    # Compute metrics
    cosine_sim = F.cosine_similarity(micro_feats, hubert_feats, dim=-1).mean()
    l1_dist = F.l1_loss(micro_feats, hubert_feats)
    
    print("\nResults:")
    print(f"Cosine Similarity: {cosine_sim.item():.4f}")
    print(f"L1 Distance:      {l1_dist.item():.4f}")
    
    print("-" * 30)
    print("Sample feature values (first 5 dims, first frame):")
    print("Micro:", micro_feats[0, 0, :5].detach().cpu().numpy())
    print("HuBERT:", hubert_feats[0, 0, :5].detach().cpu().numpy())

if __name__ == "__main__":
    main()
