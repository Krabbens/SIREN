"""
Visualize MicroEncoder features vs HuBERT for a specific file.
Generates a side-by-side spectrogram comparison.
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoder

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_path = "jakubie_16k.wav"
    checkpoint_path = "checkpoints_micro_encoder/best_model.pt" # Using best model
    
    # 1. Load Audio
    print(f"Loading {audio_path}...")
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio.mean(0) # Mono
    
    # Cut to reasonable length for viz (e.g. 5 sec)
    if len(audio) > 16000 * 5:
        audio = audio[:16000 * 5]
    
    audio_batch = audio.unsqueeze(0).to(device) # (1, T)
    
    # 2. Load MicroEncoder
    print("Loading MicroEncoder...")
    model = MicroEncoder().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # 3. Load HuBERT
    print("Loading HuBERT...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    hubert.eval()
    
    # 4. Extract Features
    with torch.no_grad():
        # Micro
        micro_feats = model(audio_batch) # (1, frames, 768)
        
        # HuBERT
        inputs = processor(
            audio_batch.squeeze(0).cpu().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        inputs = inputs.input_values.to(device)
        hubert_out = hubert(inputs, output_hidden_states=True)
        hubert_feats = hubert_out.hidden_states[9] # Layer 9
        
    # 5. Visualize
    print("Generating plot...")
    micro_spec = micro_feats.squeeze(0).transpose(0, 1).cpu().numpy() # (768, T)
    hubert_spec = hubert_feats.squeeze(0).transpose(0, 1).cpu().numpy() # (768, T)
    
    # Ensure same length
    min_len = min(micro_spec.shape[1], hubert_spec.shape[1])
    micro_spec = micro_spec[:, :min_len]
    hubert_spec = hubert_spec[:, :min_len]
    
    # Compute Cosine Sim per frame
    sim = F.cosine_similarity(
        torch.tensor(micro_spec.T), 
        torch.tensor(hubert_spec.T), 
        dim=1
    ).numpy()
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # HuBERT
    im1 = axs[0].imshow(hubert_spec, origin='lower', aspect='auto', interpolation='nearest', vmin=-3, vmax=3)
    axs[0].set_title("Target: HuBERT Features (Layer 9)")
    axs[0].set_ylabel("Dim (768)")
    fig.colorbar(im1, ax=axs[0])
    
    # MicroEncoder
    im2 = axs[1].imshow(micro_spec, origin='lower', aspect='auto', interpolation='nearest', vmin=-3, vmax=3)
    axs[1].set_title("Prediction: MicroEncoder (SnakePhase)")
    axs[1].set_ylabel("Dim (768)")
    fig.colorbar(im2, ax=axs[1])
    
    # Similarity
    axs[2].plot(sim, color='green')
    axs[2].set_title(f"Cosine Similarity (Avg: {sim.mean():.4f})")
    axs[2].set_ylim(0, 1.1)
    axs[2].set_ylabel("Similarity")
    axs[2].set_xlabel("Frame Index (20ms)")
    axs[2].axhline(y=sim.mean(), color='r', linestyle='--', label='Mean')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig("jakubIE_spectrogram_viz.png", dpi=150)
    print(f"Saved visualization to jakubIE_spectrogram_viz.png")
    print(f"Average Similarity: {sim.mean():.4f}")

if __name__ == "__main__":
    main()
