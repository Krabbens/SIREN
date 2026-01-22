"""
End-to-End Pipeline Visualization (MicroEncoder -> Adapter -> BitVocoder).
Generates spectrograms of original vs reconstructed audio.

Using trained FeatureAdapter to bridge MicroEncoder (768-dim) and BitVocoder (512-dim).
"""

import os
import sys
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoder
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.models.adapter import FeatureAdapter

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_path = "jakubie_16k.wav"
    micro_ckpt = "checkpoints_micro_encoder/best_model.pt"
    vocoder_ckpt = "checkpoints_bitnet/best_model.pt"
    adapter_ckpt = "checkpoints_adapter/final_model.pt"
    config_path = "ultra_low_bitrate_codec/configs/sub100bps.yaml"
    
    config = load_config(config_path)
    
    # 1. Load Audio
    print(f"Loading {audio_path}...")
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio.mean(0) # Mono
    
    # Cut to 5 seconds
    if len(audio) > 16000 * 5:
        audio = audio[:16000 * 5]
    
    audio_batch = audio.unsqueeze(0).to(device)
    
    # 2. Load MicroEncoder
    print("Loading MicroEncoder...")
    encoder = MicroEncoder().to(device)
    curr_micro = torch.load(micro_ckpt, map_location=device)
    encoder.load_state_dict(curr_micro['model_state_dict'])
    encoder.eval()
    
    # 3. Load BitVocoder
    print("Loading BitVocoder...")
    # Vocoder config from yaml
    voc_conf = config['model']['vocoder']
    vocoder = BitVocoder(
        input_dim=512, # Hardcoded/Known from previous check
        dim=256,       # Matching checkpoint
        n_fft=1024,
        hop_length=320,
        num_layers=voc_conf.get('num_convnext_layers', 8),
        num_res_blocks=voc_conf.get('num_res_blocks', 3)
    ).to(device)
    
    # Load vocoder weights (which are inside NeuralVocoder wrapper in checkpoint usually, or standalone)
    # Checkpoint structure: 'model_state_dict' keys could be 'model.xxx' or just 'xxx'
    voc_ckpt = torch.load(vocoder_ckpt, map_location=device)
    state = voc_ckpt['model_state_dict']
    
    # Fix keys if needed
    new_state = {}
    for k, v in state.items():
        if k.startswith('model.'):
            new_state[k[6:]] = v
        else:
            new_state[k] = v
    
    # Try loading
    try:
        vocoder.load_state_dict(new_state, strict=False)
        print("Vocoder loaded successfully.")
    except Exception as e:
        print(f"Vocoder load warning: {e}")
    
    vocoder.eval()
    
    # 4. Load Trained Adapter
    print("Loading Adapter...")
    adapter = FeatureAdapter(in_dim=768, out_dim=512).to(device)
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
    adapter.eval()
    
    # 5. Inference pipeline
    with torch.no_grad():
        # Encode
        features = encoder(audio_batch) # (1, frames, 768)
        
        # Trained Adapter: 768 -> 512
        adapted_feats = adapter(features) # (1, frames, 512)
        
        # Vocode
        audio_rec = vocoder(adapted_feats) # (1, samples)
        
    # 5. Spectrograms
    print("Generating spectrograms...")
    
    def get_spec(wav):
        wav = wav.float()
        # Normalize audio to [-1, 1] if not already
        if wav.abs().max() > 1.0:
            wav = wav / wav.abs().max()
            
        spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, 
            n_fft=1024,
            win_length=400, # 25ms window for better time resolution
            hop_length=320, 
            n_mels=80,
            power=2.0
        ).to(device)
        
        spec = spec_transform(wav)
        
        # Convert to dB
        db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        spec_db = db_transform(spec)
        
        return spec_db.squeeze(0).cpu().numpy()
        
    orig_spec = get_spec(audio_batch)
    rec_spec = get_spec(audio_rec)
    
    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Use auto scaling or reasonable dB range
    vmin = -80
    vmax = 0
    
    im1 = axs[0].imshow(orig_spec, origin='lower', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, cmap='magma')
    axs[0].set_title("Original Audio (Mel Spectrogram)")
    axs[0].set_ylabel("Mel Freq")
    fig.colorbar(im1, ax=axs[0], format='%+2.0f dB')
    
    im2 = axs[1].imshow(rec_spec, origin='lower', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, cmap='magma')
    axs[1].set_title("Reconstructed (MicroEncoder -> Adapter -> BitVocoder)")
    axs[1].set_ylabel("Mel Freq")
    axs[1].set_xlabel("Time (frames)")
    fig.colorbar(im2, ax=axs[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig("end_to_end_spectrogram.png", dpi=150)
    print("Saved end_to_end_spectrogram.png")

if __name__ == "__main__":
    main()
