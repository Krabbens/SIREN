"""
End-to-End Pipeline Visualization with TinyDiffusion Enhancement.
MicroEncoder -> Adapter -> BitVocoder -> TinyDiffusion Enhancement

Generates 3-panel spectrogram comparison:
1. Original Audio
2. Reconstructed (raw)
3. Reconstructed + TinyDiffusion (enhanced)
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
from ultra_low_bitrate_codec.models.tiny_diffusion import TinyDiffusionEnhancer

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_path = "jakubie_16k.wav"
    micro_ckpt = "checkpoints_micro_encoder/best_model.pt"
    vocoder_ckpt = "checkpoints_bitnet/best_model.pt"
    adapter_ckpt = "checkpoints_adapter/checkpoint_epoch9.pt"  # Use older non-upsampler version
    diffusion_ckpt = "checkpoints_diffusion/best_model.pt"
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
    voc_conf = config['model']['vocoder']
    vocoder = BitVocoder(
        input_dim=512,
        dim=256,
        n_fft=1024,
        hop_length=320,
        num_layers=voc_conf.get('num_convnext_layers', 8),
        num_res_blocks=voc_conf.get('num_res_blocks', 3)
    ).to(device)
    
    voc_ckpt_data = torch.load(vocoder_ckpt, map_location=device)
    state = voc_ckpt_data['model_state_dict']
    new_state = {k[6:] if k.startswith('model.') else k: v for k, v in state.items()}
    vocoder.load_state_dict(new_state, strict=False)
    vocoder.eval()
    print("Vocoder loaded.")
    
    # 4. Load Adapter (without upsampler)
    print("Loading Adapter...")
    adapter = FeatureAdapter(in_dim=768, out_dim=512, upsample_factor=1).to(device)
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device), strict=False)
    adapter.eval()
    
    # 5. Load TinyDiffusion
    print("Loading TinyDiffusion Enhancer...")
    diffusion = TinyDiffusionEnhancer(n_mels=80, hidden_dim=64).to(device)
    diffusion.load_state_dict(torch.load(diffusion_ckpt, map_location=device))
    diffusion.eval()
    
    # 6. Inference pipeline
    print("Running inference...")
    with torch.no_grad():
        # Encode
        features = encoder(audio_batch)
        
        # Adapt
        adapted_feats = adapter(features)
        
        # Vocode
        audio_rec = vocoder(adapted_feats)
        
    # 7. Mel Spectrograms
    print("Generating spectrograms...")
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, win_length=400, hop_length=320, n_mels=80, power=2.0
    ).to(device)
    
    db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
    
    def audio_to_mel_db(wav):
        wav = wav.float()
        if wav.abs().max() > 1.0:
            wav = wav / wav.abs().max()
        mel = mel_transform(wav)
        return db_transform(mel)
    
    # Get mels
    orig_mel_db = audio_to_mel_db(audio_batch)
    rec_mel_db = audio_to_mel_db(audio_rec)
    
    # 8. Apply TinyDiffusion enhancement on rec mel
    print("Applying TinyDiffusion enhancement...")
    with torch.no_grad():
        # TinyDiffusion expects log mel (not dB scaled)
        rec_mel_log = torch.log(mel_transform(audio_rec.float()) + 1e-5)
        enhanced_mel_log = diffusion.sample(rec_mel_log, num_steps=4)
        
        # Convert back to dB for visualization
        enhanced_mel = torch.exp(enhanced_mel_log)
        enhanced_mel_db = db_transform(enhanced_mel)
    
    # 9. Plot 3-panel comparison
    orig_spec = orig_mel_db.squeeze(0).cpu().numpy()
    rec_spec = rec_mel_db.squeeze(0).cpu().numpy()
    enh_spec = enhanced_mel_db.squeeze(0).cpu().numpy()
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    vmin, vmax = -80, 0
    
    im1 = axs[0].imshow(orig_spec, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='magma')
    axs[0].set_title("Original Audio")
    axs[0].set_ylabel("Mel Freq")
    fig.colorbar(im1, ax=axs[0], format='%+2.0f dB')
    
    im2 = axs[1].imshow(rec_spec, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='magma')
    axs[1].set_title("Reconstructed (MicroEncoder → Adapter → BitVocoder)")
    axs[1].set_ylabel("Mel Freq")
    fig.colorbar(im2, ax=axs[1], format='%+2.0f dB')
    
    im3 = axs[2].imshow(enh_spec, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='magma')
    axs[2].set_title("Enhanced (+ TinyDiffusion)")
    axs[2].set_ylabel("Mel Freq")
    axs[2].set_xlabel("Time (frames)")
    fig.colorbar(im3, ax=axs[2], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig("end_to_end_spectrogram.png", dpi=150)
    print("Saved end_to_end_spectrogram.png")

if __name__ == "__main__":
    main()

