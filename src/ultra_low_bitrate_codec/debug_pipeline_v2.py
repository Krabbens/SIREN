#!/usr/bin/env python3
"""
Debug Pipeline for Flow Matching V2.
Visualizes every intermediate step to diagnose quality issues.
"""
import os
import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
import yaml
import soundfile as sf
import numpy as np

# Avoid torchcodec issues
sys.modules['torchcodec'] = type(sys)('torchcodec')
sys.modules['torchcodec.decoders'] = type(sys)('torchcodec.decoders')

from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.tiny_hubert import TinyHubert
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def save_plot(data, title, path, range_min=None, range_max=None):
    plt.figure(figsize=(12, 4))
    if data.ndim == 3:
        data = data.squeeze()
    if data.shape[0] > data.shape[1]: # Transpose if needed (C, T) -> (T, C)
        data = data.T
        
    plt.imshow(data, origin='lower', aspect='auto', cmap='viridis', vmin=range_min, vmax=range_max)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    base_dir = "/home/sperm/siren/SIREN"
    input_wav = f"{base_dir}/data/jakubie_16k.wav"
    output_dir = f"{base_dir}/outputs/debug_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Config
    with open(f"{base_dir}/src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml") as f:
        config = yaml.safe_load(f)
        
    # 1. Load Audio & Compute GT Mel
    print("\n--- Step 1: Input Audio & GT Mel ---")
    wav_data, sr = sf.read(input_wav)
    wav = torch.tensor(wav_data, dtype=torch.float32).to(device)
    if wav.dim() == 1: wav = wav.unsqueeze(0)
    else: wav = wav.T
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
        wav = resampler(wav)
        
    wav = wav / (wav.abs().max() + 1e-6)
    
    # Compute GT Mel for comparison
    n_fft = 1024
    hop_length = 320
    n_mels = 80
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=False
    ).to(device)
    
    gt_mel = mel_transform(wav)
    gt_mel = torch.log10(gt_mel + 1e-5)
    # Normalize with training stats
    gt_mel_norm = (gt_mel - (-5.0)) / 2.0
    
    save_plot(gt_mel.cpu().numpy(), "Ground Truth Mel (Log10)", f"{output_dir}/01_gt_mel.png")
    save_plot(gt_mel_norm.cpu().numpy(), "Ground Truth Mel (Normalized)", f"{output_dir}/01_gt_mel_norm.png", -4, 4)

    # 2. TinyHubert Features
    print("\n--- Step 2: TinyHubert Features ---")
    tiny_hubert = TinyHubert(out_dim=768, hidden_dim=384, num_layers=12).to(device)
    tiny_hubert.load_state_dict(torch.load(f"{base_dir}/checkpoints/tiny_hubert_best.pt", map_location=device))
    tiny_hubert.eval()
    
    with torch.no_grad():
        features = tiny_hubert(wav) # (B, T, 768)
    
    save_plot(features.cpu().numpy().T, "TinyHubert Features", f"{output_dir}/02_hubert_features.png")
    
    # 3. Factorization & Quantization
    print("\n--- Step 3: Factorization & Quantization ---")
    factorizer = InformationFactorizerV2(config).to(device)
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device)
    
    ckpt_path = f"{base_dir}/checkpoints/checkpoints_stable/step_87000"
    factorizer.load_state_dict(torch.load(f"{ckpt_path}/factorizer.pt", map_location=device), strict=False)
    sem_vq.load_state_dict(torch.load(f"{ckpt_path}/sem_rfsq.pt", map_location=device), strict=False)
    pro_vq.load_state_dict(torch.load(f"{ckpt_path}/pro_rfsq.pt", map_location=device), strict=False)
    spk_pq.load_state_dict(torch.load(f"{ckpt_path}/spk_pq.pt", map_location=device), strict=False)
    
    with torch.no_grad():
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
    save_plot(sem_z.cpu().numpy().T, "Semantic Latents (Quantized)", f"{output_dir}/03_sem_latents.png")
    save_plot(pro_z.cpu().numpy().T, "Prosody Latents (Quantized)", f"{output_dir}/03_pro_latents.png")
    
    # 4. Fuser V2 Output (Conditioning)
    print("\n--- Step 4: Fuser V2 Conditioning ---")
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    # Load latest available fuser
    fuser_path = f"{base_dir}/checkpoints/checkpoints_flow_v2/fuser_epoch20.pt" # Adjust based on latest
    fuser.load_state_dict(torch.load(fuser_path, map_location=device), strict=False)
    fuser.eval()
    
    target_len = gt_mel.shape[2]
    with torch.no_grad():
        cond = fuser(sem_z, pro_z, spk_z, target_len) # (B, T, 512)
        
    save_plot(cond.cpu().numpy().squeeze().T, "Fused Conditioning (512-dim)", f"{output_dir}/04_conditioning.png")
    
    # 5. Flow Matching Generation
    print("\n--- Step 5: Flow Matching Generation ---")
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    flow_path = f"{base_dir}/checkpoints/checkpoints_flow_v2/flow_epoch20.pt" # Adjust based on latest
    flow_model.load_state_dict(torch.load(flow_path, map_location=device), strict=False)
    flow_model.eval()
    
    with torch.no_grad():
        gen_mel = flow_model.solve_ode(cond, steps=50, solver='euler') # (B, T, 80)
        
    # Denormalize
    gen_mel_denorm = gen_mel * 2.0 + (-5.0)
    gen_mel_denorm = torch.clamp(gen_mel_denorm, min=-12.0, max=3.0)
    
    save_plot(gen_mel.cpu().numpy().squeeze().T, "Generated Mel (Raw Output)", f"{output_dir}/05_gen_mel_raw.png", -4, 4)
    save_plot(gen_mel_denorm.cpu().numpy().squeeze().T, "Generated Mel (Denormalized)", f"{output_dir}/05_gen_mel_denorm.png", -12, 3)
    
    # 6. Vocoder Synthesis
    print("\n--- Step 6: Vocoder Synthesis ---")
    vocoder = MelVocoderBitNet().to(device)
    voc_path = f"{base_dir}/checkpoints/vocoder_mel/vocoder_latest.pt"
    vocoder.load_state_dict(torch.load(voc_path, map_location=device))
    vocoder.eval()
    
    with torch.no_grad():
        audio = vocoder(gen_mel) # Vocoder expects NORMALIZED input (-1 to 1 basically, but trained on norm stats)
        # Note: Check if vocoder expects denormalized or normalized input. 
        # Usually vocoders are trained on GT Mel which was normalized.
        # Let's try both if needed, but standard is using the normalized input directly.
        
    sf.write(f"{output_dir}/06_output_audio.wav", audio.squeeze().cpu().numpy(), 16000)
    print(f"Saved {output_dir}/06_output_audio.wav")
    
    # Create comparison image
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # GT
    im1 = axes[0].imshow(gt_mel_norm.squeeze().cpu().numpy(), origin='lower', aspect='auto', cmap='viridis', vmin=-4, vmax=4)
    axes[0].set_title("Ground Truth Mel")
    fig.colorbar(im1, ax=axes[0])
    
    # Cond
    im2 = axes[1].imshow(cond.squeeze().cpu().numpy().T[:80], origin='lower', aspect='auto', cmap='viridis')
    axes[1].set_title("Conditioning (First 80 dims)")
    fig.colorbar(im2, ax=axes[1])
    
    # Gen
    im3 = axes[2].imshow(gen_mel.squeeze().cpu().numpy().T, origin='lower', aspect='auto', cmap='viridis', vmin=-4, vmax=4)
    axes[2].set_title("Generated Mel (Flow Output)")
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_comparison_summary.png")
    print(f"Saved {output_dir}/07_comparison_summary.png")

if __name__ == "__main__":
    main()
