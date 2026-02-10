#!/usr/bin/env python3
"""
Diagnostic: Compare GT conditioning (DistilHuBERT) vs MicroEncoderTiny conditioning.
This will reveal if the bottleneck is in the Encoder or in the Flow model.
"""

import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import yaml
import soundfile as sf
from transformers import AutoModel

from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet
from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoderTiny

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_wav = "data/jakubie.wav"
    output_dir = "outputs/conditioning_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Config
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # =========================================================================
    # Load Models
    # =========================================================================
    print("Loading models...")
    
    # 1. GT Teacher (DistilHuBERT)
    teacher = AutoModel.from_pretrained("ntu-spml/distilhubert").to(device)
    teacher.eval()
    
    # 2. MicroEncoderTiny (our E2E trained encoder)
    micro_encoder = MicroEncoderTiny().to(device)
    # Load the latest checkpoint from FP32 training
    enc_ckpt_path = "checkpoints/microencoder_diagnosis_fp32/encoder_best.pt"
    if os.path.exists(enc_ckpt_path):
        ckpt = torch.load(enc_ckpt_path, map_location=device)
        micro_encoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
        print(f"  Loaded MicroEncoderTiny from {enc_ckpt_path}")
    else:
        print(f"  WARNING: No checkpoint found at {enc_ckpt_path}, using random init!")
    micro_encoder.eval()
    
    # 3. Shared: Factorizer, Fuser, Flow, VQ, Vocoder
    factorizer_dir = "checkpoints/factorizer_microhubert_finetune_v4"
    flow_dir = "checkpoints/checkpoints_flow_v2"
    
    factorizer = InformationFactorizerV2(config).to(device)
    ckpt_f = torch.load(os.path.join(factorizer_dir, "factorizer.pt"), map_location=device)
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_f.items()}, strict=False)
    factorizer.eval()

    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fu_ckpt = torch.load(os.path.join(flow_dir, "fuser_epoch31.pt"), map_location=device)
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fu_ckpt.items()})
    fuser.eval()

    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    fl_ckpt = torch.load(os.path.join(flow_dir, "flow_epoch31.pt"), map_location=device)
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fl_ckpt.items()})
    flow_model.eval()

    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)

    vocoder = MelVocoderBitNet().to(device)
    voc_ckpt = torch.load("checkpoints/vocoder_mel/vocoder_latest.pt", map_location=device)
    vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in voc_ckpt.items()})
    vocoder.eval()
    
    # Mel transform for GT comparison
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
    ).to(device)

    # =========================================================================
    # Load Audio
    # =========================================================================
    wav, sr = sf.read(input_wav)
    wav = torch.tensor(wav, dtype=torch.float32)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav / (wav.abs().max() + 1e-6)
    wav = wav.unsqueeze(0).to(device)
    
    # GT Mel
    gt_mel = mel_transform(wav)
    gt_mel = torch.log(torch.clamp(gt_mel, min=1e-5))
    
    # =========================================================================
    # Pipeline 1: GT Conditioning (DistilHuBERT)
    # =========================================================================
    print("\n--- Pipeline 1: GT Conditioning (DistilHuBERT) ---")
    with torch.no_grad():
        t_out = teacher(wav.squeeze(0).unsqueeze(0), output_hidden_states=True)
        features_gt = t_out.last_hidden_state
        print(f"  GT Features: {features_gt.shape}, mean={features_gt.mean():.3f}, std={features_gt.std():.3f}")
        
        sem, pro, spk = factorizer(features_gt)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        cond_gt = fuser(sem_z, pro_z, spk_z, features_gt.shape[1])
        print(f"  Conditioning: {cond_gt.shape}, mean={cond_gt.mean():.3f}, std={cond_gt.std():.3f}")
        
        mel_gt_pipeline = flow_model.solve_ode(cond_gt, steps=50, solver='rk4', cfg_scale=1.0)
        mel_gt_pipeline = mel_gt_pipeline * 3.5 - 5.0
        
        audio_gt = vocoder(mel_gt_pipeline)
        sf.write(os.path.join(output_dir, "output_gt_conditioning.wav"), audio_gt.squeeze().cpu().numpy(), 16000)
    
    # =========================================================================
    # Pipeline 2: MicroEncoderTiny Conditioning
    # =========================================================================
    print("\n--- Pipeline 2: MicroEncoderTiny Conditioning ---")
    with torch.no_grad():
        features_micro = micro_encoder(wav)
        print(f"  Micro Features: {features_micro.shape}, mean={features_micro.mean():.3f}, std={features_micro.std():.3f}")
        
        sem, pro, spk = factorizer(features_micro)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        cond_micro = fuser(sem_z, pro_z, spk_z, features_micro.shape[1])
        print(f"  Conditioning: {cond_micro.shape}, mean={cond_micro.mean():.3f}, std={cond_micro.std():.3f}")
        
        mel_micro_pipeline = flow_model.solve_ode(cond_micro, steps=50, solver='rk4', cfg_scale=1.0)
        mel_micro_pipeline = mel_micro_pipeline * 3.5 - 5.0
        
        audio_micro = vocoder(mel_micro_pipeline)
        sf.write(os.path.join(output_dir, "output_micro_conditioning.wav"), audio_micro.squeeze().cpu().numpy(), 16000)
    
    # =========================================================================
    # Pipeline 3: MicroEncoderTiny + CFG (scale=2.0)
    # =========================================================================
    print("\n--- Pipeline 3: MicroEncoderTiny + CFG (scale=2.0) ---")
    with torch.no_grad():
        # Re-use features/cond from Pipeline 2
        mel_micro_cfg = flow_model.solve_ode(cond_micro, steps=50, solver='rk4', cfg_scale=2.0)
        mel_micro_cfg = mel_micro_cfg * 3.5 - 5.0
        
        audio_micro_cfg = vocoder(mel_micro_cfg)
        sf.write(os.path.join(output_dir, "output_micro_cfg2.0.wav"), audio_micro_cfg.squeeze().cpu().numpy(), 16000)

    # =========================================================================
    # Comparison Plot
    # =========================================================================
    print("\n--- Generating Comparison Plot ---")
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    
    # GT Mel
    axes[0].imshow(gt_mel[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("Ground Truth Mel Spectrogram")
    axes[0].set_ylabel("Mel Bins")
    
    # GT Conditioning Output
    axes[1].imshow(mel_gt_pipeline[0].transpose(0, 1).cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title("Flow Output (GT Cond, CFG=1.0)")
    axes[1].set_ylabel("Mel Bins")
    
    # MicroEncoderTiny Conditioning Output
    axes[2].imshow(mel_micro_pipeline[0].transpose(0, 1).cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title("Flow Output (Micro Cond, CFG=1.0)")
    axes[2].set_ylabel("Mel Bins")

    # MicroEncoderTiny CFG Output
    axes[3].imshow(mel_micro_cfg[0].transpose(0, 1).cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[3].set_title("Flow Output (Micro Cond, CFG=2.0)")
    axes[3].set_ylabel("Mel Bins")
    axes[3].set_xlabel("Time Frames")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conditioning_comparison.png"), dpi=150)
    print(f"\nSaved comparison to {output_dir}/conditioning_comparison.png")
    print(f"Audio files: output_gt_conditioning.wav, output_micro_conditioning.wav")

if __name__ == "__main__":
    main()
