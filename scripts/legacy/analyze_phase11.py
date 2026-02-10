#!/usr/bin/env python3
"""
Phase 11 Full Quality Analysis
Generates: Audio, Spectrograms (GT vs Predicted), MOS Metrics
"""
import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import yaml
import soundfile as sf
import numpy as np
from pesq import pesq
from pystoi import stoi

# Models
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet
from ultra_low_bitrate_codec.models.micro_transformer import MicroTransformer

def compute_mel(wav, sr=16000, n_fft=1024, hop_length=320, n_mels=80):
    """Compute log-mel spectrogram"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel = mel_transform(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel

def plot_comparison(gt_mel, pred_mel, output_path):
    """Plot GT vs Predicted spectrograms side by side"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    vmin, vmax = -11.5, 3.0
    
    axes[0].imshow(gt_mel, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth Mel Spectrogram', fontsize=14)
    axes[0].set_ylabel('Mel Bins')
    
    axes[1].imshow(pred_mel, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Phase 11 Prediction (Student + Finetuned Flow)', fontsize=14)
    axes[1].set_ylabel('Mel Bins')
    axes[1].set_xlabel('Time Frames')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison: {output_path}")

def compute_metrics(ref_wav, deg_wav, sr=16000):
    """Compute PESQ and STOI"""
    # Ensure same length
    min_len = min(len(ref_wav), len(deg_wav))
    ref_wav = ref_wav[:min_len]
    deg_wav = deg_wav[:min_len]
    
    # PESQ (wb = wideband)
    try:
        pesq_score = pesq(sr, ref_wav, deg_wav, 'wb')
    except Exception as e:
        print(f"PESQ Error: {e}")
        pesq_score = -1
    
    # STOI
    try:
        stoi_score = stoi(ref_wav, deg_wav, sr, extended=False)
    except Exception as e:
        print(f"STOI Error: {e}")
        stoi_score = -1
    
    return pesq_score, stoi_score

def main():
    input_file = "data/jakubie.wav"
    output_dir = "outputs/phase11_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("PHASE 11 FULL QUALITY ANALYSIS")
    print("=" * 70)
    
    # ========== Load Models ==========
    print("\n[1/5] Loading Models...")
    
    # Student
    student = MicroTransformer(hidden_dim=384, num_layers=8).to(device)
    ckpt_s = torch.load("checkpoints/microtransformer_v2/microtransformer_ep75.pt", map_location=device)
    student.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_s.items()}, strict=False)
    student.eval()
    print("   ✓ Student (MicroTransformer EP75)")
    
    # Factorizer
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path) as f: config = yaml.safe_load(f)
    
    factorizer = InformationFactorizerV2(config).to(device)
    fq_path = "checkpoints/factorizer_microhubert_finetune_v4/factorizer.pt"
    ckpt_f = torch.load(fq_path, map_location=device)
    if isinstance(ckpt_f, dict) and 'model_state_dict' in ckpt_f: ckpt_f = ckpt_f['model_state_dict']
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_f.items()}, strict=False)
    factorizer.eval()
    print("   ✓ Factorizer (V4)")
    
    # Fuser
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fu_ckpt = torch.load("checkpoints/checkpoints_flow_v2/fuser_epoch31.pt", map_location=device)
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fu_ckpt.items()})
    fuser.eval()
    print("   ✓ Fuser (Epoch 31)")
    
    # Flow (Finetuned)
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    fl_ckpt = torch.load("checkpoints/flow_finetune_student/flow_ft_epoch50.pt", map_location=device)
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fl_ckpt.items()}, strict=False)
    flow_model.eval()
    print("   ✓ Flow (Finetuned EP50)")
    
    # Quantizers
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    # Vocoder
    vocoder = MelVocoderBitNet().to(device)
    voc_ckpt = torch.load("checkpoints/vocoder_mel/vocoder_latest.pt", map_location=device)
    vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in voc_ckpt.items()})
    vocoder.eval()
    print("   ✓ Vocoder")
    
    # ========== Load Audio ==========
    print(f"\n[2/5] Loading Audio: {input_file}")
    wav_orig, sr = sf.read(input_file)
    wav_orig = torch.tensor(wav_orig, dtype=torch.float32)
    if wav_orig.dim() > 1: wav_orig = wav_orig.mean(dim=0)
    if sr != 16000:
        wav_orig = torchaudio.functional.resample(wav_orig, sr, 16000)
        sr = 16000
    
    print(f"   Duration: {len(wav_orig)/sr:.2f}s, Samples: {len(wav_orig)}")
    
    # ========== Compute Ground Truth Mel ==========
    print("\n[3/5] Computing Ground Truth Mel...")
    gt_mel = compute_mel(wav_orig.unsqueeze(0))  # (1, 80, T)
    print(f"   GT Mel shape: {gt_mel.shape}")
    
    # ========== Run Pipeline ==========
    print("\n[4/5] Running Phase 11 Pipeline...")
    wav_norm = (wav_orig - wav_orig.mean()) / (wav_orig.std() + 1e-6)
    target_mel_len = wav_orig.shape[0] // 320
    
    with torch.no_grad():
        # Student
        features = student(wav_norm.unsqueeze(0).to(device))
        print(f"   Student features: {features.shape}")
        
        # Factorizer
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        print(f"   Semantic: {sem_z.shape}, Prosody: {pro_z.shape}, Speaker: {spk_z.shape}")
        
        # Fuser
        cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
        print(f"   Conditioning: {cond.shape}")
        
        # Flow
        mel_pred = flow_model.solve_ode(cond, steps=50, solver='rk4', cfg_scale=1.0)
        mel_pred = mel_pred * 3.5 - 5.0  # Denormalize
        mel_pred = torch.clamp(mel_pred, min=-11.5, max=4.0)
        print(f"   Predicted Mel: {mel_pred.shape}")
        
        # Vocoder
        audio_out = vocoder(mel_pred)
        audio_np = audio_out.squeeze().cpu().numpy()
    
    # ========== Save Outputs ==========
    print("\n[5/5] Saving Results & Computing Metrics...")
    
    # Save audio
    audio_path = f"{output_dir}/phase11_jakubie.wav"
    sf.write(audio_path, audio_np, 16000)
    print(f"   Saved: {audio_path}")
    
    # Plot comparison
    gt_mel_np = gt_mel.squeeze().numpy()
    pred_mel_np = mel_pred.squeeze().T.cpu().numpy()  # Transpose to (80, T)
    
    # Align lengths for comparison
    min_t = min(gt_mel_np.shape[1], pred_mel_np.shape[1])
    gt_mel_np = gt_mel_np[:, :min_t]
    pred_mel_np = pred_mel_np[:, :min_t]
    
    comparison_path = f"{output_dir}/spectrogram_comparison.png"
    plot_comparison(gt_mel_np, pred_mel_np, comparison_path)
    
    # Compute metrics
    ref_np = wav_orig.numpy()
    deg_np = audio_np
    
    # Ensure same length
    min_len = min(len(ref_np), len(deg_np))
    ref_np = ref_np[:min_len]
    deg_np = deg_np[:min_len]
    
    pesq_score, stoi_score = compute_metrics(ref_np, deg_np)
    
    # Compute Mel MSE
    mel_mse = np.mean((gt_mel_np - pred_mel_np) ** 2)
    
    # ========== Print Report ==========
    print("\n" + "=" * 70)
    print("PHASE 11 QUALITY REPORT")
    print("=" * 70)
    print(f"Input:  {input_file}")
    print(f"Output: {audio_path}")
    print("-" * 70)
    print(f"PESQ (WB):      {pesq_score:.3f}  (Range: 1.0-4.5, Higher=Better)")
    print(f"STOI:           {stoi_score:.3f}  (Range: 0.0-1.0, Higher=Better)")
    print(f"Mel MSE:        {mel_mse:.4f}  (Lower=Better)")
    print("-" * 70)
    
    # Quality interpretation
    if pesq_score >= 3.5:
        quality = "EXCELLENT"
    elif pesq_score >= 3.0:
        quality = "GOOD"
    elif pesq_score >= 2.5:
        quality = "FAIR"
    elif pesq_score >= 2.0:
        quality = "POOR"
    else:
        quality = "BAD"
    
    print(f"Overall Quality: {quality}")
    print("=" * 70)
    
    # Save report
    report_path = f"{output_dir}/report.txt"
    with open(report_path, 'w') as f:
        f.write("PHASE 11 QUALITY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input:  {input_file}\n")
        f.write(f"Output: {audio_path}\n\n")
        f.write(f"PESQ (WB):      {pesq_score:.3f}\n")
        f.write(f"STOI:           {stoi_score:.3f}\n")
        f.write(f"Mel MSE:        {mel_mse:.4f}\n\n")
        f.write(f"Overall Quality: {quality}\n")
    
    print(f"\nReport saved: {report_path}")
    
    return {
        'pesq': pesq_score,
        'stoi': stoi_score,
        'mel_mse': mel_mse,
        'quality': quality,
        'audio_path': audio_path,
        'comparison_path': comparison_path
    }

if __name__ == "__main__":
    main()
