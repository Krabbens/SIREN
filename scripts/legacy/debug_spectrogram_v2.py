
import os
import sys
# Add CWD and CWD/src to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

import torch
import torch.nn.functional as F
import soundfile as sf
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchaudio

# Reuse load_models from inference_v2 (assuming it's importable or I copy it. I'll copy for safety)
# Imports
from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoderTiny, MicroEncoder
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder

def load_models(checkpoint_dir, config_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    print(f"Loading checkpoints from {checkpoint_dir}...")
    
    # Auto-detect latest epoch (Epoch 30 specifically)
    tag = "ep30"
    
    # --- 1. Encoder (Auto-detect Tiny vs Standard) ---
    enc_ckpt_path = os.path.join(checkpoint_dir, f"encoder_{tag}.pt")
    if not os.path.exists(enc_ckpt_path):
        # Fallback to whatever is there if ep30 missing (unlikely since we just ran inference)
        import glob
        files = glob.glob(os.path.join(checkpoint_dir, "encoder_ep*.pt"))
        if files:
            tag = f"ep{max([int(f.split('ep')[-1].split('.')[0]) for f in files])}"
            enc_ckpt_path = os.path.join(checkpoint_dir, f"encoder_{tag}.pt")
    
    enc_ckpt = torch.load(enc_ckpt_path, map_location='cpu')
    enc_state = {k.replace("_orig_mod.", ""): v for k, v in enc_ckpt.items()}
    
    is_tiny = False
    if 'output_proj.weight' in enc_state:
         if enc_state['output_proj.weight'].shape[1] == 128:
             is_tiny = True
    
    if is_tiny:
        encoder = MicroEncoderTiny(hidden_dim=128, output_dim=768, num_layers=2).to(device)
    else:
        encoder = MicroEncoder(hidden_dim=256, output_dim=768, num_layers=4).to(device)
    encoder.load_state_dict(enc_state)
    
    # --- 2. Factorizer (V2 Config: Prosody 4D) ---
    PRO_DIM = 4 
    config['model']['prosody']['output_dim'] = PRO_DIM
    factorizer = InformationFactorizerV2(config).to(device)
    fac_ckpt = torch.load(os.path.join(checkpoint_dir, f"factorizer_{tag}.pt"), map_location=device)
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fac_ckpt.items()})
    
    # --- 3. Quantizers ---
    full_levels = config['model']['fsq_levels']
    if len(full_levels) < PRO_DIM:
        pro_fsq_levels = [8] * PRO_DIM
    else:
        pro_fsq_levels = full_levels[:PRO_DIM]
    PRO_LEVELS = 2 
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=pro_fsq_levels, num_levels=PRO_LEVELS, input_dim=PRO_DIM).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    # --- 4. Fuser ---
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=PRO_DIM, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fus_ckpt = torch.load(os.path.join(checkpoint_dir, f"fuser_{tag}.pt"), map_location=device)
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fus_ckpt.items()})
    
    # --- 5. Flow ---
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow = ConditionalFlowMatching(config).to(device)
    flow_ckpt = torch.load(os.path.join(checkpoint_dir, f"flow_{tag}.pt"), map_location=device)
    flow.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()})
    
    return encoder, factorizer, sem_vq, pro_vq, spk_pq, fuser, flow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ckpt_dir", default="checkpoints/microencoder_v2")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load
    encoder, factorizer, sem_vq, pro_vq, spk_pq, fuser, flow = load_models(args.ckpt_dir, "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml", device)
    
    # Process Audio
    wav, sr = sf.read(args.input)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.ndim > 1: wav = wav.mean(dim=-1)
    if sr != 16000:
        import torchaudio.functional as FAudio
        wav = FAudio.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    
    wav = wav / (wav.abs().max() + 1e-6)
    wav = wav.unsqueeze(0).to(device)
    
    # GT Mel
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, win_length=1024, hop_length=320, n_mels=80, f_min=0, f_max=8000
    ).to(device)
    with torch.no_grad():
        gt_mel = mel_transform(wav)
        gt_log_mel = torch.log(torch.clamp(gt_mel, min=1e-5))
    
    # Inference
    with torch.no_grad():
        feat = encoder(wav)
        s, p, spk = factorizer(feat)
        sz, _, _ = sem_vq(s)
        pz, _, _ = pro_vq(p)
        spkz, _, _ = spk_pq(spk)
        
        target_len = sz.shape[1] * 2
        c = fuser(sz, pz, spkz, target_len)
        
        pred = flow.solve_ode(c, steps=50, solver='midpoint', cfg_scale=1.5) # (B, T, 80)
        
        MEAN = -5.0
        STD = 3.5
        pred_denorm = pred * STD + MEAN
        
    # Plot
    pred_cpu = pred_denorm.transpose(1, 2).squeeze(0).cpu().numpy()
    gt_cpu = gt_log_mel.squeeze(0).cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(gt_cpu, origin='lower', aspect='auto', cmap='viridis')
    plt.title(f"Ground Truth ({args.input})")
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(2, 1, 2)
    plt.imshow(pred_cpu, origin='lower', aspect='auto', cmap='viridis')
    plt.title("Prediction (V2 - Epoch 30)")
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig("debug_v2_spectrogram.png")
    print("Saved debug_v2_spectrogram.png")

if __name__ == "__main__":
    main()
