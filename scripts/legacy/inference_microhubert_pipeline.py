#!/usr/bin/env python3
"""
Full Pipeline Inference with MicroHuBERT

MicroHuBERT (3.2MB) → Factorizer → VQ → Fuser → Flow → Vocoder → Audio

Usage:
    python inference_microhubert_pipeline.py \
        --input data/jakubie_16k.wav \
        --output outputs/microhubert_pipeline.wav
"""

import os
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import yaml
import soundfile as sf
import argparse

# Models
from ultra_low_bitrate_codec.models.micro_hubert import MicroHuBERT
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/jakubie_16k.wav")
    parser.add_argument("--output", default="outputs/microhubert_pipeline_ep95_final.wav")
    parser.add_argument("--microhubert_ckpt", default="checkpoints/microhubert/microhubert_ep95.pt")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--gain", type=float, default=0.0, help="Gain adjustment in dB (added to Log Mel)")
    parser.add_argument("--feature_scale", type=float, default=1.0, help="Scale factor for MicroHuBERT features")
    # Prioritize the 'best_step' checkpoint if available
    default_ckpt = "checkpoints/factorizer_microhubert_finetune/factorizer.pt"
    if os.path.exists("checkpoints/factorizer_microhubert_finetune/factorizer_best_step.pt"):
        default_ckpt = "checkpoints/factorizer_microhubert_finetune/factorizer_best_step.pt"
        print(f"   >>> Using Best Step Checkpoint: {default_ckpt}")
        
    parser.add_argument("--factorizer_ckpt", default=default_ckpt, help="Path to custom factorizer checkpoint")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Flow Matching CFG Scale")
    parser.add_argument("--solver", default="rk4", choices=['euler', 'midpoint', 'rk4', 'heun'], help="ODE Solver")
    parser.add_argument("--checkpoint_dir", default=None, help="Directory containing full trained suite (factorizer, fuser, flow)")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("=" * 60)
    print("Full Pipeline with MicroHuBERT (3.2MB)")
    print("=" * 60)
    
    # Config
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # =========================================================================
    # Load MicroHuBERT (replaces 360MB HuBERT!)
    # =========================================================================
    print("\n1. Loading MicroHuBERT...")
    micro = MicroHuBERT().to(device)
    
    # Load and fix compiled keys
    ckpt = torch.load(args.microhubert_ckpt, map_location=device)
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    micro.load_state_dict(ckpt)
    
    micro.eval()
    print(f"   Checkpoint: {args.microhubert_ckpt}")
    print(f"   Size: 3.22 MB (INT8)")
    
    # =========================================================================
    # Load Pipeline Components
    # =========================================================================
    # =========================================================================
    # Load Pipeline Components (Dynamic Loading)
    # =========================================================================
    print("\n2. Loading Pipeline...")
    
    # Checkpoint Priorities:
    # 1. --checkpoint_dir (Loads all: factorizer, fuser, flow)
    # 2. Individual Arguments (Legacy)
    
    ckpt_dir = args.checkpoint_dir
    
    # 1. Factorizer
    factorizer = InformationFactorizerV2(config).to(device)
    if ckpt_dir:
        # Try best step first, then epoch, then standard
        f_path = os.path.join(ckpt_dir, "factorizer_best_step.pt")
        if not os.path.exists(f_path): f_path = os.path.join(ckpt_dir, "factorizer.pt")
        print(f"   Loading Factorizer from {f_path}")
        state = torch.load(f_path, map_location=device)
        if 'model_state_dict' in state: state = state['model_state_dict']
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        factorizer.load_state_dict(state)
        
    # 2. Quantizers (Frozen)
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device)
    
    # Load quantizers (usually static or from specific path, assume static/pretrained logic for now)
    # Ideally should be in checkpoint dir too if we fine-tune them (we don't currently)
    # For now, we assume they are unused/frozen or standard.
    # Actually, we need to load them if they were part of the factorizer package? No, they are separate.
    # Let's assume standard location or inside checkpoint_dir if available.
    q_dir = "checkpoints/checkpoints_stable/step_87000" # Fallback
    for name, model in [("sem_rfsq", sem_vq), ("pro_rfsq", pro_vq), ("spk_pq", spk_pq)]:
        p = os.path.join(ckpt_dir, f"{name}.pt") if ckpt_dir else os.path.join(q_dir, f"{name}.pt")
        if not os.path.exists(p) and ckpt_dir: # Fallback to stable if not in new dir
             p = os.path.join(q_dir, f"{name}.pt")
        
        if os.path.exists(p):
            s = torch.load(p, map_location=device)
            if 'model_state_dict' in s: s = s['model_state_dict']
            model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in s.items()})
    
    # 3. Flow Model
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    
    if ckpt_dir:
        fl_path = os.path.join(ckpt_dir, "flow_best_step.pt")
        if not os.path.exists(fl_path): fl_path = os.path.join(ckpt_dir, "flow_model.pt")
        print(f"   Loading Flow form {fl_path}")
        flow_ckpt = torch.load(fl_path, map_location=device)
        if 'model_state_dict' in flow_ckpt: flow_ckpt = flow_ckpt['model_state_dict']
        
        # Load flow state dict (directly compatible with ManualFlashAttention)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()}
        flow_model.load_state_dict(state_dict, strict=False)
        
    flow_model.eval()
    
    # 4. Fuser
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    if ckpt_dir:
        fu_path = os.path.join(ckpt_dir, "fuser_best_step.pt")
        if not os.path.exists(fu_path): fu_path = os.path.join(ckpt_dir, "fuser.pt")
        print(f"   Loading Fuser from {fu_path}")
        fuser_ckpt = torch.load(fu_path, map_location=device)
        if 'model_state_dict' in fuser_ckpt: fuser_ckpt = fuser_ckpt['model_state_dict']
        fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fuser_ckpt.items()}, strict=False)
    
    fuser.eval()
    
    # Vocoder
    vocoder = MelVocoderBitNet().to(device)
    voc_ckpt = torch.load("checkpoints/vocoder_mel/vocoder_latest.pt", map_location=device)
    if isinstance(voc_ckpt, dict) and 'model_state_dict' in voc_ckpt:
        voc_ckpt = voc_ckpt['model_state_dict']
    vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in voc_ckpt.items()})
    vocoder.eval()
    print("   Loaded Vocoder")
    
    # =========================================================================
    # Load Audio
    # =========================================================================
    print(f"\n3. Loading audio: {args.input}")
    print(f"\n3. Loading audio: {args.input}")
    wav, sr = sf.read(args.input)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() > 1:
        wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav / (wav.abs().max() + 1e-6)
    
    # MATCH TRAINING HOP LENGTH (320)
    target_mel_len = wav.shape[0] // 320
    print(f"   {wav.shape[0]} samples -> {target_mel_len} frames (hop=320)")
    
    # =========================================================================
    # Inference
    # =========================================================================
    print(f"\n4. Running inference...")
    
    with torch.no_grad():
        # MicroHuBERT features
        features = micro(wav.unsqueeze(0).to(device))
        
        # Scaling Fix: MicroHuBERT vs Official HuBERT
        # User requested scaling sweep
        SCALE_FACTOR = args.feature_scale
        features = features * SCALE_FACTOR
        print(f"   MicroHuBERT: {features.shape} (Scaled x{SCALE_FACTOR:.2f})")
        
        # Factorize
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        print(f"   Factors: sem={sem_z.shape}, pro={pro_z.shape}, spk={spk_z.shape}")
        
        # Fuse
        cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
        print(f"   Conditioning: {cond.shape}")
        
        # Flow
        print(f"   Generating mel ({args.steps} steps, solver={args.solver}, cfg={args.cfg_scale})...")
        mel = flow_model.solve_ode(cond, steps=args.steps, solver=args.solver, cfg_scale=args.cfg_scale)
        
        # Denormalize
        # Training: (x - (-5.0)) / 3.5
        MEL_MEAN, MEL_STD = -5.0, 3.5
        mel = mel * MEL_STD + MEL_MEAN
        
        # Apply Gain if requested
        if args.gain != 0.0:
            print(f"   Applying gain: {args.gain} dB")
            mel = mel + args.gain
            
        mel = torch.clamp(mel, min=-12.0, max=3.0)
        print(f"   Generated mel: {mel.shape}, mean={mel.mean():.2f}, max={mel.max():.2f}")
        
        # Spectrogram Visualization (Before Resampling to show true generation)
        plt.figure(figsize=(12, 4))
        mel_vis = mel.squeeze().T.cpu().numpy()
        plt.imshow(mel_vis, origin='lower', aspect='auto', cmap='viridis', vmin=-12, vmax=3)
        plt.colorbar()
        plt.title("MicroHuBERT Pipeline (3.2MB feature extractor)")
        spec_path = args.output.replace(".wav", ".png")
        plt.savefig(spec_path, dpi=150)
        print(f"   Spectrogram: {spec_path}")

        # No resampling needed if hop=320
        mel_resampled = mel
        
        # Vocoder
        audio = vocoder(mel_resampled)
    
    # =========================================================================
    # Save
    # =========================================================================
    print(f"\n5. Saving outputs...")
    audio = audio / (audio.abs().max() + 1e-6)
    audio_np = audio.squeeze().detach().cpu().numpy()
    
    sf.write(args.output, audio_np, 16000)
    print(f"   Audio: {args.output}")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    print("=" * 60)



if __name__ == "__main__":
    main()
