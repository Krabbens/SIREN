#!/usr/bin/env python3
"""
Flow Matching V2 Inference with TinyHubert + Adapted Factorizer
"""
import torch
import torchaudio
import yaml
import argparse
import os
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

from ultra_low_bitrate_codec.models.tiny_hubert import TinyHubert
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def fix_state_dict_keys(state_dict):
    """Remove keys not in model (e.g. from compilation)"""
    new_sd = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "")
        new_sd[k] = v
    return new_sd

def load_safe(model, state_dict, prefix=""):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if own_state[name].shape != param.shape:
            print(f"⚠️ Skipping {prefix}{name} due to size mismatch: {param.shape} vs {own_state[name].shape}")
            continue
        own_state[name].copy_(param)

def main():
    parser = argparse.ArgumentParser(description="Flow Matching V2 Inference (TinyHubert)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--factorizer_dir", type=str, required=True, help="Path to adapted factorizer checkpoint dir (e.g. step_89000)")
    parser.add_argument("--flow_checkpoint", type=str, required=True)
    parser.add_argument("--vocoder_checkpoint", type=str, required=True)
    parser.add_argument("--input_wav", type=str, required=True)
    parser.add_argument("--output_wav", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # === Load TinyHubert ===
    print("Loading TinyHubert...")
    hubert_model = TinyHubert(out_dim=768, hidden_dim=384, num_layers=12).to(device)
    # Using the same checkpoint used for training adaptation
    hubert_model.load_state_dict(torch.load("checkpoints/tiny_hubert_best.pt", map_location=device))
    hubert_model.eval()

    # === Load Factorizer + Quantizers (Adapted) ===
    print(f"Loading Adapted Factorizer from {args.factorizer_dir}...")
    factorizer = InformationFactorizerV2(config).to(device)
    
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device)

    # Load from step dir
    load_safe(factorizer, fix_state_dict_keys(torch.load(f"{args.factorizer_dir}/factorizer.pt", map_location=device)), "factorizer.")
    load_safe(sem_vq, fix_state_dict_keys(torch.load(f"{args.factorizer_dir}/sem_rfsq.pt", map_location=device)), "sem.")
    load_safe(pro_vq, fix_state_dict_keys(torch.load(f"{args.factorizer_dir}/pro_rfsq.pt", map_location=device)), "pro.")
    load_safe(spk_pq, fix_state_dict_keys(torch.load(f"{args.factorizer_dir}/spk_pq.pt", map_location=device)), "spk.")
    
    factorizer.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_pq.eval()

    # === Load Flow Model ===
    print("Loading Flow Model...")
    
    # Ensure config matches V2 architecture (Large Flow)
    # fusion_dim is input/output (Mel) dimension
    config['model']['decoder']['fusion_dim'] = 80
    # hidden_dim is internal DiT dimension
    config['model']['decoder']['hidden_dim'] = 512
    
    flow_model = ConditionalFlowMatching(config).to(device)
    
    # Load Flow Checkpoint
    flow_ckpt = torch.load(args.flow_checkpoint, map_location=device)
    if isinstance(flow_ckpt, dict) and 'model_state_dict' in flow_ckpt:
        flow_ckpt = flow_ckpt['model_state_dict']
    
    msg = flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()}, strict=False)
    print(f"Flow model loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    flow_model.eval()
    
    # === Load FuserV2 (Corrected for TinyHubert + FlowV2) ===
    print("Loading FuserV2...")
    from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
    fuser = ConditionFuserV2(
        sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512,
        sem_upsample=4, pro_upsample=8
    ).to(device)
    
    # Deriving fuser path from flow path
    # flow_dir = os.path.dirname(args.flow_checkpoint)
    # Force using the V2 fuser from checkpoints_flow_v2 if flow is from there
    # Or just use the arg if I add it? For now, let's hardcode fallbacks or assume args pass usage.
    # Actually, flow_dir logic works if I pass flow from flow_v2 folder.
    
    flow_dir = os.path.dirname(args.flow_checkpoint)
    flow_base = os.path.basename(args.flow_checkpoint)
    fuser_path = os.path.join(flow_dir, flow_base.replace("flow_", "fuser_"))
    
    if os.path.exists(fuser_path):
        print(f"Loading fuser from {fuser_path}")
        ckpt = torch.load(fuser_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}, strict=False)
    else:
        print(f"⚠️ Could not load fuser from {fuser_path}! Using random weights")
    flow_dir = os.path.dirname(args.flow_checkpoint)
    flow_base = os.path.basename(args.flow_checkpoint)
    fuser_path = os.path.join(flow_dir, flow_base.replace("flow_", "fuser_"))
    
    if os.path.exists(fuser_path):
        print(f"Loading fuser from {fuser_path}")
        fuser.load_state_dict(fix_state_dict_keys(torch.load(fuser_path, map_location=device)), strict=False)
    else:
        print(f"⚠️ Could not load fuser from {fuser_path}! Using random weights")

    fuser.eval()

    # === Load MelVocoder ===
    print("Loading MelVocoder...")
    vocoder = MelVocoderBitNet().to(device)
    load_safe(vocoder, fix_state_dict_keys(torch.load(args.vocoder_checkpoint, map_location=device)))
    vocoder.eval()

    # === Inference ===
    print(f"Loading audio: {args.input_wav}")
    print(f"Loading audio: {args.input_wav}")
    wav, sr = sf.read(args.input_wav)
    wav = torch.tensor(wav, dtype=torch.float32)
    # wav is (T, C) or (T,) from soundfile
    if wav.dim() > 1: wav = wav.mean(1) # Flatten check
    
    if sr != 16000:
        # We need resampler. If torchaudio fails, we might be stuck.
        # But import was fine. Let's try to use torchaudio transforms only if needed.
        # But we converted input to 16000 with ffmpeg! So sr should match.
        print(f"Warning: Input SR is {sr}, expected 16000")
        resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
        wav = resampler(wav.unsqueeze(0)).squeeze(0)

    # Pad to multiple of hop
    wav = wav.unsqueeze(0).to(device) # (1, T)
    
    target_mel_len = wav.shape[1] // 320
    
    with torch.no_grad():
        # Extract TinyHubert features
        hubert_out = hubert_model(wav) # (1, T_h, 768)
        print(f"  TinyHubert Features: {hubert_out.shape}")
        
        # Factorize (Adapted!)
        sem, pro, spk = factorizer(hubert_out)
        print(f"  Factors: sem={sem.shape}, pro={pro.shape}, spk={spk.shape}")
        
        # Quantize
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        # Fuse
        cond = fuser(sem_z, pro_z, spk_z, target_len=target_mel_len)
        print(f"  Conditioning: {cond.shape}")
        
        # Generate Mel
        print("  Generating Mel with 50 ODE steps...")
        # Start from noise
        # T_mel matches conditioning length
        batch_size = 1
        seq_len = cond.shape[1]
        
        # Flow Matching Inference
        t = torch.linspace(0, 1, 50).to(device)
        x = torch.randn(batch_size, seq_len, 80).to(device)
        
        dt = t[1] - t[0]
        for i in range(len(t) - 1):
            t_batch = torch.ones(batch_size).to(device) * t[i]
            v_pred = flow_model(x, mask=None, cond=cond, t=t_batch)
            x = x + v_pred * dt
            
        gen_mel = x
        print(f"  Generated Mel: {gen_mel.shape}")
        
        # Vocode
        # Reconstruct waveform
        audio_hat = vocoder(gen_mel)
        
        # Save
        sf.write(args.output_wav, audio_hat.cpu().squeeze().numpy(), 16000)
        print(f"Saved: {args.output_wav}")
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.imshow(gen_mel.cpu().squeeze().transpose(0, 1).numpy(), aspect='auto', origin='lower', vmin=-12, vmax=3)
        plt.colorbar()
        plt.title("Generated Mel (Flow V2 + TinyHubert)")
        plt.savefig(args.output_wav.replace(".wav", ".png"))
        print(f"Saved: {args.output_wav.replace('.wav', '.png')}")

if __name__ == "__main__":
    main()
