#!/usr/bin/env python3
"""
Inference pipeline for Flow Matching V2 with ConditionFuserV2.
Uses learned upsampling + cross-attention for better conditioning.
"""
import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import yaml

from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def main():
    parser = argparse.ArgumentParser(description="Flow Matching V2 Inference")
    parser.add_argument("--flow_checkpoint", required=True, help="Flow model checkpoint")
    parser.add_argument("--fuser_checkpoint", required=True, help="FuserV2 checkpoint")
    parser.add_argument("--bitnet_dir", required=True, help="Directory with factorizer, quantizers")
    parser.add_argument("--vocoder_checkpoint", required=True, help="MelVocoder checkpoint")
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--output_wav", default="outputs/flow_v2_output.wav")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps")
    parser.add_argument("--config", default="src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    os.makedirs(os.path.dirname(args.output_wav), exist_ok=True)
    
    # === Load Official HuBERT ===
    print("Loading Official HuBERT (matches training data)...")
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    # === Load Factorizer + Quantizers ===
    print("Loading Factorizer + Quantizers...")
    factorizer = InformationFactorizerV2(config).to(device)
    sem_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'], 
        num_levels=config['model']['rfsq_num_levels'], 
        input_dim=config['model']['semantic']['output_dim']
    ).to(device)
    pro_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'], 
        num_levels=config['model']['rfsq_num_levels'], 
        input_dim=config['model']['prosody']['output_dim']
    ).to(device)
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    
    def load_part(name, obj):
        p = os.path.join(args.bitnet_dir, f"{name}.pt")
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                 ckpt = ckpt['model_state_dict']
            ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
            obj.load_state_dict(ckpt, strict=False)
            print(f"  Loaded {name}")
    
    load_part("factorizer", factorizer)
    load_part("sem_rfsq", sem_vq)
    load_part("pro_rfsq", pro_vq)
    load_part("spk_pq", spk_pq)
    
    # === Load Flow Model ===
    # === Load Flow Model ===
    print("Loading Flow Model...")
    # Baseline comparison: Large 512-dim Flow
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    flow_ckpt = torch.load(args.flow_checkpoint, map_location=device)
    if isinstance(flow_ckpt, dict) and 'model_state_dict' in flow_ckpt:
        flow_ckpt = flow_ckpt['model_state_dict']
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()}, strict=False)
    flow_model.eval()
    
    # === Load FuserV2 (The real one from checkpoints_flow_v2) ===
    print("Loading FuserV2...")
    fuser = ConditionFuserV2(
        sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512,
        sem_upsample=4, pro_upsample=8
    ).to(device)
    fuser_ckpt = torch.load(args.fuser_checkpoint, map_location=device)
    if isinstance(fuser_ckpt, dict) and 'model_state_dict' in fuser_ckpt:
        fuser_ckpt = fuser_ckpt['model_state_dict']
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fuser_ckpt.items()}, strict=False)
    fuser.eval()
    
    # === Load Vocoder ===
    print("Loading MelVocoder...")
    vocoder = MelVocoderBitNet().to(device)
    voc_ckpt = torch.load(args.vocoder_checkpoint, map_location=device)
    if isinstance(voc_ckpt, dict) and 'model_state_dict' in voc_ckpt:
        voc_ckpt = voc_ckpt['model_state_dict']
    voc_ckpt = {k.replace("_orig_mod.", ""): v for k, v in voc_ckpt.items()}
    vocoder.load_state_dict(voc_ckpt)
    vocoder.eval()
    
    # === Load Audio ===
    print(f"Loading audio: {args.input_wav}")
    import soundfile as sf
    wav_data, sr = sf.read(args.input_wav)
    wav = torch.tensor(wav_data, dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.T  # (C, T)
        
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(device)
        wav = wav.to(device)
        wav = resampler(wav)
    else:
        wav = wav.to(device)

    # Normalize
    wav = wav / (wav.abs().max() + 1e-6)
    
    target_mel_len = wav.shape[1] // 320
    print(f"  Audio: {wav.shape[1]} samples -> {target_mel_len} mel frames")
    
    # === Inference ===
    with torch.no_grad():
        # Extract HuBERT features (Official Layer 9)
        # Input: (B, T) raw waveform
        wav_np = wav.cpu().numpy().squeeze()
        inputs = hubert_processor(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
        hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        features = hubert_out.hidden_states[9] # Layer 9 as used in training
        print(f"  Official HuBERT: {features.shape}")
        
        # Factorize
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        print(f"  Factors: sem={sem_z.shape}, pro={pro_z.shape}, spk={spk_z.shape}")
        
        # Fuse with V2
        cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
        print(f"  Conditioning: {cond.shape}")
        
        # Generate Mel
        print(f"  Generating Mel with {args.steps} ODE steps...")
        mel = flow_model.solve_ode(cond, steps=args.steps, solver='euler')
        print(f"  Generated Mel: {mel.shape}")
        
        # Denormalize
        MEL_MEAN, MEL_STD = -5.0, 2.0
        mel = mel * MEL_STD + MEL_MEAN
        mel = torch.clamp(mel, min=-12.0, max=3.0)
        
        # Vocoder
        print("  Synthesizing audio...")
        audio = vocoder(mel)
    
    # === Save ===
    audio = audio / (audio.abs().max() + 1e-6)
    audio = audio.squeeze().cpu().numpy()
    
    import soundfile as sf
    sf.write(args.output_wav, audio, 16000)
    print(f"Saved: {args.output_wav}")
    
    # === Visualize ===
    plt.figure(figsize=(12, 4))
    mel_vis = mel.squeeze().T.cpu().numpy()  # (80, T)
    plt.imshow(mel_vis, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(f"Generated Mel (Flow V2)")
    plt.savefig(args.output_wav.replace(".wav", ".png"))
    print(f"Saved: {args.output_wav.replace('.wav', '.png')}")


if __name__ == "__main__":
    main()
