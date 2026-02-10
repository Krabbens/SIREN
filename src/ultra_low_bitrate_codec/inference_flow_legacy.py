#!/usr/bin/env python3
"""
Inference script for Flow Matching using Legacy model (checkpoints_flow_matching).

This checkpoint outputs Complex STFT (1026 = 513 real + 513 imag), not Mel.
Audio is synthesized using ISTFT directly.
"""
import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from scipy.io import wavfile

from ultra_low_bitrate_codec.models.flow_matching_legacy import ConditionalFlowMatchingLegacy
from ultra_low_bitrate_codec.models.fuser import ConditionFuser
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.tiny_hubert import TinyHubert


def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Flow Matching checkpoint")
    parser.add_argument("--bitnet_checkpoint", required=True, help="Directory with factorizer, quantizers")
    parser.add_argument("--tiny_hubert_path", required=True, help="TinyHubert checkpoint")
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--output_wav", default="outputs/flow_legacy_output.wav")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps")
    parser.add_argument("--config", default="src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    os.makedirs(os.path.dirname(args.output_wav), exist_ok=True)
    
    # --- Load TinyHubert ---
    print(f"Loading TinyHubert from {args.tiny_hubert_path}...")
    tiny_hubert = TinyHubert(out_dim=768, hidden_dim=384, num_layers=12).to(device)
    tiny_hubert.load_state_dict(torch.load(args.tiny_hubert_path, map_location=device))
    tiny_hubert.eval()

    # --- Load Encoder Components ---
    print("Loading Encoder Components...")
    factorizer = InformationFactorizerV2(config).to(device)
    fsq_levels = config['model']['fsq_levels']
    rfsq_num_levels = config['model']['rfsq_num_levels']
    sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, 
                         input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, 
                         input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    
    ckpt_dir = args.bitnet_checkpoint
    if os.path.isfile(ckpt_dir):
        ckpt_dir = os.path.dirname(ckpt_dir)
    
    def load_part(name, obj):
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location=device)
            obj.load_state_dict({k.replace("_orig_mod.", ""): v for k,v in d.items()}, strict=False)
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {name} not found at {p}")
    
    load_part("factorizer", factorizer)
    load_part("sem_rfsq", sem_vq)
    load_part("pro_rfsq", pro_vq)
    load_part("spk_pq", spk_pq)
    
    # --- Load Flow Model (Legacy) ---
    print("Loading Flow Model (Legacy)...")
    flow_model = ConditionalFlowMatchingLegacy(
        hidden_dim=256,
        cond_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    flow_ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(flow_ckpt, dict) and 'model_state_dict' in flow_ckpt:
        flow_ckpt = flow_ckpt['model_state_dict']
    
    # Load with strict=False to see what's missing
    msg = flow_model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()}, 
        strict=False
    )
    print(f"  Loaded flow model. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"    Missing: {msg.missing_keys[:5]}...")
    if msg.unexpected_keys:
        print(f"    Unexpected: {msg.unexpected_keys[:5]}...")
    flow_model.eval()
    
    # --- Load Fuser ---
    flow_dir = os.path.dirname(args.model_path)
    flow_basename = os.path.basename(args.model_path)
    fuser_path = os.path.join(flow_dir, flow_basename.replace("flow_", "fuser_"))
    
    # Default fuser dimensions from training (8+8+256 = 272 -> 512)
    fuser = ConditionFuser(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512).to(device)
    
    if os.path.exists(fuser_path):
        fuser_ckpt = torch.load(fuser_path, map_location=device)
        fuser.load_state_dict(
            {k.replace("_orig_mod.", ""): v for k, v in fuser_ckpt.items()}, 
            strict=False
        )
        print(f"  Loaded fuser from {fuser_path}")
    else:
        print(f"  WARNING: Fuser not found at {fuser_path}")
    fuser.eval()
    
    # --- Load Audio ---
    print(f"Loading audio: {args.input_wav}")
    try:
        sr, wav_data = wavfile.read(args.input_wav)
        wav = torch.tensor(wav_data, dtype=torch.float32)
    except:
        wav, sr = torchaudio.load(args.input_wav)
    
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav / (wav.abs().max() + 1e-6)
    wav = wav.to(device)
    
    # STFT params matching training
    n_fft = 1024
    hop_length = 320
    win_length = 1024
    
    target_stft_len = wav.shape[1] // hop_length
    print(f"  Audio samples: {wav.shape[1]}, Target STFT frames: {target_stft_len}")
    
    # --- Inference ---
    with torch.no_grad():
        # Extract features
        features = tiny_hubert(wav)  # (B, T_feat, 768)
        print(f"  TinyHubert features: {features.shape}")
        
        # Factorize
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        print(f"  Factorized: sem={sem_z.shape}, pro={pro_z.shape}, spk={spk_z.shape}")
        
        # Fuse conditioning
        cond = fuser(sem_z, pro_z, spk_z, target_stft_len)
        print(f"  Fused conditioning: {cond.shape}")  # Should be (B, T, 512)
        
        # Generate Complex STFT
        print(f"Generating with {args.steps} ODE steps...")
        stft_pred = flow_model.solve_ode(cond, steps=args.steps, solver='euler')
        print(f"  Generated STFT: {stft_pred.shape}")  # (B, T, 1026)
        
    # --- Denormalize STFT ---
    # The training normalized the STFT. Need to denormalize.
    # Typical normalization: divide by 20 (to bring into -1 to 1 range)
    stft_pred = stft_pred * 20.0
    
    # Split real and imaginary
    stft_pred = stft_pred.squeeze(0)  # (T, 1026)
    real = stft_pred[:, :513].T  # (513, T)
    imag = stft_pred[:, 513:].T  # (513, T)
    complex_stft = torch.complex(real, imag)  # (513, T)
    
    print(f"  Complex STFT: {complex_stft.shape}")
    print(f"  Magnitude range: [{complex_stft.abs().min():.2f}, {complex_stft.abs().max():.2f}]")
    
    # --- ISTFT ---
    window = torch.hann_window(win_length).to(device)
    audio = torch.istft(
        complex_stft.unsqueeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=False
    )
    print(f"  Reconstructed audio: {audio.shape}")
    
    # Normalize and save
    audio = audio / (audio.abs().max() + 1e-6)
    audio = audio.squeeze().cpu().numpy()
    
    import soundfile as sf
    sf.write(args.output_wav, audio, 16000)
    print(f"Saved to {args.output_wav}")
    
    # --- Visualize ---
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(complex_stft.abs().cpu().numpy(), origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Generated STFT Magnitude")
    
    plt.subplot(2, 2, 2)
    plt.imshow(torch.angle(complex_stft).cpu().numpy(), origin='lower', aspect='auto', cmap='twilight')
    plt.colorbar()
    plt.title("Generated STFT Phase")
    
    plt.subplot(2, 2, 3)
    plt.plot(audio)
    plt.title("Reconstructed Waveform")
    
    plt.subplot(2, 2, 4)
    cond_vis = cond.squeeze().T.cpu().numpy()
    plt.imshow(cond_vis[:64], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Conditioning (first 64 dims)")
    
    plt.tight_layout()
    plt.savefig(args.output_wav.replace(".wav", ".png"))
    print(f"Saved visualization to {args.output_wav.replace('.wav', '.png')}")


if __name__ == "__main__":
    main()
