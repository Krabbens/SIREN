
import os
import sys
# Add CWD and CWD/src to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))
print(f"DEBUG: sys.path: {sys.path}")

import torch
import torch.nn.functional as F
import soundfile as sf
import yaml
import argparse
import numpy as np

# Imports
from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoderTiny
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder

def load_models(checkpoint_dir, config_path, device, vocoder_path=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    print(f"Loading checkpoints from {checkpoint_dir}...")
    
    # Auto-detect latest epoch
    import glob
    import re
    
    files = glob.glob(os.path.join(checkpoint_dir, "encoder_ep*.pt"))
    if not files:
        raise FileNotFoundError(f"No encoder checkpoints found in {checkpoint_dir}")
        
    epochs = []
    for f in files:
        match = re.search(r"encoder_ep(\d+).pt", f)
        if match:
            epochs.append(int(match.group(1)))
            
    if not epochs:
        print("No numbered epochs found. Trying 'encoder_best.pt'.")
        tag = "best"
    else:
        latest_epoch = max(epochs)
        tag = f"ep{latest_epoch}"
        print(f"Detected latest checkpoint: Epoch {latest_epoch}")
    
    # 1. Encoder (Tiny)
    encoder = MicroEncoderTiny(hidden_dim=128, output_dim=768, num_layers=2).to(device)
    encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"encoder_{tag}.pt"), map_location=device))
    
    # 2. Factorizer
    factorizer = InformationFactorizerV2(config).to(device)
    factorizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"factorizer_{tag}.pt"), map_location=device))
    
    # 3. Quantizers (No weights, just init)
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    # 4. Fuser
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fuser.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"fuser_{tag}.pt"), map_location=device))
    
    # 5. Flow
    # Config patch from training script
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow = ConditionalFlowMatching(config).to(device)
    flow.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"flow_{tag}.pt"), map_location=device))
    
    # 6. Vocoder
    vocoder = None
    try:
        path = vocoder_path if vocoder_path else "checkpoints/vocoder_mel/vocoder_latest.pt"
        voc_ckpt = torch.load(path, map_location=device)
        
        # Check dict vs flat
        state_dict = voc_ckpt
        if isinstance(voc_ckpt, dict) and 'model_state_dict' in voc_ckpt: 
            state_dict = voc_ckpt['model_state_dict']
            
        # Detect Vocoder Type
        if 'conv_in.weight' in state_dict:
            # BitVocoder
            # Determine DIM from weight shape: [dim, input_dim, K]
            w_shape = state_dict['conv_in.weight'].shape
            dim = w_shape[0]
            input_dim = w_shape[1]
            print(f"  Vocoder: Detected BitVocoder (dim={dim}, input_dim={input_dim})")
            
            vocoder = BitVocoder(input_dim=input_dim, dim=dim, num_layers=4, num_res_blocks=2).to(device)
        else:
            # MelVocoderBitNet
            print(f"  Vocoder: Detected MelVocoderBitNet")
            vocoder = MelVocoderBitNet().to(device)

        vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
        print(f"  Vocoder: loaded from {path}")
    except Exception as e:
        print(f"  [Warning] Failed to load vocoder: {e}")
        vocoder = None
    
    return encoder, factorizer, sem_vq, pro_vq, spk_pq, fuser, flow, vocoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output", default="output_siamese.wav", help="Output audio file")
    parser.add_argument("--ckpt_dir", default="checkpoints/microencoder_aug_siamese", help="Checkpoint directory")
    parser.add_argument("--config", default="src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml", help="Config file")
    parser.add_argument("--vocoder_ckpt", default="checkpoints/vocoder_mel/vocoder_latest.pt", help="Vocoder checkpoint path")
    parser.add_argument("--steps", type=int, default=50, help="ODE solver steps")
    parser.add_argument("--solver", default="midpoint", choices=["euler", "midpoint", "rk4", "heun"], help="ODE solver")
    parser.add_argument("--cfg_scale", type=float, default=1.5, help="CFG scale (1.0=no guidance)")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Models
    encoder, factorizer, sem_vq, pro_vq, spk_pq, fuser, flow, vocoder = load_models(args.ckpt_dir, args.config, device, args.vocoder_ckpt)
    
    # Load Audio
    # Use torchaudio if strictly needed for resample, but soundfile logic preferred for IO
    wav, sr = sf.read(args.input)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    
    # Resample to 16k
    if sr != 16000:
        import torchaudio.functional as F_audio
        wav = F_audio.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    
    # Normalize
    wav = wav / (wav.abs().max() + 1e-6)
    wav = wav.unsqueeze(0).to(device) # (1, T)
    
    print(f"Running inference on {args.input}...")
    
    with torch.no_grad():
        # Encode
        feat = encoder(wav)
        
        # Factorize
        s, p, spk = factorizer(feat)
        
        # Quantize
        sz, _, _ = sem_vq(s)
        pz, _, _ = pro_vq(p)
        spkz, _, _ = spk_pq(spk)
        
        # Fuse
        # Semantic is 25Hz (16000/640). Mel is 50Hz (16000/320). 
        # So we want target_len = 2 * semantic_len. 
        # Fuser naturally upsamples by 4 (to 100Hz) then interpolates down to target.
        target_len = sz.shape[1] * 2
        c = fuser(sz, pz, spkz, target_len)
        
        # Flow Decode
        print("  Solving Flow ODE...")
        pred = flow.solve_ode(c, steps=args.steps, solver=args.solver, cfg_scale=args.cfg_scale)
        print(f"  ODE: {args.steps} steps, solver={args.solver}, cfg={args.cfg_scale}")
        
        # Denormalize Mel for saving plot
        # Denormalize Mel for saving plot
        MEAN = -6.0
        STD = 3.5
        pred_denorm = pred * STD + MEAN
        
        print(f"DEBUG: pred shape (native T,C): {pred.shape}")


        
        # --- Visualization: Comparison with Ground Truth ---
        import torchaudio
        
        # 1. Compute GT Mel
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=320,
            n_mels=80,
            f_min=0,
            f_max=8000
        ).to(device)
        
        with torch.no_grad():
            gt_mel = mel_transform(wav)
            gt_log_mel = torch.log(torch.clamp(gt_mel, min=1e-5))
            # No need to normalize for visual comparison, just log-scale
        
        # 2. Plotting
        # pred_denorm is (B, T, C). We need (C, T) for plotting
        pred_cpu = pred_denorm.transpose(1, 2).squeeze(0).cpu().numpy()
        gt_cpu = gt_log_mel.squeeze(0).cpu().numpy()

        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        
        # Ground Truth
        plt.subplot(2, 1, 1)
        plt.imshow(gt_cpu, origin='lower', aspect='auto', cmap='viridis')
        plt.title(f"Ground Truth ({args.input})")
        plt.colorbar(format='%+2.0f dB')
        
        # Prediction
        plt.subplot(2, 1, 2)
        plt.imshow(pred_cpu, origin='lower', aspect='auto', cmap='viridis')
        plt.title("Prediction (Siamese Model)")
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(args.output.replace(".wav", ".png"))
        plt.close()
        print(f"Saved comparison spectrogram to {args.output.replace('.wav', '.png')}")
        
    # Vocode
    if vocoder is not None:
        print("  Vocoding...")
        with torch.no_grad():
            # Vocoder expects denormalized log-mel? Or normalized? 
            # In training, vocoder usually trained on GT Mels.
            # Assuming trained on same normalization stats or unnormalized.
            # Usually MelVocoderBitNet trained on unnormalized, or standard.
            # Let's try passing the denormalized one.
            
            # Check dimension mismatch (e.g. 80 vs 100)
            voc_in = pred_denorm # (B, T, C)
            target_dim = 80
            if hasattr(vocoder, 'input_dim'):
                target_dim = vocoder.input_dim
            elif hasattr(vocoder, 'input_conv'): # MelVocoderBitNet
                target_dim = vocoder.input_conv.in_channels
                
            if target_dim != 80:
                print(f"  Resizing Mel from 80 to {target_dim} for vocoder...")
                # Resize channels 80 -> 100 using 2D interpolation (Treating (B, 1, H, W))
                # Input: (B, T, C) -> (B, 1, C, T)
                voc_in = voc_in.transpose(1, 2).unsqueeze(1) # (B, 1, 80, T)
                voc_in = F.interpolate(voc_in, size=(target_dim, voc_in.shape[-1]), mode='bilinear', align_corners=False)
                # Output: (B, 1, 100, T) -> (B, T, 100)
                voc_in = voc_in.squeeze(1).transpose(1, 2)
            
            audio_out = vocoder(voc_in)

            
            # Save
            audio_out = audio_out.squeeze().cpu().numpy()
            sf.write(args.output, audio_out, 16000)
            print(f"Saved audio to {args.output}")
    else:
        print("Skipping vocoding (model not loaded).")


if __name__ == "__main__":
    main()
