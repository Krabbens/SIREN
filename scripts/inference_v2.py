
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

# Imports
from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoderTiny, MicroEncoder
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder

def load_models(checkpoint_dir, config_path, device, vocoder_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    print(f"Loading checkpoints from {checkpoint_dir}...")
    
    # Auto-detect latest epoch
    import glob
    import re
    
    files = glob.glob(os.path.join(checkpoint_dir, "encoder_ep*.pt"))
    if not files:
        # Try finding 'encoder_best.pt' if no numbered ones
        if os.path.exists(os.path.join(checkpoint_dir, "encoder_best.pt")):
            tag = "best"
            print("Detected 'encoder_best.pt'")
        else:
            raise FileNotFoundError(f"No encoder checkpoints found in {checkpoint_dir}")
    else:
        epochs = []
        for f in files:
            # Extract number
            match = re.search(r"encoder_ep(\d+).pt", f)
            if match:
                epochs.append(int(match.group(1)))
        
        if not epochs:
             tag = "best"
        else:
            latest_epoch = max(epochs)
            tag = f"ep{latest_epoch}"
            print(f"Detected latest checkpoint: Epoch {latest_epoch} (tag={tag})")
    
    # --- 1. Encoder (Auto-detect Tiny vs Standard) ---
    # We inspect the checkpoint to see hidden dimension
    enc_ckpt_path = os.path.join(checkpoint_dir, f"encoder_{tag}.pt")
    enc_ckpt = torch.load(enc_ckpt_path, map_location='cpu')
    enc_state = {k.replace("_orig_mod.", ""): v for k, v in enc_ckpt.items()}
    
    is_tiny = False
    if 'output_proj.weight' in enc_state:
         if enc_state['output_proj.weight'].shape[1] == 128:
             is_tiny = True
    
    if is_tiny:
        print("  Encoder: Starting MicroEncoderTiny...")
        encoder = MicroEncoderTiny(hidden_dim=128, output_dim=768, num_layers=2).to(device)
    else:
        print("  Encoder: Starting MicroEncoder (Standard)...")
        # Assuming standard config if not tiny (dim=256, layers=4 usually, but let's try to match)
        # Check config or default
        encoder = MicroEncoder(hidden_dim=256, output_dim=768, num_layers=4).to(device)
        
    encoder.load_state_dict(enc_state)
    
    # --- 2. Factorizer (V2 Config: Prosody 4D) ---
    # V2 Training used pro_dim=4
    PRO_DIM = 4 
    config['model']['prosody']['output_dim'] = PRO_DIM
    factorizer = InformationFactorizerV2(config).to(device)
    
    fac_ckpt = torch.load(os.path.join(checkpoint_dir, f"factorizer_{tag}.pt"), map_location=device)
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fac_ckpt.items()})
    
    # --- 3. Quantizers ---
    # Metric B3: Smaller Prosody VQ
    # Levels: sliced from config
    full_levels = config['model']['fsq_levels']
    if len(full_levels) < PRO_DIM:
        pro_fsq_levels = [8] * PRO_DIM
    else:
        pro_fsq_levels = full_levels[:PRO_DIM]
        
    PRO_LEVELS = 2 # From V2 script default
    
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=pro_fsq_levels, num_levels=PRO_LEVELS, input_dim=PRO_DIM).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    # --- 4. Fuser ---
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=PRO_DIM, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fus_ckpt = torch.load(os.path.join(checkpoint_dir, f"fuser_{tag}.pt"), map_location=device)
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fus_ckpt.items()})
    
    # --- 5. Flow ---
    # Config patch from training script
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow = ConditionalFlowMatching(config).to(device)
    flow_ckpt = torch.load(os.path.join(checkpoint_dir, f"flow_{tag}.pt"), map_location=device)
    flow.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()}) # Also handle orig_mod here for EMA/Compile
    
    # --- 6. Vocoder (80-band BitVocoder) ---
    print(f"Loading Vocoder from {vocoder_path}...")
    # Matches `train_bitvocoder_mel.py` config
    vocoder = BitVocoder(
        input_dim=80, 
        dim=256, 
        n_fft=1024, 
        hop_length=320, 
        num_layers=4, 
        num_res_blocks=1
    ).to(device)
    
    voc_ckpt = torch.load(vocoder_path, map_location=device)
    
    # Handle both full checkpoint (with opt state) and model-only
    state_dict = voc_ckpt
    if isinstance(voc_ckpt, dict):
        if 'vocoder' in voc_ckpt:
            state_dict = voc_ckpt['vocoder']
            print(f"  Vocoder: Loaded from checkpoint dictionary (Epoch {voc_ckpt.get('epoch', '?')})")
        elif 'model_state_dict' in voc_ckpt:
            state_dict = voc_ckpt['model_state_dict']
            
    # Clean keys
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    vocoder.load_state_dict(state_dict)
    
    return encoder, factorizer, sem_vq, pro_vq, spk_pq, fuser, flow, vocoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output_v2.wav")
    parser.add_argument("--ckpt_dir", default="checkpoints/microencoder_v2")
    parser.add_argument("--config", default="src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    # Default to last vocoder checkpoint
    parser.add_argument("--vocoder_ckpt", default="checkpoints/vocoder_80band/bitvocoder_last.pt")
    
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--solver", default="midpoint")
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load
    encoder, factorizer, sem_vq, pro_vq, spk_pq, fuser, flow, vocoder = load_models(args.ckpt_dir, args.config, device, args.vocoder_ckpt)
    
    # Process Audio
    wav, sr = sf.read(args.input)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.ndim > 1: wav = wav.mean(dim=-1)
    
    # Resample 16k
    import torchaudio.functional as FAudio
    if sr != 16000:
        wav = FAudio.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
        
    # Normalize
    wav = wav / (wav.abs().max() + 1e-6)
    wav = wav.unsqueeze(0).to(device)
    
    print(f"Inference on {args.input}...")
    
    with torch.no_grad():
        # Encoder -> Factorizer -> Quantizer
        feat = encoder(wav)
        s, p, spk = factorizer(feat)
        sz, _, _ = sem_vq(s)
        pz, _, _ = pro_vq(p)
        spkz, _, _ = spk_pq(spk)
        
        # Fuser
        # Target len = 50Hz (Mel). Semantic is 25Hz.
        target_len = sz.shape[1] * 2
        c = fuser(sz, pz, spkz, target_len)
        
        # Flow
        print(f"  CFG Scale: {args.cfg_scale}, Steps: {args.steps}")
        pred = flow.solve_ode(c, steps=args.steps, solver=args.solver, cfg_scale=args.cfg_scale) # (B, T, 80)
        
        # Denormalize
        MEAN = -5.0 # Updated from train_e2e_encoder_v2.py / train_bitvocoder_mel.py
        STD = 3.5
        pred_denorm = pred * STD + MEAN
        
        # Vocoder
        # BitVocoder (80-band) expects (B, 80, T) ? No wait.
        # Let's check train_bitvocoder_mel.py
        # input_dim=80. 
        # conv_in = BitConv1d(input_dim, dim, ...)
        # So it expects (B, C, T).
        # Our pred is (B, T, C). Need transpose.
        voc_in = pred_denorm.transpose(1, 2) # (B, 80, T)
        
        print("  Vocoding...")
        audio = vocoder(voc_in)
        
        # Save
        audio = audio.squeeze().cpu().numpy()
        sf.write(args.output, audio, 16000)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
