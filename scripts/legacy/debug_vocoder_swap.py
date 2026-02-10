
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

# Imports (Copied from inference_v2.py)
from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoderTiny, MicroEncoder
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def load_models(checkpoint_dir, config_path, device, vocoder_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    tag = "ep30" # Force Epoch 30
    
    # ... (Loading Logic simplified for brevity, assuming standard V2 components) ...
    # 1. Encoder
    encoder = MicroEncoder(hidden_dim=256, output_dim=768, num_layers=4).to(device)
    enc_ckpt = torch.load(os.path.join(checkpoint_dir, f"encoder_{tag}.pt"), map_location=device)
    encoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in enc_ckpt.items()})
    
    # 2. Factorizer
    config['model']['prosody']['output_dim'] = 4
    factorizer = InformationFactorizerV2(config).to(device)
    fac_ckpt = torch.load(os.path.join(checkpoint_dir, f"factorizer_{tag}.pt"), map_location=device)
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fac_ckpt.items()})
    
    # 3. Quantizers
    full_levels = config['model']['fsq_levels']
    pro_fsq_levels = full_levels[:4]
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=pro_fsq_levels, num_levels=2, input_dim=4).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    # 4. Fuser
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=4, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fus_ckpt = torch.load(os.path.join(checkpoint_dir, f"fuser_{tag}.pt"), map_location=device)
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fus_ckpt.items()})
    
    # 5. Flow
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow = ConditionalFlowMatching(config).to(device)
    flow_ckpt = torch.load(os.path.join(checkpoint_dir, f"flow_{tag}.pt"), map_location=device)
    flow.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()})
    
    # 6. Vocoder (OLD PHASE 1)
    print(f"Loading OLD Vocoder from {vocoder_path}...")
    # NOTE: Old vocoder might be MelVocoderBitNet or BitVocoder.
    # We'll use the logic from inference_siamese.py to detect.
    voc_ckpt = torch.load(vocoder_path, map_location=device)
    state_dict = voc_ckpt
    if isinstance(voc_ckpt, dict) and 'model_state_dict' in voc_ckpt: 
        state_dict = voc_ckpt['model_state_dict']
        
    if 'conv_in.weight' in state_dict:
        # BitVocoder
        w_shape = state_dict['conv_in.weight'].shape
        input_dim = w_shape[1]
        vocoder = BitVocoder(input_dim=input_dim, dim=w_shape[0], num_layers=4, num_res_blocks=2).to(device)
        print("  Detected BitVocoder")
    else:
        # MelVocoderBitNet
        vocoder = MelVocoderBitNet().to(device)
        print("  Detected MelVocoderBitNet")
        
    vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
    
    return encoder, factorizer, sem_vq, pro_vq, spk_pq, fuser, flow, vocoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output_swap.wav")
    parser.add_argument("--ckpt_dir", default="checkpoints/microencoder_v2")
    parser.add_argument("--vocoder_ckpt", default="checkpoints/vocoder_mel/vocoder_latest.pt")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder, factorizer, sem_vq, pro_vq, spk_pq, fuser, flow, vocoder = load_models(args.ckpt_dir, "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml", device, args.vocoder_ckpt)
    
    wav, sr = sf.read(args.input)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.ndim > 1: wav = wav.mean(dim=-1)
    if sr != 16000:
        import torchaudio.functional as FA
        wav = FA.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    wav = wav / (wav.abs().max() + 1e-6)
    wav = wav.unsqueeze(0).to(device)
    
    print(f"Inference on {args.input}...")
    with torch.no_grad():
        feat = encoder(wav)
        s, p, spk = factorizer(feat)
        sz, _, _ = sem_vq(s)
        pz, _, _ = pro_vq(p)
        spkz, _, _ = spk_pq(spk)
        target_len = sz.shape[1] * 2
        c = fuser(sz, pz, spkz, target_len)
        pred = flow.solve_ode(c, steps=50, solver='midpoint', cfg_scale=1.5) # (B, T, 80)
        
        # --- NORM ADJUSTMENT ---
        # Current Pred (V2): Normalized with MEAN=-5.0, STD=3.5
        # Old Vocoder (Phase 1): Expects MEAN=-6.0, STD=3.5 (Inferred)
        
        # pred_real_log_mel = pred * 3.5 - 5.0
        # voc_in = (pred_real_log_mel - (-6.0)) / 3.5 
        #        = (pred * 3.5 - 5.0 + 6.0) / 3.5 
        #        = pred + (1.0 / 3.5)
        
        shift = 1.0 / 3.5
        print(f"  Adjusting normalization: Shift += {shift:.4f} (Mapping -5.0 to -6.0 space)")
        
        voc_in = pred + shift
        
        # Check vocoder input dim
        # MelVocoderBitNet usually expects 100 bands? Or 80?
        # If MelVocoderBitNet and trained on 80, okay.
        # But wait, did we verify Phase 1 was 80 bands?
        # inference_siamese.py line 190 says n_mels=80.
        # So assumed 80.
        
        # Transpose for Vocoder based on type
        # MelVocoderBitNet expects (B, T, C) and transposes internally
        # BitVocoder expects (B, C, T) directly
        
        is_mel_vocoder = hasattr(vocoder, 'input_conv') # MelVocoderBitNet has input_conv
        
        if not is_mel_vocoder:
             # BitVocoder: Needs (B, C, T)
             voc_in = voc_in.transpose(1, 2)
        else:
             # MelVocoderBitNet: Keeps (B, T, C)
             pass
        
        # Resize if needed (e.g. if Old Vocoder was 100 bands)
        target_dim = 80
        if hasattr(vocoder, 'input_conv'): # MelVocoderBitNet
             target_dim = vocoder.input_conv.in_channels
        elif hasattr(vocoder, 'conv_in'): # BitVocoder
             target_dim = vocoder.conv_in.in_channels
             
        if target_dim != 80:
            print(f"  Resizing 80 -> {target_dim} bands...")
            voc_in = voc_in.unsqueeze(1)
            voc_in = F.interpolate(voc_in, size=(target_dim, voc_in.shape[-1]), mode='bilinear', align_corners=False)
            voc_in = voc_in.squeeze(1)
            
        print("  Vocoding...")
        audio = vocoder(voc_in)
        audio = audio.squeeze().cpu().numpy()
        sf.write(args.output, audio, 16000)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
