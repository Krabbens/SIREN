import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuser
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.tiny_hubert import TinyHubert
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Flow Matching checkpoint")
    parser.add_argument("--bitnet_checkpoint", required=True, help="BitNet checkpoint (for encoder)")
    parser.add_argument("--tiny_hubert_path", required=True, help="Distilled TinyHubert checkpoint")
    parser.add_argument("--vocoder_checkpoint", required=True, help="MelVocoder checkpoint")
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--output_wav", default="outputs/jakubie_tiny_hubert.wav")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps")
    parser.add_argument("--config", default="src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
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
    sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device)
    
    ckpt_dir = os.path.dirname(args.bitnet_checkpoint)
    def load_part(name, obj):
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location=device)
            obj.load_state_dict({k.replace("_orig_mod.", ""): v for k,v in d.items()}, strict=True)
            print(f"Loaded {name}")
    
    load_part("factorizer", factorizer)
    load_part("sem_rfsq", sem_vq)
    load_part("pro_rfsq", pro_vq)
    load_part("spk_pq", spk_pq)
    
    # --- Load Flow Model ---
    print("Loading Flow Model...")
    
    # Load checkpoint first to detect dimensions
    flow_ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(flow_ckpt, dict) and 'model_state_dict' in flow_ckpt:
        flow_ckpt = flow_ckpt['model_state_dict']
    
    # Auto-detect fusion_dim from checkpoint (input_proj.weight shape is [hidden_dim, fusion_dim])
    input_proj_key = next((k for k in flow_ckpt.keys() if 'input_proj.weight' in k), None)
    if input_proj_key:
        fusion_dim = flow_ckpt[input_proj_key].shape[1]
        hidden_dim = flow_ckpt[input_proj_key].shape[0]
        print(f"Auto-detected from checkpoint: fusion_dim={fusion_dim}, hidden_dim={hidden_dim}")
    else:
        fusion_dim = 80
        hidden_dim = 512
        print(f"Could not auto-detect, using defaults: fusion_dim={fusion_dim}, hidden_dim={hidden_dim}")
    
    config['model']['decoder']['fusion_dim'] = fusion_dim
    config['model']['decoder']['hidden_dim'] = hidden_dim
    flow_model = ConditionalFlowMatching(config).to(device)
    
    try:
        flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()})
    except RuntimeError as e:
        print(f"Warning: strict loading failed ({e}), retrying non-strict...")
        flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()}, strict=False)
        
    flow_model.eval()
    
    # --- Load Condition Fuser ---
    flow_dir = os.path.dirname(args.model_path)
    flow_basename = os.path.basename(args.model_path)
    fuser_path = os.path.join(flow_dir, flow_basename.replace("flow_", "fuser_"))
    
    fuser_in_dim = 272 
    fuser_out_dim = 512
    
    if os.path.exists(fuser_path):
        fuser_ckpt = torch.load(fuser_path, map_location=device)
        f_weight = fuser_ckpt['proj.weight']
        fuser_in_dim = f_weight.shape[1]
        fuser_out_dim = f_weight.shape[0]
        print(f"Fuser dimensions: {fuser_in_dim} -> {fuser_out_dim}")
    else:
        print(f"Warning: Fuser checkpoint not found at {fuser_path}")
    
    fuser = ConditionFuser(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=fuser_out_dim).to(device)
    if os.path.exists(fuser_path):
        fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fuser_ckpt.items()}, strict=False)
    fuser.eval()

    # --- Process Audio ---
    # use scipy for robustness
    from scipy.io import wavfile
    try:
        sr, wav_data = wavfile.read(args.input_wav)
        wav = torch.tensor(wav_data, dtype=torch.float32)
    except:
        wav, sr = torchaudio.load(args.input_wav)
    
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() == 2:
        if wav.shape[0] > wav.shape[1]: 
             wav = wav.t()
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        
    wav = wav / (torch.abs(wav).max() + 1e-6)
    wav = wav.to(device)
    
    target_mel_len = wav.shape[1] // 320

    with torch.no_grad():
        features = tiny_hubert(wav) 
        
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
        print(f"Generating Mel spectrogram with {args.steps} steps...")
        mel = flow_model.solve_ode(cond, steps=args.steps, solver='euler')
        
    # --- Denormalize ---
    MEL_MEAN, MEL_STD = -5.0, 2.0
    mel = mel * MEL_STD + MEL_MEAN
    
    # Training uses log10 (train_flow_matching.py L177), Vocoder expects log10 - no conversion needed
    mel = torch.clamp(mel, min=-12.0, max=3.0)  # Range from train_flow_matching.py
    
    print(f"Flow Mel Shape: {mel.shape}")
    
    # Vocoder expects (B, T, 80)
    # Check if we need transpose
    if mel.shape[1] == 80:
         # (B, 80, T) -> (B, T, 80)
         mel = mel.transpose(1, 2)
    
    print(f"Vocoder Input Shape: {mel.shape}")

    # Save PNG
    plt.figure(figsize=(10, 4))
    # Transpose back to (Freq, Time) for plotting
    to_plot = mel.cpu().squeeze().transpose(0, 1).numpy()
    plt.imshow(to_plot, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(f"Final Mel (TinyHubert + Flow Epoch {args.model_path.split('_')[-1]})")
    plt.savefig(args.output_wav.replace(".wav", ".png"))
    
    # --- Vocoder ---
    print("Synthesizing Audio (MelVocoderBitNet)...")
    vocoder = MelVocoderBitNet().to(device)
    voc_ckpt = torch.load(args.vocoder_checkpoint, map_location=device)
    vocoder.load_state_dict(voc_ckpt)
    vocoder.eval()
    
    with torch.no_grad():
        # Vocoder takes (B, 80, T) -> Wait, forward says x = mel.transpose(1, 2), assumes input (B, T, 80)
        # Verify MelVocoder.forward:
        # def forward(self, mel):
        #     x = mel.transpose(1, 2) # (B, 80, T)
        #     x = self.input_conv(x)
        # So yes, it expects (B, T, 80).
        wave = vocoder(mel)
        
    import soundfile as sf
    sf.write(args.output_wav, wave.cpu().squeeze().numpy(), 16000)
    print(f"Saved to {args.output_wav}")

if __name__ == "__main__":
    main()
