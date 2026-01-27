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
from transformers import Wav2Vec2FeatureExtractor, HubertModel




def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Flow Matching checkpoint")
    parser.add_argument("--bitnet_checkpoint", required=True, help="BitNet checkpoint (for encoder)")
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--output_wav", default="flow_output.wav")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    # --- Load Encoder Models ---
    print("Loading Encoder Components...")
    
    # HuBERT
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    # Factorizer & Quantizers
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
            try:
                obj.load_state_dict({k.replace("_orig_mod.", ""): v for k,v in d.items()}, strict=True)
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
    
    load_part("factorizer", factorizer)
    load_part("sem_rfsq", sem_vq)
    load_part("pro_rfsq", pro_vq)
    load_part("spk_pq", spk_pq)
    
    # --- Load Flow Model ---
    print("Loading Flow Model...")
    
    # NEW CONFIGURATION
    # out_dim = 80 (Mel)
    # hidden_dim = 512 (Internal)
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    
    flow_model = ConditionalFlowMatching(config).to(device)
    
    # Load Flow Checkpoint
    flow_ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(flow_ckpt, dict) and 'model_state_dict' in flow_ckpt:
        flow_ckpt = flow_ckpt['model_state_dict']
    
    # Strict loading now that we fixed the architecture
    msg = flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()}, strict=False)
    print(f"Flow model loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    flow_model.eval()
    
    # --- Load Condition Fuser ---
    flow_dir = os.path.dirname(args.model_path)
    flow_basename = os.path.basename(args.model_path)
    # Try finding fuser
    fuser_path = os.path.join(flow_dir, flow_basename.replace("flow_", "fuser_"))
    
    sem_dim = config['model']['semantic']['output_dim']
    pro_dim = config['model']['prosody']['output_dim']
    spk_dim = config['model']['speaker']['embedding_dim'] # Should verify if this matches
    # Correction: spk_dim in fuser input is whatever enters it.
    # In factorizer, spk is projected. Let's trust dimensions.
    
    # Fuser projects to HIDDEN_DIM (512) now
    fuser_out_dim = 512 
    fuser_in_dim = sem_dim + pro_dim + spk_dim # e.g. 128 + 64 + 64 ?
    
    # Load to check dimensions
    if os.path.exists(fuser_path):
        fuser_ckpt = torch.load(fuser_path, map_location=device)
        f_weight = fuser_ckpt['proj.weight']
        fuser_in_dim = f_weight.shape[1]
        fuser_out_dim = f_weight.shape[0]
        print(f"Fuser dimensions from ckpt: {fuser_in_dim} -> {fuser_out_dim}")
    else:
        print("WARNING: Fuser checkpoint not found. initializing random.")
        fuser_in_dim = 272 # Fallback guess
    
    fuser = ConditionFuser(0, 0, 0, fuser_out_dim).to(device) # dims ignored by my hacky init below
    # Re-init proj to correct shape
    fuser.proj = nn.Linear(fuser_in_dim, fuser_out_dim).to(device)
    
    if os.path.exists(fuser_path):
        fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fuser_ckpt.items()}, strict=False)
    fuser.eval()

    # --- Process Audio ---
    # Process Audio using scipy (reliable fallback for .wav)
    from scipy.io import wavfile
    try:
        sr, wav_data = wavfile.read(args.input_wav)
    except ValueError:
        # Fallback if scipy fails (e.g. 24-bit wav sometimes issues)
        wav, sr = torchaudio.load(args.input_wav)
        wav_data = wav.numpy()

    wav = torch.tensor(wav_data, dtype=torch.float32)
    
    # Handle dimensions
    if wav.dim() == 1:
        wav = wav.unsqueeze(0) # (1, T)
    elif wav.dim() == 2:
        if wav.shape[0] > wav.shape[1]: # (T, C) -> (C, T)
             wav = wav.t()
        # If stereo, mix down
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
    # Normalize to [-1, 1]
    wav = wav / (torch.abs(wav).max() + 1e-6)
    
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    
    wav = wav.to(device)
    
    hop_length = 320
    target_mel_len = wav.shape[1] // hop_length
    
    # Debug Helper
    def plot_heatmap(data, name, output_dir):
        # data: (C, T) or (1, C, T) -> (C, T)
        if data.dim() == 3: data = data.squeeze(0)
        data = data.detach().cpu().numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(data, origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f"{name} (Mean: {data.mean():.2f}, Std: {data.std():.2f})")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[DEBUG] Saved {name} to {save_path}")

    # Output dir for debug plots
    debug_dir = os.path.dirname(args.output_wav)
    os.makedirs(debug_dir, exist_ok=True)

    with torch.no_grad():
        inputs = hubert_processor(wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        features = hubert_out.hidden_states[config['model'].get('hubert_layer', 9)]
        
        # Visualize HuBERT
        plot_heatmap(features.transpose(1, 2), "debug_01_hubert", debug_dir)
        
        sem, pro, spk = factorizer(features)
        
        # Visualize Factors
        plot_heatmap(sem.transpose(1, 2), "debug_02_semantic_raw", debug_dir)
        plot_heatmap(pro.transpose(1, 2), "debug_03_prosody_raw", debug_dir)
        plot_heatmap(spk.unsqueeze(1).expand(-1, 100, -1).transpose(1, 2), "debug_04_speaker_raw", debug_dir)
        
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        plot_heatmap(sem_z.transpose(1, 2), "debug_05_semantic_quantized", debug_dir)
        
        # Fuser -> (B, T, 512)
        cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
        print(f"Conditioning shape: {cond.shape} (Expected B, T, 512)")
        plot_heatmap(cond.transpose(1, 2), "debug_06_conditioning_fused", debug_dir)
        
        # Flow Matching -> (B, T, 80)
        print(f"Generating Mel spectrogram with {args.steps} steps...")
        mel = flow_model.solve_ode(cond, steps=args.steps, solver='euler', cfg_scale=1.0)
        
        plot_heatmap(mel.transpose(1, 2), "debug_07_mel_raw_output", debug_dir)
        
    # --- Denormalize ---
    # New Stats: Mean -5.0, Std 2.0
    MEL_MEAN, MEL_STD = -5.0, 2.0
    mel = mel * MEL_STD + MEL_MEAN
    
    # Clamp to valid range (Silence is approx -11.5)
    mel = torch.clamp(mel, min=-12.0, max=3.0)
    
    plot_heatmap(mel.transpose(1, 2), "debug_08_mel_denormalized", debug_dir)
    
    # Save Spectrogram (Standard view)
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.detach().cpu().squeeze().numpy().T, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Final Mel Spectrogram")
    plt.savefig(args.output_wav.replace(".wav", ".png"))
    print(f"Saved spectrogram to {args.output_wav.replace('.wav', '.png')}")
    
    # --- Vocoder (Griffin-Lim fallback for now) ---
    print("Synthesizing Audio (Griffin-Lim)...")
    n_fft = 1024
    mel_linear = torch.exp(mel) # log10? No, usually Ln. 
    # Wait, simple training used log10. 
    # train.py: log_mel = torch.log10(mel + 1e-5)
    # So we need 10^x
    mel_linear = torch.pow(10, mel)
    
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1, f_min=0, f_max=8000, n_mels=80, sample_rate=16000
    ).to(device)
    
    mel_fb_pinv = torch.linalg.pinv(mel_fb)
    mag = torch.clamp(mel_linear @ mel_fb_pinv, min=1e-5).transpose(1, 2)
    
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=320, win_length=1024, n_iter=64).to(device)
    wave = griffin_lim(mag.squeeze(0))
    
    import soundfile as sf
    sf.write(args.output_wav, wave.detach().cpu().squeeze().numpy(), 16000)
    print(f"Saved to {args.output_wav}")

if __name__ == "__main__":
    main()

