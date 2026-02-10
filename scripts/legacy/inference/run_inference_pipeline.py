import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
import torch.nn as nn

# Redefine ConditionFuser locally - matches checkpoint (proj only, no smooth)
class ConditionFuser(nn.Module):
    def __init__(self, sem_dim, pro_dim, spk_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(sem_dim + pro_dim + spk_dim, out_dim)

    def forward(self, s, p, spk, target_frames):
        # s, p: (1, D, T) (Transposed inputs for interpolate logic)
        # spk: (1, D)
        
        # Upsample to target_frames
        if s.shape[-1] != target_frames:
            s_up = F.interpolate(s, size=target_frames, mode='linear', align_corners=False)
        else:
            s_up = s
            
        if p.shape[-1] != target_frames:
            p_up = F.interpolate(p, size=target_frames, mode='linear', align_corners=False)
        else:
            p_up = p
            
        # Back to (B, T, D) for cat
        s_up = s_up.transpose(1, 2) 
        p_up = p_up.transpose(1, 2)
        
        spk = spk.unsqueeze(1).expand(-1, target_frames, -1)
        
        # Concat and project
        cat = torch.cat([s_up, p_up, spk], dim=-1)
        x = self.proj(cat)  # (1, T, D)
        
        # Apply Manual Gaussian Smoothing to fix banding (missing in checkpoint)
        # Replicate training initialization: sigma=1.5, k=5
        # x is (B, T, D) -> need (B, 1, T, D) for Conv2d
        x = x.unsqueeze(1)
        
        # Create kernel on the fly
        k = 5
        sigma = 1.5
        arange = torch.arange(k, device=x.device).float() - k//2
        xv, yv = torch.meshgrid(arange, arange, indexing='ij')
        gauss = torch.exp(-0.5 * (xv**2 + yv**2) / sigma**2)
        kernel = (gauss / gauss.sum()).view(1, 1, k, k)
        
        # Apply smoothing
        # Padding=2 to maintain size
        x = F.conv2d(x, kernel, padding=2)
        
        x = x.squeeze(1) # (B, T, D)

        return x

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_spectrogram(data, path, title="Spectrogram"):
    viz = data.T.cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(viz, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Updated Mean/Std based on precompute analysis
MEL_MEAN = -2.9
MEL_STD = 4.3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", required=True, help="Path to input audio (16kHz or 24kHz)")
    parser.add_argument("--output_dir", default="inference_results")
    parser.add_argument("--flow_ckpt", default="checks/flow_dryrun/flow_epoch6.pt") # Latest Checkpoint
    parser.add_argument("--fuser_ckpt", default="checks/flow_dryrun/fuser_epoch6.pt")
    # DEFAULT TO THE GOOD VOCODER
    parser.add_argument("--vocoder_ckpt", default="checkpoints/checkpoints_bitnet_mel_v2/mel_vocoder_epoch82.pt")
    parser.add_argument("--factorizer_ckpt", default="checkpoints/checkpoints_stable/step_87000/decoder.pt")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Config
    config = load_config(args.config)
    config['model']['decoder']['fusion_dim'] = 100 # Override for Mel target

    # 2. Load Models
    print("Loading Models...")
    
    # Factorizer (for extracting conditions)
    factorizer = InformationFactorizerV2(config).to(device).eval()
    
    # Quantizers (needed to discretize factorizer output)
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device).eval()
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device).eval()
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device).eval()

    # Load Factorizer Weights (tricky part, similar to train_flow_matching)
    ckpt_dir = os.path.dirname(args.factorizer_ckpt)
    def load_helper(name, obj):
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location=device)
            new_d = {k.replace("_orig_mod.", ""): v for k,v in d.items()}
            obj.load_state_dict(new_d)
            print(f"Loaded {name}")
    
    load_helper("factorizer", factorizer)
    load_helper("sem_rfsq", sem_vq)
    load_helper("pro_rfsq", pro_vq)
    load_helper("spk_pq", spk_pq)

    # Flow Matching
    # Using Epoch 49 model (80-dim)
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 80
    flow_model = ConditionalFlowMatching(config).to(device).eval()
    flow_model.load_state_dict(torch.load(args.flow_ckpt, map_location=device))
    print(f"Loaded Flow Model form {args.flow_ckpt}")

    fuser = ConditionFuser(
        config['model']['semantic']['output_dim'],
        config['model']['prosody']['output_dim'],
        256, 
        80  # Match Flow model (80 dim)
    ).to(device).eval()
    fuser.load_state_dict(torch.load(args.fuser_ckpt, map_location=device))
    print(f"Loaded Fuser from {args.fuser_ckpt}")

    # BitVocoder
    vocoder = BitVocoder(
        input_dim=100, 
        dim=512, # Matches training args
        num_layers=12,
        num_res_blocks=2,
        hop_length=256 
    ).to(device).eval()
    vocoder.load_state_dict(torch.load(args.vocoder_ckpt, map_location=device))
    print(f"Loaded Vocoder from {args.vocoder_ckpt}")

    # 3. Process Audio
    print(f"Processing {args.input_audio}...")
    audio, sr = sf.read(args.input_audio)
    if audio.ndim > 1: audio = audio.mean(1)
    
    # Resample for 16k input (Factorizer expects 16k)
    audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    if sr != 16000:
        audio_16k = torchaudio.functional.resample(audio_t, sr, 16000)
    else:
        audio_16k = audio_t

    # 4. Extract HubERT & Factorize
    # Note: InformationFactorizerV2 expects HuBERT features input.
    # We need to load HuBERT model as well.
    print("Loading HuBERT...")
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()

    inputs = feature_extractor(audio_16k.squeeze(0).cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        # Config specifies Layer 9
        outputs = hubert(inputs.input_values.to(device), output_hidden_states=True)
        hubert_out = outputs.hidden_states[9] # Layer 9

    # Factorize
    with torch.no_grad():
        sem, pro, spk = factorizer(hubert_out)
        
        # Quantize (Simulate low bitrate channel)
        sem_q, _, _ = sem_vq(sem)
        pro_q, _, _ = pro_vq(pro)
        spk_q, _, _ = spk_pq(spk) # (1, num_groups) or (1, D) depending on impl

    # 5. Flow Generation
    # We need target length for Mel. 
    # Mel duration = Audio Duration (16k) * (24000/16000) / 256 ?
    # Let's approximate: 24kHz target sample rate.
    # target_samples = audio_duration * 24000
    # target_frames = target_samples / 256
    
    target_duration_sec = audio_16k.shape[1] / 16000
    target_frames = int(target_duration_sec * 24000 / 256)
    
    print(f"Generating Mel ({target_frames} frames)...")
    
    # Gaussian Blur for Smoothing Condition
    
    # Inputs are (1, T, D). Transpose to (1, D, T) for interpolation in Fuser
    sem_q = sem_q.transpose(1, 2)
    pro_q = pro_q.transpose(1, 2)
    
    with torch.no_grad():
        # Fuse
        cond = fuser(sem_q, pro_q, spk_q, target_frames) # (1, T, 512)
        
        
        # Conv1d smoothing is now inside fuser
        
        # Solve ODE
        mel_pred = flow_model.solve_ode(cond, steps=50, solver='euler') # (1, T, 100)
        
        # Denormalize Mel
        # Updated based on Dataset Stats (Mean=-2.9, Std=4.3)
        MEL_MEAN, MEL_STD = -2.9, 4.3
        mel_denorm = mel_pred * MEL_STD + MEL_MEAN
        
        print(f"DEBUG STATS:")
        print(f"  Flow Output (Norm):   Min={mel_pred.min():.2f}, Max={mel_pred.max():.2f}, Mean={mel_pred.mean():.2f}")
        print(f"  Denorm (Vocoder In): Min={mel_denorm.min():.2f}, Max={mel_denorm.max():.2f}, Mean={mel_denorm.mean():.2f}")
        
        # Save Spectrogram
        save_spectrogram(mel_denorm[0], os.path.join(args.output_dir, "generated_mel.png"), "Generated Mel")
        
        # GRIFFIN-LIM FALLBACK
        print("Generating Griffin-Lim fallback...")
        gl_transform = torchaudio.transforms.GriffinLim(n_fft=1024, n_iter=32, hop_length=256, power=2.0).to(device)
        inv_mel = torchaudio.transforms.InverseMelScale(n_stft=1024 // 2 + 1, n_mels=100, sample_rate=24000).to(device)
        
        # GL expects (..., F, T)
        # melan_denorm is (1, T, 100)
        mel_input_gl = mel_denorm.transpose(1, 2) # (1, 100, T)
        
        linear_mel = torch.exp(mel_input_gl)
        
        try:
            linear_spec = inv_mel(linear_mel)
            wav_gl = gl_transform(linear_spec)
            sf.write(os.path.join(args.output_dir, "reconstructed_gl.wav"), wav_gl.squeeze().cpu().numpy(), 24000)
            print("Saved reconstructed_gl.wav")
        except Exception as e:
            print(f"GL Failed: {e}")

    # 6. Vocode
    print("Vocoding...")
    with torch.no_grad():
        # Vocoder expects (B, 100, T)
        mel_input = mel_denorm.transpose(1, 2)
        audio_pred = vocoder(mel_input) # (1, 1, T_audio)

    # Save Audio
    output_path = os.path.join(args.output_dir, "reconstructed.wav")
    sf.write(output_path, audio_pred.squeeze().cpu().numpy(), 24000)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
