import os
import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultra_low_bitrate_codec.models.flow_matching import FlowMatchingHead
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
    parser.add_argument("--bitnet_checkpoint", required=True, help="BitNet checkpoint")
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--output_wav", default="flow_output.wav")
    parser.add_argument("--steps", type=int, default=10, help="ODE steps")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    # Load Models
    print("Loading BitNet...")
    bitnet = SpeechDecoderV2(config).to(device)
    # Load bitnet weights logic (simplified)
    # ... (omitted for brevity, assume user provides full checkpoint or I reuse loading logic)
    # Actually need to load properly.
    
    ckpt = torch.load(args.bitnet_checkpoint, map_location=device)
    # Load Decoder logic (robust)
    d_state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    bitnet_state = bitnet.state_dict()
    
    # Filter
    new_d = {}
    for k, v in d_state.items():
        k_clean = k.replace("_orig_mod.", "")
        if k_clean in bitnet_state:
            if v.shape == bitnet_state[k_clean].shape:
                new_d[k_clean] = v
    
    bitnet.load_state_dict(new_d, strict=False)
    bitnet.eval()
    
    # Load Encoder side (need for inference from raw audio)
    # For now, let's assume we implement the full loading logic or import it
    # Reuse loading logic from precompute script? 
    # To keep this script simple, let's just assume we have the 'cond' already or 
    # implement the basic encoder pass.
    
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
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
                print(f"Loaded {name} (strict=True)")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
    
    load_part("factorizer", factorizer)
    load_part("sem_rfsq", sem_vq)
    load_part("pro_rfsq", pro_vq)
    load_part("spk_pq", spk_pq)
    
    # Load Flow Model
    print("Loading Flow Model...")
    flow_model = FlowMatchingHead(
        in_channels=1026,
        cond_channels=512,
        hidden_dim=256,
        depth=6,
        heads=8
    ).to(device)
    flow_model.load_state_dict(torch.load(args.model_path, map_location=device))
    flow_model.eval()
    
    # Process Audio
    import soundfile as sf
    wav, sr = sf.read(args.input_wav)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() == 1: wav = wav.unsqueeze(0)
    else: wav = wav.t().mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        
    wav = wav.to(device)
    
    with torch.no_grad():
        # Get Cond
        inputs = hubert_processor(wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        features = hubert_out.hidden_states[config['model'].get('hubert_layer', 9)]
        
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        cond = bitnet.reconstructor(sem_z, pro_z, spk_z) # (B, T, 512)
        cond = cond.transpose(1, 2) # (B, 512, T)
        
        # DEBUG: Save cond for analysis
        torch.save(cond, "debug_cond.pt")
        print("Saved debug_cond.pt")
        
    # Flow Matching Inference (Euler)
    print(f"Generating with {args.steps} steps...")
    sigma_min = 1e-4
    B, C, T = cond.shape
    # Complex STFT dim (513 Real + 513 Imag)
    ComplexDim = 1026
    
    # x0 ~ N(0, 1)
    x = torch.randn(B, ComplexDim, T, device=device)
    
    dt = 1.0 / args.steps
    
    for i in range(args.steps):
        t_scalar = i / args.steps
        t = torch.full((B,), t_scalar, device=device)
        
        # Predict vector field
        v = flow_model(x, t, cond)
        
        # Euler Step
        x = x + v * dt
        
    # x is now generated Mel (Log Magnitude)
    # No scaling (data is unit variance)
    
    # Clamp to avoid explosions
    x = torch.clamp(x, min=-100.0, max=100.0)

    # Convert to Audio
    # For now: Griffin-Lim
    # x is now generated Complex STFT
    # (B, 1026, T) -> (B, 513, T, 2)
    print("Synthesizing Audio (iSTFT)...")
    
    real, imag = torch.chunk(x, 2, dim=1) # Split channels
    stft = torch.complex(real, imag) # (B, 513, T)
    
    # iSTFT parameters must match precompute
    n_fft = 1024
    hop_length = 320
    win_length = 1024
    
    wave = torch.istft(
        stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(device),
        return_complex=False
    )
    
    sf.write(args.output_wav, wave.detach().cpu().squeeze().numpy(), 16000)
    print(f"Saved to {args.output_wav}")
    
    # Visualize Magnitude Spectrogram
    mag = torch.abs(stft)
    mel_transform = torchaudio.transforms.MelScale(
        n_mels=80, sample_rate=16000, n_stft=n_fft // 2 + 1
    ).to(device)
    mel = mel_transform(mag)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.detach().cpu().squeeze().numpy(), origin='lower', aspect='auto')
    plt.title("Generated Complex Spectrogram (Mag)")
    plt.colorbar()
    plt.savefig(args.output_wav.replace(".wav", ".png"))

if __name__ == "__main__":
    main()
