"""
End-to-End Pipeline Visualization: Phase 7 (Complex Flow Matching)
Input Audio -> BitNet Encoder -> Quantizers -> Decoder (Reconstructor) -> Flow Matching Head (Complex STFT) -> iSTFT -> Output Audio

Generates comparison spectrograms:
1. Original Audio
2. Flow Matching Regeneration (No-GAN)
"""

import os
import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.flow_matching import FlowMatchingHead
from transformers import Wav2Vec2FeatureExtractor, HubertModel

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    audio_path = "jakubie_16k.wav"
    config_path = "ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    bitnet_ckpt = "checkpoints_ultra200bps_large/step_14500/decoder.pt"
    flow_ckpt = "checkpoints_flow_matching/flow_epoch20.pt"
    
    config = load_config(config_path)
    
    # 1. Load Audio
    print(f"Loading {audio_path}...")
    import soundfile as sf
    audio_data, sr = sf.read(audio_path)
    audio = torch.tensor(audio_data, dtype=torch.float32)
    if audio.dim() == 1: audio = audio.unsqueeze(0)
    else: audio = audio.t().mean(0, keepdim=True)
    
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Cut to 5 seconds max for viz
    if audio.shape[1] > 16000 * 5:
        audio = audio[:, :16000 * 5]
        
    audio_batch = audio.to(device) # (1, T)
    
    # 2. Load Models
    print("Loading Models...")
    
    # Encoder / Feature Extractor
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    factorizer = InformationFactorizerV2(config).to(device)
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device)
    
    decoder = SpeechDecoderV2(config).to(device)
    
    # Load BitNet/Encoder Weights (using helper logic)
    print("Loading BitNet weights...")
    ckpt_dir = os.path.dirname(bitnet_ckpt)
    
    def load_part(name, obj):
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location=device)
            obj.load_state_dict({k.replace("_orig_mod.", ""): v for k,v in d.items()})
    
    load_part("factorizer", factorizer)
    load_part("sem_rfsq", sem_vq)
    load_part("pro_rfsq", pro_vq)
    load_part("spk_pq", spk_pq)
    
    # Load Decoder (Custom filtering for old checkpoints)
    d_ckpt = torch.load(bitnet_ckpt, map_location=device)
    d_state = d_ckpt['model_state_dict'] if 'model_state_dict' in d_ckpt else d_ckpt
    dec_state = decoder.state_dict()
    new_d = {}
    for k, v in d_state.items():
        k_clean = k.replace("_orig_mod.", "")
        if k_clean in dec_state:
            if v.shape == dec_state[k_clean].shape:
                new_d[k_clean] = v
    decoder.load_state_dict(new_d, strict=False)
    decoder.eval()
    
    # Flow Model
    print("Loading Flow Model...")
    flow_model = FlowMatchingHead(
        in_channels=1026, # Complex STFT
        cond_channels=512,
        hidden_dim=256,
        depth=6,
        heads=8
    ).to(device)
    
    if os.path.exists(flow_ckpt):
        try:
            flow_model.load_state_dict(torch.load(flow_ckpt, map_location=device))
            print("Flow Model loaded successfully.")
        except Exception as e:
            print(f"Error loading flow model: {e}")
            return
    else:
        print(f"Flow checkpoint not found: {flow_ckpt}")
        return
        
    flow_model.eval()
    
    # 3. Pipeline Execution
    print("Executing Pipeline...")
    with torch.no_grad():
        # A. Audio -> Features (HuBERT)
        inputs = hubert_processor(audio_batch.cpu().squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        features = hubert_out.hidden_states[config['model'].get('hubert_layer', 9)]
        
        # B. Features -> Codes (Quantization)
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        # C. Codes -> BitNet Cond (Decoder Reconstructor)
        cond = decoder.reconstructor(sem_z, pro_z, spk_z) # (B, T, 512)
        cond = cond.transpose(1, 2) # (B, 512, T)
        
        # D. Flow Matching Generation
        steps = 20
        print(f"Generating Flow (steps={steps})...")
        B, C, T = cond.shape
        x = torch.randn(B, 1026, T, device=device) # x0
        dt = 1.0 / steps
        
        for i in range(steps):
            t_scalar = i / steps
            t = torch.full((B,), t_scalar, device=device)
            v = flow_model(x, t, cond)
            x = x + v * dt
            
        # x is Complex Spectrogram (Real, Imag)
        real, imag = torch.chunk(x, 2, dim=1)
        stft_gen = torch.complex(real, imag)
        
        # E. iSTFT Synthesis
        n_fft = 1024
        hop_length = 320
        win_length = 1024
        window = torch.hann_window(win_length).to(device)
        
        audio_gen = torch.istft(
            stft_gen,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=False
        )
        
        # Match length
        min_len = min(audio_batch.shape[1], audio_gen.shape[1])
        audio_batch = audio_batch[:, :min_len]
        audio_gen = audio_gen[:, :min_len]
        
    print("Generation complete.")
    
    # 4. Save Audio
    sf.write("final_flow_test.wav", audio_gen.cpu().squeeze().numpy(), 16000)
    print("Saved final_flow_test.wav")
    
    # 5. Visualization (Spectrogram Comparison)
    print("Generating Spectrogram Comparison...")
    
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=1024, win_length=1024, hop_length=320, power=2.0
    ).to(device)
    
    def to_db(spec):
        return 10 * torch.log10(spec + 1e-9).cpu().numpy()
        
    orig_spec = to_db(spec_transform(audio_batch).squeeze())
    gen_spec  = to_db(spec_transform(audio_gen).squeeze())
    
    # Limit frequency range for visual clarity (0-8kHz)
    orig_spec = orig_spec[:513, :]
    gen_spec = gen_spec[:513, :]
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    im1 = axs[0].imshow(orig_spec, origin='lower', aspect='auto', cmap='magma', vmin=-100, vmax=0)
    axs[0].set_title("Original Audio (Ground Truth)")
    axs[0].set_ylabel("Freq Bin")
    fig.colorbar(im1, ax=axs[0], format='%+2.0f dB')
    
    im2 = axs[1].imshow(gen_spec, origin='lower', aspect='auto', cmap='magma', vmin=-100, vmax=0)
    axs[1].set_title(f"BitNet Flow Matching (No-GAN, Epoch 20) - 1.58MB Model")
    axs[1].set_ylabel("Freq Bin")
    axs[1].set_xlabel("Time Frame")
    fig.colorbar(im2, ax=axs[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig("flow_pipeline_comparison.png", dpi=150)
    print("Saved flow_pipeline_comparison.png")

if __name__ == "__main__":
    main()
