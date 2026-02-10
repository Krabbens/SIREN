import torch
import torchaudio
import matplotlib.pyplot as plt
import sys
import os
import yaml
import numpy as np
import torch.nn.functional as F

# SIREN imports
sys.path.append(os.getcwd())
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel


def save_spectrogram(audio, filename, sr=16000):
    audio = audio.float().cpu().numpy().squeeze()
    
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap='viridis')
    plt.title('Reconstructed Spectrogram')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved spectrogram to {filename}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    # step_122000 corresponds to adaptation step 2000
    checkpoint_dir = "checkpoints/checkpoints_factorizer_tiny_frozen/step_122000"
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_tiny_adapt.yaml"
    feature_path = "data/features_tiny_train/1034_1034_121119_000010_000004.pt"
    
    output_wav = "output_step_2000.wav"
    output_png = "output_step_2000.png"

    # LOad Config
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize models
    print("Initializing models...")
    factorizer = InformationFactorizerV2(config).to(device)
    decoder = SpeechDecoderV2(config).to(device)
    
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
    
    # Load Weights
    print(f"Loading checkpoints from {checkpoint_dir}...")
    
    # Helper to clean keys
    def clean_keys(sd, prefix=""):
        return {k.replace("_orig_mod.", "").replace(prefix, ""): v for k, v in sd.items()}

    factorizer.load_state_dict(clean_keys(torch.load(f"{checkpoint_dir}/factorizer.pt", map_location=device)), strict=False)
    decoder.load_state_dict(clean_keys(torch.load(f"{checkpoint_dir}/decoder.pt", map_location=device)), strict=False)
    sem_vq.load_state_dict(clean_keys(torch.load(f"{checkpoint_dir}/sem_rfsq.pt", map_location=device)), strict=False)
    pro_vq.load_state_dict(clean_keys(torch.load(f"{checkpoint_dir}/pro_rfsq.pt", map_location=device)), strict=False)
    spk_pq.load_state_dict(clean_keys(torch.load(f"{checkpoint_dir}/spk_pq.pt", map_location=device)), strict=False)
    
    factorizer.eval()
    decoder.eval()
    
    # Load Feature
    print(f"Loading feature from {feature_path}")
    feature = torch.load(feature_path).to(device) # Expected (T, C) or (C, T)
    
    print(f"Feature shape: {feature.shape}")
    if feature.dim() == 2:
        # Check dim 768
        if feature.shape[0] == 768:
            feature = feature.transpose(0, 1) # (T, 768)
        feature = feature.unsqueeze(0) # (1, T, 768)
        
    # Factorizer usually expects (B, T, C) or (B, C, T) depending on implementation
    # Let's inspect config
    # TinyHubert features are usually (B, T, D)
    # The default conv1d usually expects (B, D, T)
    # Let's try to infer from error or just transpose if needed. 
    # BUT encoder is InformationFactorizerV2
    # Standard PyTorch Conv1d wants (N, C, L).
    # Factorizer V2 expects (B, T, C) based on error message
    # feature = feature.transpose(1, 2) 
    # print(f"Input to factorizer: {feature.shape}")
    print(f"Feature path: {feature_path}")
    print(f"Feature stats: Min={feature.min():.4f}, Max={feature.max():.4f}, Mean={feature.mean():.4f}, Std={feature.std():.4f}")
    
    print("Running Inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sem, pro, spk = factorizer(feature)
            print(f"Latents: Sem={sem.shape}, Pro={pro.shape}, Spk={spk.shape}")
            print(f"Sem Stats: Min={sem.min():.4f}, Max={sem.max():.4f}, Mean={sem.mean():.4f}, Std={sem.std():.4f}")
            print(f"Pro Stats: Min={pro.min():.4f}, Max={pro.max():.4f}, Mean={pro.mean():.4f}, Std={pro.std():.4f}")
            
            sem_z, _, sem_idx = sem_vq(sem)
            pro_z, _, pro_idx = pro_vq(pro)
            spk_z, _, _ = spk_pq(spk)
            
            print(f"Quantized: SemIndices={sem_idx.shape}, ProIndices={pro_idx.shape}")
            print(f"Unique Sem Indices: {torch.unique(sem_idx).tolist()}")
            print(f"Unique Pro Indices: {torch.unique(pro_idx).tolist()}")
            
            audio_hat = decoder(sem_z, pro_z, spk_z)
            print(f"Output Audio: {audio_hat.shape}")
            print(f"Audio Stats: Min={audio_hat.min():.4f}, Max={audio_hat.max():.4f}, Mean={audio_hat.mean():.4f}, Std={audio_hat.std():.4f}")

    # Save output (flattened)
    if audio_hat.dim() == 3: audio_hat = audio_hat.squeeze(1)
    
    # Save WAV
    try:
        from scipy.io import wavfile
        # Normalize to 16-bit PCM for compatibility
        audio_np = audio_hat.float().cpu().numpy().flatten()
        
        # Check for NaN/Inf
        if np.any(np.isnan(audio_np)) or np.any(np.isinf(audio_np)):
            print("WARNING: Audio contains NaN or Inf!")
            audio_np = np.nan_to_num(audio_np)
            
        audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-6)
        wavfile.write(output_wav, 16000, (audio_np * 32767).astype(np.int16))
    except ImportError:
        print("Scipy not found. Trying soundfile as valid fallback...")
        import soundfile as sf
        sf.write(output_wav, audio_hat.float().cpu().numpy(), 16000)
    
    print(f"Saved audio to {output_wav}")
    
    # Save Spectrogram
    save_spectrogram(audio_hat, output_png)
    
    # Show log info
    # Estimate bitrate
    duration = audio_hat.shape[-1] / 16000.0
    sem_bits = sem_idx.shape[1] * np.log2(256) # Approx if entropy not used, but indices are codebook indices?
    # Actually FSQ is different.
    # Sem idx returns indices into codebook.
    # Let's just say we ran it.
    
if __name__ == "__main__":
    main()
