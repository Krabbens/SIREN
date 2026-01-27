#!/usr/bin/env python3
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
import os
import sys
from transformers import Wav2Vec2Processor, HubertModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_spectrogram(y, y_hat, title, save_path):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Original
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)(y.cpu())
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec).squeeze().numpy()
    axs[0].imshow(spec_db, aspect='auto', origin='lower')
    axs[0].set_title(f"Original: {title}")
    
    # Reconstructed
    spec_hat = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)(y_hat.cpu())
    spec_hat_db = torchaudio.transforms.AmplitudeToDB()(spec_hat).squeeze().numpy()
    axs[1].imshow(spec_hat_db, aspect='auto', origin='lower')
    axs[1].set_title("Reconstructed")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input audio file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/checkpoints_multispeaker")
    parser.add_argument("--step", type=int, default=12500)
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/multispeaker_optimized.yaml")
    parser.add_argument("--output_dir", type=str, default="inference_results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Load Models
    print("Loading models...")
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
    ckpt_dir = args.checkpoint_dir
    step = args.step
    try:
        factorizer.load_state_dict(torch.load(f"{ckpt_dir}/factorizer_{step}.pt", map_location=device))
        decoder.load_state_dict(torch.load(f"{ckpt_dir}/decoder_{step}.pt", map_location=device))
        sem_vq.load_state_dict(torch.load(f"{ckpt_dir}/sem_rfsq_{step}.pt", map_location=device))
        pro_vq.load_state_dict(torch.load(f"{ckpt_dir}/pro_rfsq_{step}.pt", map_location=device))
        spk_pq.load_state_dict(torch.load(f"{ckpt_dir}/spk_pq_{step}.pt", map_location=device))
        print("Model weights loaded!")
    except FileNotFoundError as e:
        print(f"Error loading checkpoints: {e}")
        return

    factorizer.eval()
    decoder.eval()
    
    # Load HuBERT for features
    print("Loading HuBERT...")
    from transformers import Wav2Vec2FeatureExtractor
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    hubert_model.eval()

    # Process Audio
    print(f"Processing {args.input}...")
    import soundfile as sf
    audio_np, sr = sf.read(args.input)
    audio = torch.from_numpy(audio_np).float()
    
    # Ensure [Channels, Time]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.t() # soundfile returns [Time, Channels]
        
    # Mono & Resample
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Clip to reasonable length if too long (e.g. 30s) to avoid OOM or wait times
    MAX_SEC = 30
    if audio.shape[1] > 16000 * MAX_SEC:
        print(f"Trimming audio to {MAX_SEC}s")
        audio = audio[:, :16000 * MAX_SEC]

    audio_dev = audio.to(device)
    
    # Extract Features
    with torch.no_grad():
        inputs = hubert_processor(audio_dev.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        outputs = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        # Config says layer 9? Let's check config. 
        # config['model']['hubert_layer'] is typically 9.
        layer = config['model'].get('hubert_layer', 9)
        features = outputs.hidden_states[layer] 
        # [Batch, Time, Dim]
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sem, pro, spk = factorizer(features)
            sem_z, _, _ = sem_vq(sem)
            pro_z, _, _ = pro_vq(pro)
            spk_z, _, _ = spk_pq(spk)
            audio_hat = decoder(sem_z, pro_z, spk_z)
            
    # Save Output
    if audio_hat.dim() == 2:
        audio_hat = audio_hat.squeeze(0)
    
    output_filename = os.path.join(args.output_dir, f"recon_{step}_{os.path.basename(args.input)}")
    # Ensure raw output is float32 on cpu
    audio_hat_cpu = audio_hat.float().cpu() # Assuming shape [1, T] or [T]
    if audio_hat_cpu.dim() == 2:
        audio_hat_cpu = audio_hat_cpu.squeeze(0)
    elif audio_hat_cpu.dim() == 0:
        pass # Should not happen
        
    # soundfile expects [Time] or [Time, Channels]
    sf.write(output_filename, audio_hat_cpu.numpy(), 16000)
    print(f"Saved reconstruction to {output_filename}")
    
    # Spectrograms
    spec_filename = os.path.join(args.output_dir, f"spec_{step}_{os.path.basename(args.input)}.png")
    plot_spectrogram(audio, audio_hat_cpu, os.path.basename(args.input), spec_filename)
    print(f"Saved spectrogram to {spec_filename}")

if __name__ == "__main__":
    main()
