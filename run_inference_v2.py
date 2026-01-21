#!/usr/bin/env python3
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
import os
import sys
import math
from transformers import Wav2Vec2Processor, HubertModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_spectrogram(y, y_hat, title, save_path):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Original
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)(y.cpu())
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec).squeeze().numpy()
    im1 = axs[0].imshow(spec_db, aspect='auto', origin='lower', vmin=-80, vmax=20)
    axs[0].set_title(f"Original: {title}")
    plt.colorbar(im1, ax=axs[0], format='%+2.0f dB')
    
    # Reconstructed
    spec_hat = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)(y_hat.cpu())
    spec_hat_db = torchaudio.transforms.AmplitudeToDB()(spec_hat).squeeze().numpy()
    im2 = axs[1].imshow(spec_hat_db, aspect='auto', origin='lower', vmin=-80, vmax=20)
    axs[1].set_title("Reconstructed")
    plt.colorbar(im2, ax=axs[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="jakubie.wav", help="Input audio file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_multispeaker")
    parser.add_argument("--step", type=int, default=57000)
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
    
    entropy_model = EntropyModel(config).to(device)

    # Load Weights
    ckpt_dir = args.checkpoint_dir
    step = args.step
    def load_clean_state_dict(model, path):
        state_dict = torch.load(path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    try:
        # Dictionary mapping model objects to their expected filenames
        models_to_load = {
            'factorizer': (factorizer, 'factorizer'),
            'decoder': (decoder, 'decoder'),
            'sem_vq': (sem_vq, 'sem_rfsq'),
            'pro_vq': (pro_vq, 'pro_rfsq'),
            'spk_pq': (spk_pq, 'spk_pq'),
            'entropy': (entropy_model, 'entropy')
        }

        for name, (model, filename_base) in models_to_load.items():
            # Try flat structure first: {ckpt_dir}/{filename_base}_{step}.pt
            flat_path = os.path.join(ckpt_dir, f"{filename_base}_{step}.pt")
            # Try nested structure: {ckpt_dir}/step_{step}/{filename_base}.pt
            nested_path = os.path.join(ckpt_dir, f"step_{step}", f"{filename_base}.pt")

            if os.path.exists(flat_path):
                load_path = flat_path
            elif os.path.exists(nested_path):
                load_path = nested_path
            else:
                # Don't fail if entropy model missing (legacy checkpoints)
                if name == 'entropy':
                    print(f"Warning: Entropy model checkpoint not found. Entropy metrics will be histograms only.")
                    continue
                raise FileNotFoundError(f"Could not find checkpoint for {name}. Checked:\n1. {flat_path}\n2. {nested_path}")
            
            print(f"Loading {name} from {load_path}")
            load_clean_state_dict(model, load_path)

        print("Model weights loaded!")
    except FileNotFoundError as e:
        print(f"Error loading checkpoints: {e}")
        return

    factorizer.eval()
    decoder.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_pq.eval()
    entropy_model.eval()
    
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
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.t()
        
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    duration = audio.shape[1] / 16000
    print(f"Audio duration: {duration:.2f}s")
    
    audio_dev = audio.to(device)
    
    # Extract Features
    with torch.no_grad():
        inputs = hubert_processor(audio_dev.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        outputs = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        layer = config['model'].get('hubert_layer', 9)
        features = outputs.hidden_states[layer] 
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        # Factorize
        sem, pro, spk = factorizer(features)
        
        # Quantize and get indices
        sem_z, _, sem_indices = sem_vq(sem)
        pro_z, _, pro_indices = pro_vq(pro)
        spk_z, _, spk_indices = spk_pq(spk)
        
        # Decode
        audio_hat = decoder(sem_z, pro_z, spk_z)

    # --- Analysis & Multi-level Reconstruction ---
    
    def calculate_entropy(indices, vocab_size):
        """Calculate Shannon entropy for indices."""
        # indices: [B, T, Levels] or [B, T]
        # Flatten
        flat = indices.cpu().flatten().float()
        counts = torch.histc(flat, bins=vocab_size, min=0, max=vocab_size-1)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        return entropy

    # Vocab sizes
    # Calculate dynamically from config
    fsq_levels = config['model']['fsq_levels']
    fsq_vocab = 1
    for l in fsq_levels:
        fsq_vocab *= l
    
    # Speaker has 8 groups of 256 -> 256
    spk_vocab = 256

    # Calculate entropies using the trained EntropyModel (True compression rate)
    print(f"\n{'='*20} ENTROPY ANALYSIS {'='*20}")
    
    # Defaults (if model fails)
    sem_bits_per_token = math.log2(fsq_vocab)
    pro_bits_per_token = math.log2(fsq_vocab)
    
    with torch.no_grad():
        sem_idx_flat = sem_indices.view(sem_indices.shape[0], -1)
        pro_idx_flat = pro_indices.view(pro_indices.shape[0], -1)
        
        try:
             # Returns total bits for the sequence
             sem_bits_total, pro_bits_total = entropy_model.estimate_bits(sem_idx_flat, pro_idx_flat)
             
             # Convert to avg bits per token
             sem_bits_per_token = sem_bits_total.sum().item() / sem_idx_flat.numel()
             pro_bits_per_token = pro_bits_total.sum().item() / pro_idx_flat.numel()
             
             print(f"Semantic Model Entropy: {sem_bits_per_token:.2f} bits/code")
             print(f"Prosody Model Entropy:  {pro_bits_per_token:.2f} bits/code")
             
        except Exception as e:
             print(f"Entropy Model Eval Failed: {e}")
    
    spk_ent_per_code = calculate_entropy(spk_indices, spk_vocab)
    print(f"Speaker Entropy (Hist):  {spk_ent_per_code:.2f} bits/code (Theoretical vs {math.log2(spk_vocab):.2f} Raw)")

    print(f"{'='*58}\n")

    # Levels to test
    max_levels = config['model']['rfsq_num_levels']
    
    results = []

    for l in range(1, max_levels + 1):
        print(f"--- Reconstructing with {l} levels ---")
        
        # Reconstruct with limited levels
        # We need to manually call decoder or modify it? 
        # The decoder takes z_q. We need to reconstruct z_q from partial indices.
        
        # Helper to reconstruct z from indices up to level l
        def get_partial_z(vq_module, indices, num_levels):
            # indices: [B, T, L_total]
            # Take first num_levels
            partial_indices = indices[..., :num_levels]
            return vq_module.from_indices(partial_indices)
            
        sem_z_partial = get_partial_z(sem_vq, sem_indices, l)
        pro_z_partial = get_partial_z(pro_vq, pro_indices, l)
        
        # DEBUG: Check if indices are diverse
        if l > 1:
            curr_sem_indices = sem_indices[..., l-1]
            curr_pro_indices = pro_indices[..., l-1]
            print(f"  Level {l} Unique Indices - Sem: {len(torch.unique(curr_sem_indices))} / Pro: {len(torch.unique(curr_pro_indices))}")
        
        # Speaker is always full resolution for now (it's small anyway)
        # spk_z is already computed
        
        with torch.no_grad():
            audio_hat_l = decoder(sem_z_partial, pro_z_partial, spk_z)
            
        if audio_hat_l.dim() == 2:
            audio_hat_l = audio_hat_l.squeeze(0)
        
        # Calculate Bitrates
        # 1. Raw Bitrate (just counting bits used by levels)
        # 2. Entropy Bitrate (using calculated entropy * count)
        
        # Num codes
        n_sem_codes = sem_indices.shape[1] * l
        n_pro_codes = pro_indices.shape[1] * l
        if spk_indices.dim() == 2:
            n_spk_codes = spk_indices.shape[1]
        else:
            n_spk_codes = spk_indices.shape[1] * spk_indices.shape[2]
        
        # Raw Bits
        bits_sem_raw = n_sem_codes * math.log2(fsq_vocab)
        bits_pro_raw = n_pro_codes * math.log2(fsq_vocab)
        bits_spk_raw = n_spk_codes * math.log2(spk_vocab)
        total_bits_raw = bits_sem_raw + bits_pro_raw + bits_spk_raw
        bps_raw = total_bits_raw / duration
        
        # Entropy Bits (Assuming we can achieve the calculated entropy per code)
        # Note: Entropy might change slightly if we only consider the first L levels, 
        # Entropy Bits (Using Model Estimation)
        # We use the average bits per token derived from the full sequence.
        # This is an approximation for partial levels (assuming equal distribution across levels)
        # but much more accurate than the broken histogram.
        
        bits_sem_ent = n_sem_codes * sem_bits_per_token
        bits_pro_ent = n_pro_codes * pro_bits_per_token
        bits_spk_ent = n_spk_codes * spk_ent_per_code 
        
        total_bits_ent = bits_sem_ent + bits_pro_ent + bits_spk_ent
        bps_ent = total_bits_ent / duration
        
        print(f"Level {l}: Raw {bps_raw:.0f} bps | Entropy {bps_ent:.0f} bps")
        
        # Save Audio
        out_name = os.path.join(args.output_dir, f"jakubie_{step}_L{l}.wav")
        audio_hat_cpu = audio_hat_l.float().cpu()
        if audio_hat_cpu.dim() == 2: audio_hat_cpu = audio_hat_cpu.squeeze(0)
        sf.write(out_name, audio_hat_cpu.numpy(), 16000)
        
        # Save Spectrogram
        spec_name = os.path.join(args.output_dir, f"jakubie_{step}_L{l}_spec.png")
        plot_spectrogram(audio, audio_hat_cpu, f"jakubie (L{l})", spec_name)
        
        results.append({
            'level': l,
            'bps_raw': bps_raw,
            'bps_ent': bps_ent,
            'wav': out_name,
            'spec': spec_name
        })

    # Summary
    print("\n" + "="*40)
    print("FINAL RESULTS SUMMARY")
    print(f"{'Lvl':<4} | {'Raw BPS':<10} | {'Entropy BPS':<12} | {'Ratio':<6}")
    print("-" * 40)
    for r in results:
        print(f"{r['level']:<4} | {r['bps_raw']:<10.1f} | {r['bps_ent']:<12.1f} | {r['bps_raw']/r['bps_ent']:.2f}x")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
