#!/usr/bin/env python3
import torch
import yaml
import argparse
import os
import sys
import soundfile as sf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input bitstream file (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Output wav file")
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/multispeaker_optimized.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/checkpoints_multispeaker")
    parser.add_argument("--step", type=int, default=1500)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = load_config(args.config)
    
    # Load Decoder Models
    print("Loading decoder models...")
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
    decoder.load_state_dict(torch.load(f"{ckpt_dir}/decoder_{step}.pt", map_location=device))
    sem_vq.load_state_dict(torch.load(f"{ckpt_dir}/sem_rfsq_{step}.pt", map_location=device))
    pro_vq.load_state_dict(torch.load(f"{ckpt_dir}/pro_rfsq_{step}.pt", map_location=device))
    spk_pq.load_state_dict(torch.load(f"{ckpt_dir}/spk_pq_{step}.pt", map_location=device))
    
    decoder.eval()
    
    # Load Entropy Model
    print("Loading Entropy Model...")
    from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
    entropy_model = EntropyModel(config).to(device)
    entropy_model.load_state_dict(torch.load(f"{ckpt_dir}/entropy_{step}.pt", map_location=device))
    
    # Load Bitstream
    import pickle
    print(f"Loading bitstream {args.input}...")
    with open(args.input, 'rb') as f:
        bitstream = pickle.load(f)
        
    sem_shape = bitstream['sem_shape'] # (B, T, 4)
    pro_shape = bitstream['pro_shape'] # (B, T, 4)
    spk_indices = bitstream['spk_indices'].to(device).long()
    
    # Flatten shapes for decoding
    sem_len = sem_shape[1] * sem_shape[2]
    pro_len = pro_shape[1] * pro_shape[2]
    
    from scripts.arithmetic_coding import ArithmeticDecoder
    
    MAX_LEN = 256
    
    def decode_stream(byte_stream, length, lm_func):
        decoder = ArithmeticDecoder(byte_stream)
        decoded_indices = []
        
        full_indices = []
        
        total_freq = 10000
        vocab_size = config['model']['semantic']['vocab_size']
        freq_per_sym = total_freq // vocab_size
        uniform_cum_freqs = [i * freq_per_sym for i in range(vocab_size + 1)]
        uniform_cum_freqs[-1] = total_freq
        uniform_freqs = [freq_per_sym] * vocab_size
        uniform_freqs[-1] += (total_freq - sum(uniform_freqs))
        
        for i in range(0, length, MAX_LEN):
            chunk_len = min(MAX_LEN, length - i)
            chunk_indices = []
            
            # Decode first token of chunk (Uniform)
            sym = decoder.decode(uniform_cum_freqs, uniform_freqs, total_freq)
            chunk_indices.append(sym)
            full_indices.append(sym)
            
            # Autoregressive loop for rest of chunk
            for t in range(1, chunk_len):
                # Context is current chunk so far
                ctx = torch.tensor([chunk_indices], device=device).long()
                
                with torch.no_grad():
                    logits = lm_func(ctx) # (1, t, V)
                    # Last logit predicts next token
                    p = torch.softmax(logits[:, -1, :], dim=-1).squeeze()
                
                freqs = (p * total_freq).long().cpu()
                freqs = torch.maximum(freqs, torch.tensor(1))
                current_sum = freqs.sum()
                freqs[-1] += (total_freq - current_sum)
                
                cum_freqs = torch.cat([torch.tensor([0]), torch.cumsum(freqs, 0)]).numpy().tolist()
                freqs = freqs.numpy().tolist()
                
                sym = decoder.decode(cum_freqs, freqs, total_freq)
                chunk_indices.append(sym)
                full_indices.append(sym)
                
        return torch.tensor(full_indices, device=device).long()

    print("Decoding Semantic Stream...")
    sem_flat = decode_stream(bitstream['sem_bytes'], sem_len, lambda x: entropy_model.sem_lm(x))
    
    print("Decoding Prosody Stream...")
    # vocab for prosody might differ? RFSQ levels same?
    # Usually same FSQ levels -> same vocab.
    # But check config.
    pro_flat = decode_stream(bitstream['pro_bytes'], pro_len, lambda x: entropy_model.pro_lm(x))
    
    sem_indices = sem_flat.view(*sem_shape)
    pro_indices = pro_flat.view(*pro_shape)
    
    # Reconstruct
    with torch.no_grad():
        sem_z = sem_vq.from_indices(sem_indices)
        pro_z = pro_vq.from_indices(pro_indices)
        spk_z = spk_pq.from_indices(spk_indices)
        if spk_z.dim() == 3 and spk_z.shape[1] == 1:
            spk_z = spk_z.squeeze(1)
        
        audio_hat = decoder(sem_z, pro_z, spk_z)
    
    # Save Audio
    audio_hat = audio_hat.cpu().squeeze()
    if audio_hat.dim() == 2:
        audio_hat = audio_hat.squeeze(0)
    
    sf.write(args.output, audio_hat.numpy(), 16000)
    print(f"Decompressed to {args.output}")

if __name__ == "__main__":
    main()
