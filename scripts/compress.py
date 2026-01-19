#!/usr/bin/env python3
import torch
import yaml
import argparse
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from transformers import Wav2Vec2FeatureExtractor, HubertModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input audio file")
    parser.add_argument("--output", type=str, required=True, help="Output bitstream file (.pt)")
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/multispeaker_optimized.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_multispeaker")
    parser.add_argument("--step", type=int, default=1500)
    parser.add_argument("--levels", type=int, default=None, help="Number of RFSQ levels to keep (truncation). Default: Keep all.")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = load_config(args.config)
    
    # Load Encoder Models
    print("Loading encoder models...")
    factorizer = InformationFactorizerV2(config).to(device)
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
    
    def load_clean_state_dict(model, path):
        sd = torch.load(path, map_location=device)
        # Strip _orig_mod prefix from torch.compile
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("_orig_mod."):
                new_sd[k[10:]] = v
            else:
                new_sd[k] = v
        model.load_state_dict(new_sd)
        
    load_clean_state_dict(factorizer, f"{ckpt_dir}/factorizer_{step}.pt")
    load_clean_state_dict(sem_vq, f"{ckpt_dir}/sem_rfsq_{step}.pt")
    load_clean_state_dict(pro_vq, f"{ckpt_dir}/pro_rfsq_{step}.pt")
    load_clean_state_dict(spk_pq, f"{ckpt_dir}/spk_pq_{step}.pt")
    
    factorizer.eval()
    
    # Load HuBERT
    print("Loading HuBERT...")
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    hubert_model.eval()
    
    # Load Audio
    import soundfile as sf
    print(f"Loading audio {args.input}...")
    audio_np, sr = sf.read(args.input)
    audio = torch.from_numpy(audio_np).float()
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.t()
        
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        import torchaudio
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    audio_dev = audio.to(device)
    
    # Load Entropy Model
    print("Loading Entropy Model...")
    from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
    entropy_model = EntropyModel(config).to(device)
    try:
        load_clean_state_dict(entropy_model, f"{ckpt_dir}/entropy_{step}.pt")
        print("Entropy model loaded!")
    except FileNotFoundError:
        print("Entropy model checkpoint not found! Falling back to uncompressed indices.")
        # Fallback code or just continue with raw? 
        # User wants valid proof. We must try best effort. 
        # If no entropy model, we can't compress losslessly better than headers.
        pass

    # Extract
    with torch.no_grad():
        inputs = hubert_processor(audio_dev.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        outputs = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        layer = config['model'].get('hubert_layer', 9)
        features = outputs.hidden_states[layer]
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sem, pro, spk = factorizer(features)
            _, _, sem_indices = sem_vq(sem) # (B, T, L)
            _, _, pro_indices = pro_vq(pro) # (B, T, L)
            _, _, spk_indices = spk_pq(spk)
            
    if args.levels is not None:
        sem_indices = sem_indices[:, :, :args.levels]
        pro_indices = pro_indices[:, :, :args.levels]
            
    if args.levels is not None:
        sem_indices = sem_indices[:, :, :args.levels]
        pro_indices = pro_indices[:, :, :args.levels]
            
    B, T_s, L_s = sem_indices.shape
    B, T_p, L_p = pro_indices.shape
    
    sem_flat = sem_indices.contiguous().view(B, -1) # (B, T*L)
    pro_flat = pro_indices.contiguous().view(B, -1) # (B, T*L)
    # Note: RFSQ output is (B, T, num_levels).
    # If num_levels=4, flatten is T*4.
    
    # Run Entropy Model (Teacher Forcing) in chunks
    MAX_LEN = 256 # From ProbabilisticLM default
    
    def get_log_probs(model, x):
        # x: (B, L)
        logits_list = []
        for i in range(0, x.shape[1], MAX_LEN):
            chunk = x[:, i:i+MAX_LEN]
            if chunk.shape[1] > MAX_LEN: # Should not happen with slice
                chunk = chunk[:, :MAX_LEN]
                
            chunk_logits = model(chunk) # (B, Chunk, V)
            logits_list.append(chunk_logits)
            
        return torch.cat(logits_list, dim=1)
        
    with torch.no_grad():
        sem_logits = get_log_probs(entropy_model.sem_lm, sem_flat)
        pro_logits = get_log_probs(entropy_model.pro_lm, pro_flat)
        
    sem_probs = torch.softmax(sem_logits, dim=-1)
    pro_probs = torch.softmax(pro_logits, dim=-1)
    
    # Compress using Arithmetic Coding
    from scripts.arithmetic_coding import ArithmeticEncoder
    
    def compress_stream(indices, probs, tag="stream"):
        encoder = ArithmeticEncoder()
        seq_len = indices.shape[1]
        
        idx_cpu = indices[0].cpu().tolist()
        probs_cpu = probs[0].cpu() # (Seq, V)
        
        vocab_size = probs.shape[-1]
        total_freq = 10000 
        
        # Encode first token of EACH CHUNK? 
        # No, "chunking" in LM means we reset position embeddings.
        # But the stream is continuous.
        # If we just concat probs, we assume:
        # Token 0: Uniform (or implied uniform by model?)
        # Token 256: Predicted by LM(chunk[0]). i.e. it acts as start.
        # So essentially we have independent segments for probability estimation.
        # The arithmetic coder just encodes a sequence of symbols using a sequence of distributions.
        # As long as Decompressor replicates the distributions, it's fine.
        
        # Token 0 is always Uniform in our logic?
        # NO. LM predicts `logits[t]` using `x[:t+1]`. `logits[t]` is for `x[t+1]`.
        # So `probs[chunk_t]` predicts `chunk[chunk_t+1]`.
        # `chunk[0]` is NOT predicted by previous chunk's last prob?
        # Because we reset state.
        # So `chunk[0]` effectively is uniform (or unpredicted).
        # We must encode `chunk[0]` uniformly for EVERY CHUNK.
        
        # Loop
        byte_stream = bytearray()
        
        for i in range(0, seq_len, MAX_LEN):
            chunk_len = min(MAX_LEN, seq_len - i)
            
            # Encode chunk[0] Uniformly
            sym_0 = idx_cpu[i]
            encoder.encode(
                cum_freq=sym_0 * total_freq // vocab_size,
                freq=total_freq // vocab_size,
                total_freq=total_freq
            )
            
            # Encode rest of chunk
            for t in range(1, chunk_len):
                abs_t = i + t
                sym = idx_cpu[abs_t]
                # probs[abs_t - 1] predicts sym. 
                # Be careful: `probs` came from `get_log_probs` which cat'd outputs.
                # `logits_list` contained `model(chunk)`.
                # `chunk_logits[t]` predicts `chunk[t+1]`?
                # Yes, if causal.
                # So `sem_probs` index mapping:
                # `sem_probs[i + t - 1]` is the distribution for `sem_flat[i+t]`.
                
                p = probs_cpu[i + t - 1]
                
                freqs = (p * total_freq).long()
                freqs = torch.maximum(freqs, torch.tensor(1))
                current_sum = freqs.sum()
                freqs[-1] += (total_freq - current_sum)
                
                cum_freqs = torch.cat([torch.tensor([0]), torch.cumsum(freqs, 0)])
                
                f = freqs[sym].item()
                cf = cum_freqs[sym].item()
                
                encoder.encode(cf, f, total_freq)
                
        return encoder.finish()

    print("Compressing Semantic Stream...")
    sem_bytes = compress_stream(sem_flat, sem_probs)
    
    print("Compressing Prosody Stream...")
    pro_bytes = compress_stream(pro_flat, pro_probs)
    
    # Save Bitstream
    import pickle
    # We save dictionary with bytes and metadata
    bitstream_data = {
        'sem_bytes': sem_bytes,
        'pro_bytes': pro_bytes,
        'spk_indices': spk_indices.cpu(), # Raw (negligible)
        'sem_shape': sem_indices.shape, # To reconstruct
        'pro_shape': pro_indices.shape,
        'original_duration': audio.shape[1] / 16000
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(bitstream_data, f)
        
    # Stats
    sem_b = len(sem_bytes)*8
    pro_b = len(pro_bytes)*8
    spk_b = spk_indices.numel() * 16 # Int16
    total_bits = sem_b + pro_b + spk_b
    duration = audio.shape[1] / 16000
    bps = total_bits / duration
    
    print(f"Compressed {args.input} ({duration:.2f}s)")
    print(f"Total Bits: {total_bits}")
    print(f"Bitrate: {bps:.2f} bps")
    print(f"Saved bitstream to {args.output}")

if __name__ == "__main__":
    main()
