
import torch
import os
import sys
import argparse
import yaml
import time
import math
import numpy as np
import torchaudio

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
# scripts is in root
from scripts.arithmetic_coding import ArithmeticEncoder, ArithmeticDecoder

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="jakubie.wav")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stable")
    parser.add_argument("--step", type=int, default=87000)
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--output", type=str, default="l1_ac_experiment.bin")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    print("Loading models...")
    factorizer = InformationFactorizerV2(config).to(device)
    sem_vq = ResidualFSQ(config['model']['fsq_levels'], config['model']['rfsq_num_levels'], config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(config['model']['fsq_levels'], config['model']['rfsq_num_levels'], config['model']['prosody']['output_dim']).to(device)
    
    # Calculate vocab size for Entropy Model
    fsq_levels = config['model']['fsq_levels'] 
    
    # EntropyModel takes config dict
    entropy_model = EntropyModel(config).to(device)
    
    def load_part(model, name):
        path = os.path.join(args.checkpoint_dir, f"step_{args.step}", f"{name}.pt")
        if not os.path.exists(path):
            path = os.path.join(args.checkpoint_dir, f"GOLDEN_step_{args.step}", f"{name}.pt")
        print(f"Loading {name} from {path}...")
        sd = torch.load(path, map_location=device)
        new_sd = {k.replace("_orig_mod.", ""): v for k,v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()

    load_part(factorizer, 'factorizer')
    load_part(sem_vq, 'sem_rfsq')
    load_part(pro_vq, 'pro_rfsq')
    load_part(entropy_model, 'entropy')
    
    print("Processing audio...")
    y, sr = torchaudio.load(args.input)
    if sr != 16000: y = torchaudio.functional.resample(y, sr, 16000)
    y = y.to(device)
    if y.shape[0] > 1: y = y.mean(0, keepdim=True)
    
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    
    inputs = processor(y.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = hubert(inputs.input_values.to(device), output_hidden_states=True)
        feats = outputs.hidden_states[9]
        sem, pro, _ = factorizer(feats)
        _, _, sem_idx = sem_vq(sem) 
        _, _, pro_idx = pro_vq(pro)
        
    sem_idx = sem_idx.squeeze() # (T, 8)
    pro_idx = pro_idx.squeeze() # (T, 8)
    
    def indices_to_bytes(indices):
        flat = indices.flatten()
        b1 = (flat >> 16) & 0xFF
        b2 = (flat >> 8) & 0xFF
        b3 = flat & 0xFF
        bytes_seq = torch.stack([b1, b2, b3], dim=1).flatten()
        return bytes_seq.long()

    sem_bytes = indices_to_bytes(sem_idx).unsqueeze(0) # (1, Seq)
    pro_bytes = indices_to_bytes(pro_idx).unsqueeze(0)
    
    stride = 24
    
    def encode_stream_l1_only(bytes_seq, output_file, model_type='sem'):
        encoder = ArithmeticEncoder(32)
        seq_len = bytes_seq.size(1)
        
        # Select sub-model
        if model_type == 'sem':
            lm = entropy_model.sem_lm
        else:
            lm = entropy_model.pro_lm
            
        chunk_size = 512
        cx_len = 2048
        
        masked_seq = bytes_seq.clone()
        indices = torch.arange(seq_len, device=device)
        mask_keep = (indices % stride) < 3
        
        masked_seq[0, ~mask_keep] = 0 # Force residuals to 0
        
        all_logits = []
        
        with torch.no_grad():
            for i in range(0, seq_len - 1, chunk_size):
                end = min(i + chunk_size, seq_len - 1)
                start_ctx = max(0, i - cx_len)
                ctx = masked_seq[:, start_ctx:end]
                
                logits = lm(ctx)
                relevant_logits = logits[:, (i - start_ctx):]
                all_logits.append(relevant_logits)

                
        all_logits = torch.cat(all_logits, dim=1).squeeze(0) # (T-1, 256)
        
        # Now encode ONLY the L1 bytes.
        # Target indices: 1..N-1.
        targets = masked_seq.squeeze(0)[1:]
        target_indices = torch.arange(len(targets), device=device) + 1
        
        l1_targets_mask = (target_indices % stride) < 3
        
        l1_logits = all_logits[l1_targets_mask]
        l1_symbols = targets[l1_targets_mask]
        
        print(f"Encoding {len(l1_symbols)} L1 bytes...")
        
        # Encode
        probs = torch.softmax(l1_logits, dim=-1)
        
        # Frequencies
        # Convert probs to CDF
        # ArithEncoder expects freq table? or CDF?
        # My implementation: `encode(sym, freq_table)`?
        # Let's check usage.
        
        # For speed, we might want to quantize CDF.
        # But let's look at `arithmetic_coding.py` interface.
        
        cdf = torch.cumsum(probs, dim=-1)
        # Ensure last is exactly 1.0 (though float precision handles it mostly)
        
        # Move to CPU
        cdf_cpu = cdf.cpu().numpy()
        sym_cpu = l1_symbols.cpu().numpy()
        
        # Scale for integer arithmetic
        SCALE = 1_000_000
        
        for k in range(len(sym_cpu)):
            sym = sym_cpu[k]
            
            # Get probs for this step
            # cdf_k is array of shape (256,)
            cdf_k = cdf_cpu[k]
            
            # Calculate integer freqs
            # Low_cum = cdf[sym-1] * SCALE
            # High_cum = cdf[sym] * SCALE
            
            high_cum = int(cdf_k[sym] * SCALE)
            if sym == 0:
                low_cum = 0
            else:
                low_cum = int(cdf_k[sym-1] * SCALE)
                
            freq = high_cum - low_cum
            
            # Avoiding 0 freq due to precision (if prob very small)
            if freq == 0:
                freq = 1
                high_cum = low_cum + 1
            
            # Encode
            encoder.encode(low_cum, freq, SCALE)
            
        bytes_out = encoder.finish()
        
        with open(output_file, 'wb') as f:
            f.write(bytes_out)
        
        return len(bytes_out)

    print("\n--- EXPERIMENTAL L1 AC ---")
    size_sem = encode_stream_l1_only(sem_bytes, "l1_ac_sem.bin", model_type='sem')
    size_pro = encode_stream_l1_only(pro_bytes, "l1_ac_pro.bin", model_type='pro')
    
    total_ac_size = size_sem + size_pro
    duration = y.shape[-1] / 16000
    
    print(f"Sem AC Size: {size_sem} bytes")
    print(f"Pro AC Size: {size_pro} bytes")
    print(f"Total AC Size: {total_ac_size} bytes")
    print(f"AC Bitrate: {(total_ac_size*8)/duration:.2f} bps")

    
if __name__ == "__main__":
    main()
