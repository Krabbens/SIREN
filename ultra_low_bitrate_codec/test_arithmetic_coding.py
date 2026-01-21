
import torch
import torch.nn.functional as F
import os
import sys
import argparse
import yaml
import time
import math

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from scripts.arithmetic_coding import ArithmeticEncoder, ArithmeticDecoder

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def logits_to_freqs(logits, precision=16):
    """
    Convert logits to integer frequencies for AC.
    """
    probs = F.softmax(logits, dim=-1)
    
    # Scale to integer precision
    total_freq = 1 << precision
    
    # We need to ensure every symbol has at least freq 1, otherwise impossible to encode/decode
    freqs = (probs * (total_freq - 256)).floor().long()
    freqs = freqs + 1 # Ensure min freq 1
    
    # Adjust last to match total exactly
    current_sum = freqs.sum()
    diff = total_freq - current_sum
    freqs[torch.argmax(freqs)] += diff
    
    # Cumulative stats
    cum_freqs = torch.zeros(len(freqs) + 1, dtype=torch.long, device=freqs.device)
    torch.cumsum(freqs, dim=0, out=cum_freqs[1:])
    
    return cum_freqs, freqs, total_freq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="jakubie.wav")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stable")
    parser.add_argument("--step", type=int, default=37000)
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--output", type=str, default="compressed.bin")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    # Load Models
    print("Loading models...")
    factorizer = InformationFactorizerV2(config).to(device)
    sem_vq = ResidualFSQ(config['model']['fsq_levels'], config['model']['rfsq_num_levels'], config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(config['model']['fsq_levels'], config['model']['rfsq_num_levels'], config['model']['prosody']['output_dim']).to(device)
    entropy_model = EntropyModel(config).to(device)
    
    # Helper to load
    def load_part(model, name):
        path = os.path.join(args.checkpoint_dir, f"GOLDEN_step_{args.step}", f"{name}.pt")
        if not os.path.exists(path):
             # Fallback to standard path
             path = os.path.join(args.checkpoint_dir, f"step_{args.step}", f"{name}.pt")
        
        print(f"Loading {name} from {path}...")
        sd = torch.load(path, map_location=device)
        # Fix keys
        new_sd = {k.replace("_orig_mod.", ""): v for k,v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()

    load_part(factorizer, 'factorizer')
    load_part(sem_vq, 'sem_rfsq')
    load_part(pro_vq, 'pro_rfsq')
    load_part(entropy_model, 'entropy')
    
    # Load Audio & Hubert
    import torchaudio
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    
    print("Extracting features...")
    y, sr = torchaudio.load(args.input)
    if sr != 16000:
        print(f"Resampling from {sr} to 16000...")
        y = torchaudio.functional.resample(y, sr, 16000)
    
    y = y.to(device)
    if y.shape[0] > 1: y = y.mean(0, keepdim=True) # Mix to mono (1, T)
    
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    
    inputs = processor(y.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = hubert(inputs.input_values.to(device), output_hidden_states=True)
        feats = outputs.hidden_states[9]
        sem, pro, _ = factorizer(feats)
        _, _, sem_idx = sem_vq(sem)
        _, _, pro_idx = pro_vq(pro)
        
    # Flatten indices (B, T, Levels) -> (B, T*L)?
    # No, EntropyModel takes (B, T) indices. But wait, we trained it on what?
    # In train.py: sem_idx.view(B, -1) -> this flattens T and Levels dimensions into one sequence!
    # Correct. So we treat all levels as a single sequence of tokens.
    
    sem_flat = sem_idx.view(1, -1)
    pro_flat = pro_idx.view(1, -1)
    
    # Convert to Bytes for EntropyModel
    sem_bytes = entropy_model.indices_to_bytes(sem_flat) # (1, N*3)
    pro_bytes = entropy_model.indices_to_bytes(pro_flat)
    
    print(f"Original Indices: Sem {sem_flat.numel()}, Pro {pro_flat.numel()}")
    print(f"Byte Sequence: Sem {sem_bytes.numel()}, Pro {pro_bytes.numel()}")
    
    # ---------------- ENCODING ----------------
    print("\n--- ENCODING ---")
    start_t = time.time()
    
    # 1. Run Transformer ONCE to get all logits (teacher forcing style)
    # We shift inputs right for causality? 
    # EntropyModel forward takes (x) and returns logits for (x).
    # Since it is causal with mask, logits[i] predicts x[i]? 
    # Usually: logits[i] predicts x[i+1]? 
    # Let's check ProbabilisticLM code.
    # forward(x) -> transformer(x) -> logits.
    # mask is causal. h[i] sees x[0]...x[i].
    # So logits[i] is the prediction for the NEXT token x[i+1]?
    # NO. Usually standard causal LM: input x[ :-1] -> logits predict x[1: ]
    # In train.py:
    #   sem_logits, pro_logits = entropy_model(sem_idx[:, :-1], pro_idx[:, :-1])
    #   target = sem_idx[:, 1:]
    # So yes: logits at pos i (given input x_i) predict x_{i+1}.
    
    # We need to prep inputs:
    # Start token? We assume implicit start or we prepend 0?
    # The model was trained on slices.
    
    # For encoding the WHOLE sequence x_0...x_N:
    # We need predictions for x_0, x_1, ... x_N.
    # Pred for x_0 comes from... fixed prior? Or dummy start token?
    # The training code does `model(x[:, :-1])` vs `target x[:, 1:]`.
    # So x_0 is NEVER predicted? It is given as context?
    # This implies we must write x_0 raw (or uniform) and then encode x_1...x_N.
    # Let's Encode x_0 uniformly (8 bits).
    
    # Predict all logits in chunks to avoid OOM / Context Limit
    def get_logits_chunked(model_func, bytes_seq, chunk_size=2048):
        logits_list = []
        L = bytes_seq.size(1)
        
        for i in range(0, L, chunk_size):
            # We treat each chunk as a fresh sequence (resetting context)
            # This is suboptimal but required given the fixed pos embeddings
            chunk = bytes_seq[:, i:i+chunk_size]
            
            # For the last token in chunk i, we don't need prediction for i+chunk_size (it's in next chunk)
            # Actually, causal LM: input x[0..T-1] predicts x[1..T].
            # If we feed chunk[0..T], we get preds for [1..T+1]?
            # Usually we feed chunk[:, :-1] to get preds for chunk[:, 1:].
            
            # To make it simple:
            # We want preds for x[1]...x[L-1].
            # And x[0] is encoded raw/uniform.
            
            # Let's just run forward on the chunk
            # But we need to handle boundary carefully.
            # If we run chunk A, we get logits A_out.
            # A_out[last] predicts A[last+1]? 
            # In current ProbabilisticLM: forward(x) returns logits matching predictions for NEXT?
            # No. forward(x) returns hidden states H. logits = head(H).
            # H[t] depends on x[0]...x[t].
            # So logits[t] is P(x_{t+1} | x_{<=t}).
            
            # So if we feed [x0, x1, x2], we get [l0, l1, l2].
            # l0 predicts x1. l1 predicts x2. l2 predicts x3.
            
            # So if we chunk:
            # Chunk 1: input x[0:2048]. Logits l[0:2048].
            # l[2047] predicts x[2048].
            # If next chunk input starts at x[2048], then its first logit l'[0] predicts x[2049] based on x[2048].
            # Wait, if we reset context, l'[0] sees provided x[2048] as start?
            # If inputs are overlapping?
            
            # Strategy:
            # Run model on valid chunks.
            # Inputs: chunk of size 2048.
            # Outputs: 2048 logits.
            # These logits predict: chunk[1], chunk[2] ... chunk[last+1].
            
            # We want logits for the whole sequence x[0]...x[N-1] (predicting x[1]...x[N]).
            # Actually current encoding loop uses:
            # logit = logits_seq[i] # predicts i+1
            # logits_seq length N-1.
            
            # So we iterate chunks of input.
            # But we must treat them as separate contexts.
            
            # Input: x[i : i+chunk]
            # Output: logits that predict x[i+1 : i+chunk+1].
            
            # The only issue is x[i+chunk] prediction.
            # For the very last token in chunk, it predicts the first token of NEXT chunk.
            # BUT the next chunk inference invalidates this because it restarts context!
            # The next chunk starts efficiently from scratch.
            
            # So:
            # Chunk A: x[0..10]. Logits predict x[1..11].
            # Chunk B: x[11..21]. Logits predict x[12..22].
            # Notice x[11] is predicted by A's last logit.
            # But B starts at x[11]. B's first logit predicts x[12].
            # So we have coverage for x[11] (from A) and x[12] (from B).
            # What about x[11]? It is the start of B.
            # We need to encode x[11] using A's last prediction?
            # YES.
            # But we also need to start B. B's context starts at x[11].
            
            # Actually simplest:
            # Encode x[0] uniform.
            # Loop i=0..N-1:
            #   If prediction for x[i+1] is available from current context, use it.
            #   If we hit context limit, we reset.
            #   Reset means: We treat x[i+1] as a START token of new sequence?
            #   If we treat x[i+1] as start, we encode it uniform (or fixed prior).
            #   Then x[i+2] is predicted from x[i+1].
            
            # So if we chunk inputs:
            # Input: bytes_seq[:, i:i+chunk]
            # Output: logits
            # Use logits to encode bytes_seq[:, i+1 : i+chunk+1]
            
            # This implies the first byte of every chunk (except very first?) might be encoded poorly or needs special handling?
            # No, if we just concatenate logits, we are lying about the history.
            # If we concat logits [L1, L2], L2[0] assumes history L1. But real L2[0] had NO history (reset).
            # So we used a "fresh" model to generate L2.
            # That matches the decoder behavior! 
            # Decoder will also chunk/reset.
            # So as long as we reset context at fixed intervals, it's consistent.
            
            with torch.no_grad():
                out_chunk = model_func(chunk)
                logits_list.append(out_chunk)
                
        return torch.cat(logits_list, dim=1)

    with torch.no_grad():
        print(f"Debug Sem Idx: Min {sem_idx.min()}, Max {sem_idx.max()}, Mean {sem_idx.float().mean()}")
        sem_logits = get_logits_chunked(entropy_model.sem_lm, sem_bytes[:, :-1])
        pro_logits = get_logits_chunked(entropy_model.pro_lm, pro_bytes[:, :-1])

    # Debug Logits
    probs = F.softmax(sem_logits[0, :10], dim=-1)
    ent = -(probs * torch.log2(probs + 1e-9)).sum(-1)
    print(f"Debug Logits (First 10): Entropy {ent}")
    print(f"Debug Logits Stats: Max {sem_logits.max()}, Min {sem_logits.min()}, Std {sem_logits.std()}")
        
    # Convert to freqs
    # sem_logits: (1, Seq-1, 256)
    
    encoder = ArithmeticEncoder(32)
    
    # Helper to encode stream
    def encode_stream(bytes_seq, logits_seq, name):
        # bytes_seq: (Seq)
        # logits_seq: (Seq-1, 256)
        
        # Write first byte raw (or assume 0?)
        # Let's simple write first byte using uniform dist (freq=1 for all)
        first = bytes_seq[0].item()
        # Uniform: cum_freq = x, freq=1, total=256
        encoder.encode(first, 1, 256)
        
        # Encode rest
        for i in range(len(bytes_seq) - 1):
            target = bytes_seq[i+1].item()
            logit = logits_seq[i] # predicts i+1
            
            cum_freqs, freqs, total_freq = logits_to_freqs(logit)
            
            c = cum_freqs[target].item()
            f = freqs[target].item()
            t = total_freq # int
            
            encoder.encode(c, f, t)
            
    print("Encoding Semantic Stream...")
    encode_stream(sem_bytes.squeeze(), sem_logits.squeeze(), "sem")
    print("Encoding Prosody Stream...")
    encode_stream(pro_bytes.squeeze(), pro_logits.squeeze(), "pro")
    
    bitstream = encoder.finish()
    
    enc_time = time.time() - start_t
    print(f"Encoding done in {enc_time:.2f}s")
    
    # Stats
    total_bytes = len(bitstream)
    duration = y.shape[-1] / 16000
    bps = (total_bytes * 8) / duration
    
    print(f"\nCOMPRESSED SIZE: {total_bytes} bytes")
    print(f"Audio Duration:  {duration:.2f}s")
    print(f"ACTUAL BITRATE:  {bps:.2f} bps")
    
    with open(args.output, 'wb') as f:
        f.write(bitstream)
    print(f"Saved to {args.output}")

    # ---------------- L1 ANALYSIS ----------------
    # To estimate L1 bitrate, we need to know WHICH bytes correspond to L1.
    # Structure: (B, T, Levels).
    # sem_idx was flattened via .view(1, -1).
    # Original shape was likely (1, T, 8)?
    # Let's verify config['model']['fsq_levels'].
    # Config has [8, 8, 8, 8, 8, 8, 8, 8] -> 8 levels.
    # So index sequence is: L1_0, L2_0 ... L8_0, L1_1, ...
    # Wait, ResidualFSQ output `indices` shape is `(B, T, n_levels)`.
    # `view(B, -1)` flattens to `t0_l0, t0_l1, ...`.
    # So every 8th index is Level 1.
    # Index 0, 8, 16... are Level 1.
    
    # However, indices are converted to BYTES.
    # `indices_to_bytes` assumes indices < 2^24.
    # It produces 3 bytes per index.
    # So each index becomes 3 bytes.
    # Sequence of bytes: [L1_B1, L1_B2, L1_B3, L2_B1...]
    # So every 24th byte corresponds to the start of L1?
    # No. 
    # Index 0 (L1) -> Bytes 0,1,2.
    # Index 1 (L2) -> Bytes 3,4,5.
    # ...
    # Index 7 (L8) -> Bytes 21,22,23.
    # Index 8 (L1 next frame) -> Bytes 24,25,26.
    
    # So L1 bytes are at indices [0,1,2] + k*24.
    
    # We need to simulate the cost of these bytes.
    # We can do this by re-running encoding or tracking it during encoding.
    # But `encoder.finish()` produces a blob. We don't know which bits belong to which symbol easily.
    # Arithmetic Coding blends bits.
    # BUT theoretical bitrate is `sum(-log2(p_target))`.
    # We have logits! We can calculate exact theoretical cross-entropy for L1.
    
    def calc_entropy_for_mask(logits, targets, mask_indices):
        # logits: (N, 256)
        # targets: (N)
        # mask_indices: list of indices to include
        
        relevant_logits = logits[mask_indices]
        relevant_targets = targets[mask_indices]
        
        log_probs = F.log_softmax(relevant_logits, dim=-1)
        # Gather target log probs
        target_log_probs = log_probs.gather(1, relevant_targets.unsqueeze(1)).squeeze(1)
        
        total_bits = -target_log_probs.sum().item() / math.log(2)
        return total_bits
    
    num_levels = len(config['model']['fsq_levels'])
    bytes_per_idx = 3
    stride = num_levels * bytes_per_idx # 8 * 3 = 24
    
    # Semantic L1
    seq_len_sem = sem_bytes.size(1)
    indices_sem = torch.arange(seq_len_sem - 1, device=sem_logits.device)
    target_indices_sem = indices_sem + 1
    l1_mask_sem = ((target_indices_sem % stride) < 3)
    l1_selection_sem = indices_sem[l1_mask_sem]
    
    sem_l1_bits = calc_entropy_for_mask(sem_logits.squeeze(0), sem_bytes.squeeze(0)[1:], l1_selection_sem)
    
    # Prosody L1
    seq_len_pro = pro_bytes.size(1)
    indices_pro = torch.arange(seq_len_pro - 1, device=pro_logits.device)
    target_indices_pro = indices_pro + 1
    l1_mask_pro = ((target_indices_pro % stride) < 3)
    l1_selection_pro = indices_pro[l1_mask_pro]
    
    pro_l1_bits = calc_entropy_for_mask(pro_logits.squeeze(0), pro_bytes.squeeze(0)[1:], l1_selection_pro)
    
    total_l1_bits = sem_l1_bits + pro_l1_bits
    l1_bps = total_l1_bits / duration
    
    print(f"\n--- L1 BITRATE ESTIMATION ---")
    print(f"L1 Sem Bits: {sem_l1_bits:.2f}")
    print(f"L1 Pro Bits: {pro_l1_bits:.2f}")
    print(f"L1 Total BPS: {l1_bps:.2f} bps")

if __name__ == "__main__":
    main()
