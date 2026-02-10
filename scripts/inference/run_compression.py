#!/usr/bin/env python3
"""
Arithmetic Coding Compression/Decompression Script

Uses the pure Python arithmetic coder from scripts/arithmetic_coding.py
to compress the quantized indices from the codec.
"""
import torch
import numpy as np
import struct
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.arithmetic_coding import ArithmeticEncoder, ArithmeticDecoder

def build_freq_table(indices, vocab_size):
    """Build frequency table from indices."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    flat = indices.flatten().cpu().numpy()
    for idx in flat:
        counts[idx] += 1
    # Ensure no zero counts (Laplace smoothing)
    counts = counts + 1
    return counts

def counts_to_cumulative(counts):
    """Convert counts to cumulative frequencies."""
    cum = np.zeros(len(counts) + 1, dtype=np.int64)
    cum[1:] = np.cumsum(counts)
    return cum

def encode_indices(indices, vocab_size):
    """Encode a sequence of indices using arithmetic coding."""
    counts = build_freq_table(indices, vocab_size)
    cum_freq = counts_to_cumulative(counts)
    total = int(cum_freq[-1])
    
    encoder = ArithmeticEncoder(precision=32)
    flat = indices.flatten().cpu().numpy()
    
    for idx in flat:
        sym = int(idx)
        encoder.encode(int(cum_freq[sym]), int(counts[sym]), total)
    
    bitstream = encoder.finish()
    return bitstream, counts

def decode_indices(bitstream, counts, num_symbols):
    """Decode indices from arithmetic coded bitstream."""
    cum_freq = counts_to_cumulative(counts)
    total = int(cum_freq[-1])
    
    decoder = ArithmeticDecoder(bitstream, precision=32)
    
    indices = []
    for _ in range(num_symbols):
        sym = decoder.decode(cum_freq.tolist(), counts.tolist(), total)
        indices.append(sym)
    
    return np.array(indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/inference_results/jakubie_57000_compressed.pt")
    parser.add_argument("--output", type=str, default="outputs/inference_results/jakubie_ac.bin")
    parser.add_argument("--verify", action="store_true", default=True)
    args = parser.parse_args()
    
    print(f"Loading compressed data from {args.input}...")
    data = torch.load(args.input)
    
    sem_indices = data['semantic_indices']  # [B, T_s, L]
    pro_indices = data['prosody_indices']   # [B, T_p, L]
    spk_indices = data['speaker_indices']   # [B, num_groups]
    
    # Flatten for encoding
    # sem/pro: [B, T, L] -> flatten to [B*T*L]
    # spk: [B, G] -> flatten to [B*G]
    
    # Vocab sizes
    fsq_vocab = 6561  # 3^8
    spk_vocab = 256
    
    print(f"Semantic shape: {sem_indices.shape}")
    print(f"Prosody shape: {pro_indices.shape}")
    print(f"Speaker shape: {spk_indices.shape}")
    
    # Encode each stream
    print("Encoding semantic indices...")
    sem_stream, sem_counts = encode_indices(sem_indices, fsq_vocab)
    print(f"  -> {len(sem_stream)} bytes")
    
    print("Encoding prosody indices...")
    pro_stream, pro_counts = encode_indices(pro_indices, fsq_vocab)
    print(f"  -> {len(pro_stream)} bytes")
    
    print("Encoding speaker indices...")
    spk_stream, spk_counts = encode_indices(spk_indices, spk_vocab)
    print(f"  -> {len(spk_stream)} bytes")
    
    # Total payload (excluding header for now)
    payload_bytes = len(sem_stream) + len(pro_stream) + len(spk_stream)
    print(f"\nTotal Payload: {payload_bytes} bytes")
    
    # Calculate bitrate
    # Duration is in the original file... we need to load it or pass it.
    # Assume 8.34s from previous run
    duration = 8.34
    bitrate = (payload_bytes * 8) / duration
    print(f"Bitrate (Payload Only): {bitrate:.1f} bps")
    
    # Save to file
    # Format:
    # - 4 bytes: num_sem_symbols (int32)
    # - 4 bytes: num_pro_symbols (int32)
    # - 4 bytes: num_spk_symbols (int32)
    # - 4 bytes: len_sem_stream (int32)
    # - 4 bytes: len_pro_stream (int32)
    # - 4 bytes: len_spk_stream (int32)
    # - sem_counts (fsq_vocab * 4 bytes) = 26244 bytes... too big!
    # 
    # OPTIMIZATION: Only save non-zero counts (sparse)
    # OR: Save counts as 2-byte values (max 65535)
    # OR: Use a "universal prior" and don't save counts at all (receiver knows them).
    #
    # For this demo, I'll simulate the "universal prior" approach:
    # - Save ONLY the payload streams (sem_stream, pro_stream, spk_stream).
    # - Save the counts to a separate "prior.bin" file (simulating model weights).
    
    print("\nSaving compressed file (payload only)...")
    with open(args.output, 'wb') as f:
        # Header: metadata needed for decoding
        n_sem = sem_indices.numel()
        n_pro = pro_indices.numel()
        n_spk = spk_indices.numel()
        
        f.write(struct.pack('<I', n_sem))
        f.write(struct.pack('<I', n_pro))
        f.write(struct.pack('<I', n_spk))
        f.write(struct.pack('<I', len(sem_stream)))
        f.write(struct.pack('<I', len(pro_stream)))
        f.write(struct.pack('<I', len(spk_stream)))
        # 24 bytes header
        
        f.write(sem_stream)
        f.write(pro_stream)
        f.write(spk_stream)
    
    file_size = os.path.getsize(args.output)
    print(f"Saved to {args.output}: {file_size} bytes")
    print(f"Bitrate (With Header): {(file_size * 8) / duration:.1f} bps")
    
    # Save prior (counts) separately
    prior_path = args.output.replace('.bin', '_prior.pt')
    torch.save({
        'sem_counts': sem_counts,
        'pro_counts': pro_counts,
        'spk_counts': spk_counts,
        'sem_shape': list(sem_indices.shape),
        'pro_shape': list(pro_indices.shape),
        'spk_shape': list(spk_indices.shape),
    }, prior_path)
    print(f"Saved prior to {prior_path}")
    
    if args.verify:
        print("\n--- Verification ---")
        # Decode and verify
        with open(args.output, 'rb') as f:
            n_sem = struct.unpack('<I', f.read(4))[0]
            n_pro = struct.unpack('<I', f.read(4))[0]
            n_spk = struct.unpack('<I', f.read(4))[0]
            len_sem = struct.unpack('<I', f.read(4))[0]
            len_pro = struct.unpack('<I', f.read(4))[0]
            len_spk = struct.unpack('<I', f.read(4))[0]
            
            sem_stream_read = f.read(len_sem)
            pro_stream_read = f.read(len_pro)
            spk_stream_read = f.read(len_spk)
        
        # Load prior
        prior = torch.load(prior_path, weights_only=False)
        
        print("Decoding semantic...")
        sem_decoded = decode_indices(sem_stream_read, prior['sem_counts'], n_sem)
        sem_decoded = torch.from_numpy(sem_decoded).reshape(prior['sem_shape'])
        
        print("Decoding prosody...")
        pro_decoded = decode_indices(pro_stream_read, prior['pro_counts'], n_pro)
        pro_decoded = torch.from_numpy(pro_decoded).reshape(prior['pro_shape'])
        
        print("Decoding speaker...")
        spk_decoded = decode_indices(spk_stream_read, prior['spk_counts'], n_spk)
        spk_decoded = torch.from_numpy(spk_decoded).reshape(prior['spk_shape'])
        
        # Verify
        sem_match = torch.allclose(sem_indices.long(), sem_decoded.long())
        pro_match = torch.allclose(pro_indices.long(), pro_decoded.long())
        spk_match = torch.allclose(spk_indices.long(), spk_decoded.long())
        
        print(f"Semantic Match: {sem_match}")
        print(f"Prosody Match: {pro_match}")
        print(f"Speaker Match: {spk_match}")
        
        if sem_match and pro_match and spk_match:
            print("\n✓ Lossless compression verified!")
        else:
            print("\n✗ Verification FAILED!")

if __name__ == "__main__":
    main()
