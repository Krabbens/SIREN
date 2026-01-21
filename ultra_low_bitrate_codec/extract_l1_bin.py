
import torch
import os
import sys
import argparse
import yaml
import time
import struct
import math
import torchaudio

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class BitPacker:
    def __init__(self):
        self.buffer = 0
        self.bits_in_buffer = 0
        self.out_bytes = bytearray()
        
    def add(self, value, bits):
        # Value must fit in bits
        value &= (1 << bits) - 1
        self.buffer = (self.buffer << bits) | value
        self.bits_in_buffer += bits
        
        while self.bits_in_buffer >= 8:
            byte = (self.buffer >> (self.bits_in_buffer - 8)) & 0xFF
            self.out_bytes.append(byte)
            self.bits_in_buffer -= 8
            
    def finish(self):
        if self.bits_in_buffer > 0:
            # Pad with zeros at the end (LSB)
            self.buffer = self.buffer << (8 - self.bits_in_buffer)
            self.out_bytes.append(self.buffer & 0xFF)
            self.bits_in_buffer = 0
        return self.out_bytes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="jakubie.wav")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stable")
    parser.add_argument("--step", type=int, default=37000)
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--output", type=str, default="l1_stream.bin")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    print("Loading models...")
    factorizer = InformationFactorizerV2(config).to(device)
    sem_vq = ResidualFSQ(config['model']['fsq_levels'], config['model']['rfsq_num_levels'], config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(config['model']['fsq_levels'], config['model']['rfsq_num_levels'], config['model']['prosody']['output_dim']).to(device)
    
    def load_part(model, name):
        path = os.path.join(args.checkpoint_dir, f"GOLDEN_step_{args.step}", f"{name}.pt")
        if not os.path.exists(path):
             path = os.path.join(args.checkpoint_dir, f"step_{args.step}", f"{name}.pt")
        
        print(f"Loading {name} from {path}...")
        sd = torch.load(path, map_location=device)
        new_sd = {k.replace("_orig_mod.", ""): v for k,v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()

    load_part(factorizer, 'factorizer')
    load_part(sem_vq, 'sem_rfsq')
    load_part(pro_vq, 'pro_rfsq')
    
    # Load Audio
    print("Processing audio...")
    y, sr = torchaudio.load(args.input)
    if sr != 16000:
        y = torchaudio.functional.resample(y, sr, 16000)
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
        _, _, sem_idx = sem_vq(sem) # (1, T, 1) -> squeezed to (1, T) or similar? 
        _, _, pro_idx = pro_vq(pro)

    # sem_idx is (B, T) or (B, T, 1)?
    # ResidualFSQ returns stacked indices. If num_levels=1, shape is (B, T, 1).
    if sem_idx.dim() == 3: sem_idx = sem_idx.squeeze(-1)
    if pro_idx.dim() == 3: pro_idx = pro_idx.squeeze(-1)
    
    # sem_idx contains packed integers (0..16M)
    # We need to unpack to get L1 (dim 0)
    # Basis: [1, 8, 64...]
    # L1 = (idx // 1) % 8
    
    sem_vals = sem_idx.reshape(-1).cpu().numpy()
    pro_vals = pro_idx.reshape(-1).cpu().numpy()
    
    print(f"Sem Vals Shape: {sem_vals.shape}")
    print(f"Total Frames: {len(sem_vals)}")
    print(f"Duration: {y.shape[-1]/16000:.2f}s")
    
    # Extract L1 (3 bits)
    sem_l1 = sem_vals % 8
    pro_l1 = pro_vals % 8
    
    import numpy as np
    print("Sem L1 Hist:", np.bincount(sem_l1.astype(int), minlength=8))
    print("Pro L1 Hist:", np.bincount(pro_l1.astype(int), minlength=8))
    
    # Pack Sem
    packer_sem = BitPacker()
    for s in sem_l1:
        packer_sem.add(s.item(), 3)
    bytes_sem = packer_sem.finish()
    
    # Pack Pro
    packer_pro = BitPacker()
    for p in pro_l1:
        packer_pro.add(p.item(), 3)
    bytes_pro = packer_pro.finish()
    
    with open(args.output.replace(".bin", "_sem.bin"), 'wb') as f:
        f.write(bytes_sem)
    with open(args.output.replace(".bin", "_pro.bin"), 'wb') as f:
        f.write(bytes_pro)
        
    # Interleaved Packing (Best for Gzip)
    # Ratio check
    ratio = len(sem_l1) // len(pro_l1)
    if ratio != 4:
        print(f"WARNING: Sem/Pro ratio is {ratio} (Expected 4). Packing might desync.")
    
    packer_inter = BitPacker()
    # Pack 4 Sem, 1 Pro
    # Assuming len(sem) is exactly 4 * len(pro)
    # Truncate sem to multiple of 4 if needed (should be exact)
    min_len = min(len(sem_l1) // 4, len(pro_l1))
    
    for i in range(min_len):
        # Pack 4 Semantic
        s_chunk = sem_l1[i*4 : (i+1)*4]
        for s in s_chunk:
            packer_inter.add(s.item(), 3)
        # Pack 1 Prosody
        p = pro_l1[i]
        packer_inter.add(p.item(), 3)
        
    bytes_inter = packer_inter.finish()
    
    with open("l1_interleaved.bin", 'wb') as f:
        f.write(bytes_inter)
        
    import gzip
    with gzip.open("l1_interleaved.bin.gz", 'wb') as f:
        f.write(bytes_inter)
        
    gz_size = os.path.getsize("l1_interleaved.bin.gz")
    duration = y.shape[-1] / 16000
    
    print(f"\n--- INTERLEAVED GZIP RESULTS ---")
    print(f"Interleaved Size: {len(bytes_inter)} bytes")
    print(f"Gzip Size:        {gz_size} bytes")
    print(f"Gzip Bitrate:     {(gz_size*8)/duration:.2f} bps")
    print(f"File:             l1_interleaved.bin.gz")
    print(f"Sem Frames: {len(sem_l1)}")
    print(f"Pro Frames: {len(pro_l1)}")
    
    raw_size_sem = len(bytes_sem) 
    raw_size_pro = len(bytes_pro)
    total_raw = raw_size_sem + raw_size_pro
    
    print(f"Total Raw Size:   {total_raw} bytes") # Should include interleaving logic overhead? No, interleaved is just bytes_inter.
    print(f"Interleaved Raw:  {len(bytes_inter)} bytes")
    print(f"Raw Bitrate:      {(len(bytes_inter)*8)/duration:.2f} bps")
    print(f"Total Bitrate: {(raw_size*8)/duration:.2f} bps")
    print(f"Files: {args.output.replace('.bin', '_sem.bin')}, {args.output.replace('.bin', '_pro.bin')}")

if __name__ == "__main__":
    main()
