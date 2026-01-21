
import torch
import os
import sys
import argparse
import yaml
import soundfile as sf
import torchaudio

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class BitUnpacker:
    def __init__(self, data):
        self.data = data
        self.idx = 0
        self.buffer = 0
        self.bits_in_buffer = 0
        
    def read(self, bits):
        while self.bits_in_buffer < bits:
            if self.idx >= len(self.data):
                # Padding checks? Assume zeros if needed, or error
                 self.buffer = (self.buffer << 8)
                 self.bits_in_buffer += 8
                 # Should technically stop?
                 # For robustness, let's just shift zeros
            else:
                byte = self.data[self.idx]
                self.idx += 1
                self.buffer = (self.buffer << 8) | byte
                self.bits_in_buffer += 8
        
        # Extract top 'bits'
        offset = self.bits_in_buffer - bits
        val = (self.buffer >> offset) & ((1 << bits) - 1)
        self.bits_in_buffer -= bits
        return val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem_bin", type=str, default="l1_stream_sem.bin")
    parser.add_argument("--pro_bin", type=str, default="l1_stream_pro.bin")
    parser.add_argument("--ref_audio", type=str, default="jakubie.wav", help="For speaker embedding")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stable")
    parser.add_argument("--step", type=int, default=37000)
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--output", type=str, default="reconstructed_from_bin.wav")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    print("Loading models...")
    # We need Factorizer for Speaker Ref
    factorizer = InformationFactorizerV2(config).to(device)
    decoder = SpeechDecoderV2(config).to(device)
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
    load_part(decoder, 'decoder')
    load_part(sem_vq, 'sem_rfsq')
    load_part(pro_vq, 'pro_rfsq')
    
    # 1. Get Speaker Embedding
    print(f"Extracting speaker ref from {args.ref_audio}...")
    y, sr = torchaudio.load(args.ref_audio)
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
        _, _, spk_ref = factorizer(feats)
        
    # 2. Decode Bins
    print("Unpacking bits...")
    with open(args.sem_bin, 'rb') as f: sem_data = f.read()
    with open(args.pro_bin, 'rb') as f: pro_data = f.read()
    
    unpacker_sem = BitUnpacker(sem_data)
    unpacker_pro = BitUnpacker(pro_data)
    
    # How many tokens?
    # We don't know exact count from bin.
    # But files are small.
    # Sem: 234 bytes = 1872 bits. 1872 / 3 = 624 tokens.
    # BitUnpacker doesn't know when to stop.
    # We can read until we run out (or calc max).
    
    sem_ind_list = []
    # Calculate expected tokens
    num_sem_tokens = (len(sem_data) * 8) // 3
    for _ in range(num_sem_tokens):
        sem_ind_list.append(unpacker_sem.read(3))
        
    pro_ind_list = []
    num_pro_tokens = (len(pro_data) * 8) // 3
    for _ in range(num_pro_tokens):
        pro_ind_list.append(unpacker_pro.read(3))

    # Convert to Tensor
    # Shape: (1, T, 1) - only Level 1
    sem_indices = torch.tensor(sem_ind_list, device=device).long().unsqueeze(0).unsqueeze(-1)
    pro_indices = torch.tensor(pro_ind_list, device=device).long().unsqueeze(0).unsqueeze(-1)
    
    print(f"Restored Sem Indices: {sem_indices.shape}")
    print(f"Restored Pro Indices: {pro_indices.shape}")
    
    # 3. Quantize to Vectors
    with torch.no_grad():
        # from_indices expects (B, T, Levels)
        # We pass (B, T, 1). ResidualFSQ loop will stop after 1st level.
        z_sem = sem_vq.from_indices(sem_indices) # [:, 0]
        z_pro = pro_vq.from_indices(pro_indices)
        
        # 4. Decode
        print("Decoding to audio...")
        y_hat = decoder(z_sem, z_pro, spk_ref)
        
    # Save
    path = args.output
    sf.write(path, y_hat.squeeze().cpu().numpy(), 16000)
    print(f"Saved reconstructed audio to {path}")

if __name__ == "__main__":
    main()
