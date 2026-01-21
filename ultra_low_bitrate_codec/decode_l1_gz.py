
import torch
import os
import sys
import argparse
import yaml
import soundfile as sf
import torchaudio
import gzip

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
                 self.buffer = (self.buffer << 8)
                 self.bits_in_buffer += 8
            else:
                byte = self.data[self.idx]
                self.idx += 1
                self.buffer = (self.buffer << 8) | byte
                self.bits_in_buffer += 8
        
        offset = self.bits_in_buffer - bits
        val = (self.buffer >> offset) & ((1 << bits) - 1)
        self.bits_in_buffer -= bits
        return val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="l1_interleaved.bin.gz")
    parser.add_argument("--ref_audio", type=str, default="jakubie.wav")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stable")
    parser.add_argument("--step", type=int, default=37000)
    parser.add_argument("--config", type=str, default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--output", type=str, default="reconstructed_from_gz.wav")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    print("Loading models...")
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
        
    print(f"Reading {args.input}...")
    with gzip.open(args.input, 'rb') as f:
        data = f.read()
        
    print(f"Uncompressed size: {len(data)} bytes")
    unpacker = BitUnpacker(data)
    
    # We don't know exact length, but data is 117 bytes = 936 bits.
    # 936 / 6 (3+3) = 156 frames.
    # We read pairs (S, P).
    
    sem_list = []
    pro_list = []
    
    try:
        while True:
            # Check if we have enough bits left for 4S + 1P (5*3 = 15 bits)
            if unpacker.idx >= len(data) and unpacker.bits_in_buffer < 15:
                break
                
            # Read 4 Semantic
            for _ in range(4):
                s = unpacker.read(3)
                sem_list.append(s)
                
            # Read 1 Prosody
            p = unpacker.read(3)
            # Repeat Prosody 4 times to match Sem length for decoding?
            # Or just append once and let decoder upsample?
            # Decoder expects compressed inputs.
            # Decoder takes (B, T_sem, 1) and (B, T_pro, 1).
            # T_pro is T_sem / 4.
            # So we append once.
            pro_list.append(p)
            
    except Exception as e:
        print(f"Stop reading: {e}")
        
    print(f"Recovered {len(sem_list)} frames.")
    
    sem_indices = torch.tensor(sem_list, device=device).long().unsqueeze(0).unsqueeze(-1)
    pro_indices = torch.tensor(pro_list, device=device).long().unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        z_sem = sem_vq.from_indices(sem_indices)
        z_pro = pro_vq.from_indices(pro_indices)
        y_hat = decoder(z_sem, z_pro, spk_ref)
        
    path = args.output
    sf.write(path, y_hat.squeeze().cpu().numpy(), 16000)
    print(f"Saved reconstructed audio to {path}")

if __name__ == "__main__":
    main()
