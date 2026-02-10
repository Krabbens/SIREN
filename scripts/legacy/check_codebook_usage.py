
import torch
import torchaudio
import soundfile as sf
import os
import yaml
import matplotlib.pyplot as plt
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.micro_hubert import MicroHuBERT

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Checking Codebooks on {device}...")
    
    # Config
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 1. Load Models
    print("Loading Models...")
    micro = MicroHuBERT().to(device)
    # micro.load_state_dict(torch.load("checkpoints/microhubert/microhubert_ep95.pt", map_location=device))
    ckpt = torch.load("checkpoints/microhubert/microhubert_ep95.pt", map_location=device)
    # Handle both plain and dict checkpoints
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    micro.load_state_dict(ckpt)
    micro.eval()
    
    factorizer = InformationFactorizerV2(config).to(device)
    ckpt_path = "checkpoints/factorizer_microhubert_finetune_v5/factorizer.pt" # V5
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/factorizer_microhubert_finetune_v5/factorizer_best_step.pt"
    
    print(f"Loading Factorizer: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in state: state = state['model_state_dict']
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    factorizer.load_state_dict(state)
    factorizer.eval()
    
    # Quantizers
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device)
    
    # Load Quantizers (if saved individually, or initialized if standard FSQ)
    # FSQ usually doesn't have learnable params except implicit level definitions, 
    # BUT ResidualFSQ likely has projections if input_dim != codebook_dim?
    # Let's check code. ResidualFSQ usually has NO learnable params if dims match.
    # But here input_dim=8. FSQ levels=[8]*8.
    # Checks...
    
    # 2. Run Inference
    audio_path = "data/jakubie.wav"
    wav, sr = sf.read(audio_path)
    wav = torch.tensor(wav, dtype=torch.float32).to(device)
    if wav.dim() > 1: wav = wav.mean(dim=0)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav / (wav.abs().max() + 1e-6)
    
    with torch.no_grad():
        features = micro(wav.unsqueeze(0)) # (1, T, 768)
        sem, pro, spk = factorizer(features)
        
        # Quantize and inspect codes
        # sem_vq returns: z_q, loss, info (indices)
        # We need INDICES to check usage.
        # FSQ implementation usually returns indices in the 3rd return value or similar.
        # Let's verify what `sem_vq` returns.
        
        sem_z, _, sem_info = sem_vq(sem)
        pro_z, _, pro_info = pro_vq(pro)
        
        # sem_info should contain indices.
        # ResidualFSQ returns "indices" normally.
        
        print("\n=== Semantic Codes Analysis ===")
        # sem_info is usually (indices_level_0, indices_level_1, ...) or just indices tensor [B, T, num_levels]
        # Let's print type
        print(f"Info Type: {type(sem_info)}")
        
        if isinstance(sem_info, torch.Tensor):
             indices = sem_info
             unique_codes = torch.unique(indices)
             print(f"Shape: {indices.shape}")
             print(f"Unique Codes Used: {len(unique_codes)}")
             print(f"Total Possible Codes (per level): {config['model']['fsq_levels'][0]}")
             print(f"Utilization: {len(unique_codes) / config['model']['fsq_levels'][0] * 100:.1f}%")
             
             # Histogram
             plt.figure()
             plt.hist(indices.cpu().flatten().numpy(), bins=range(config['model']['fsq_levels'][0]+1))
             plt.title("Semantic Code Usage")
             plt.savefig("outputs/sem_codehook_hist.png")
             print("Saved outputs/sem_codehook_hist.png")

        print("\n=== Prosody Codes Analysis ===")
        if isinstance(pro_info, torch.Tensor):
             indices = pro_info
             unique_codes = torch.unique(indices)
             print(f"Shape: {indices.shape}")
             print(f"Unique Codes Used: {len(unique_codes)}")
             print(f"Utilization: {len(unique_codes) / config['model']['fsq_levels'][0] * 100:.1f}%")

if __name__ == "__main__":
    main()
