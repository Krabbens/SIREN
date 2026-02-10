
import os
import sys
import torch
import torchaudio
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))

from ultra_low_bitrate_codec.models.micro_encoder import MicroEncoder, MicroEncoderTiny
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching

def print_stats(name, tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().float()
        print(f"{name}: shape={tensor.shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f}, min={tensor.min():.4f}, max={tensor.max():.4f}")
    else:
        print(f"{name}: {type(tensor)}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Analyzing on {device}...")

    # Load Config
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize Models
    print("Initializing models...")
    # DETECTED: Encoder is Tiny (hidden=128)
    encoder = MicroEncoderTiny(hidden_dim=128, output_dim=768, num_layers=2).to(device)
    # encoder = MicroEncoder(hidden_dim=256, output_dim=768, num_layers=4).to(device)
    factorizer = InformationFactorizerV2(config).to(device)
    
    sem_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=8
    ).to(device)
    
    pro_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=8
    ).to(device)
    
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    fuser = ConditionFuserV2(
        sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512,
        sem_upsample=4, pro_upsample=8
    ).to(device)
    
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow = ConditionalFlowMatching(config).to(device)

    # Load Checkpoints
    ckpt_dir = "checkpoints/microencoder_e2e"
    
    def load_ckpt(model, name):
        # Try best, then latest
        path = os.path.join(ckpt_dir, f"{name}_best.pt")
        if not os.path.exists(path):
            # Find latest ep
            files = [f for f in os.listdir(ckpt_dir) if f.startswith(f"{name}_ep") and f.endswith(".pt")]
            if files:
                latest = sorted(files, key=lambda x: int(x.split('_ep')[1].split('.pt')[0]))[-1]
                path = os.path.join(ckpt_dir, latest)
            else:
                print(f"WARNING: No checkpoint found for {name}")
                return
        
        print(f"Loading {name} from {path}")
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}, strict=False)

    load_ckpt(encoder, "encoder")
    load_ckpt(factorizer, "factorizer")
    load_ckpt(fuser, "fuser")
    load_ckpt(flow, "flow") # Optional if frozen, but good to check

    encoder.eval()
    factorizer.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_pq.eval()
    fuser.eval()
    flow.eval()

    # Load Audio
    audio_path = "data/jakubie.wav"
    import soundfile as sf
    wav, sr = sf.read(audio_path)
    wav = torch.tensor(wav, dtype=torch.float32)
    
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    
    if wav.dim() == 2:
        wav = wav.mean(1, keepdim=True).t() # (1, T)
    else:
        wav = wav.unsqueeze(0) # (1, T)

    wav = wav.to(device)
    
    # Run Inference
    print("\n--- Inference Stats ---", flush=True)
    with torch.no_grad():
        print_stats("Input Wav", wav)
        
        # Encoder
        print("Running Encoder...", flush=True)
        try:
            features = encoder(wav)
            print_stats("Encoder Output", features)
        except Exception as e:
            print(f"Encoder Failed: {e}", flush=True)
            return

        # Factorizer
        print("Running Factorizer...", flush=True)
        try:
            sem, pro, spk = factorizer(features)
            print_stats("Factorizer Sem", sem)
            print_stats("Factorizer Pro", pro)
            print_stats("Factorizer Spk", spk)
        except Exception as e:
            print(f"Factorizer Failed: {e}", flush=True)
            return
        
        # Quantizers
        print("Running Quantizers...", flush=True)
        sem_z, _, sem_idx = sem_vq(sem)
        pro_z, _, pro_idx = pro_vq(pro)
        spk_z, _, spk_idx = spk_pq(spk)
        
        print_stats("Sem Quantized", sem_z)
        print_stats("Pro Quantized", pro_z)
        print_stats("Spk Quantized", spk_z)
        
        print(f"Sem Indices Unique: {torch.unique(sem_idx).numel()}", flush=True)
        print(f"Pro Indices Unique: {torch.unique(pro_idx).numel()}", flush=True)
        print(f"Spk Indices Unique: {torch.unique(spk_idx).numel()}", flush=True)
        
        # Fuser
        print("Running Fuser...", flush=True)
        target_len = features.shape[1] * 320 // 256 # Rough calc or just arbitrary
        target_len = 200 # Fixed for test
        
        cond = fuser(sem_z, pro_z, spk_z, target_len)
        print_stats("Fuser Output (Cond)", cond)
        
        # Flow Check
        # Generate random x1 for loss check
        x1 = torch.randn(1, 80, target_len).transpose(1, 2).to(device)
        loss = flow.compute_loss(x1, cond)
        print(f"Flow Loss (Random Target): {loss.item():.4f}")
        
        # Solve ODE
        pred = flow.solve_ode(cond, steps=10)
        print_stats("Flow Prediction (Normalized)", pred)

if __name__ == "__main__":
    main()
