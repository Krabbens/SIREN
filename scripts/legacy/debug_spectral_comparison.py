import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ultra_low_bitrate_codec.models.micro_hubert import MicroHuBERT
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer

def get_mel_spectrogram(wav, sr=16000, n_mels=80, hop_length=320):
    # Standard Mel Transform matching training
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        win_length=1024,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=8000,
        power=2.0,
        normalized=False 
    ).to(wav.device)
    
    mel = mel_transform(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Config
    with open("src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml") as f:
        config = yaml.safe_load(f)
        
    # Load Models
    print("Loading models...")
    # Factorizer
    factorizer = InformationFactorizerV2(config).to(device)
    # Using the GOLDEN STEP model
    ckpt_path = "checkpoints/factorizer_microhubert_finetune/factorizer_best_step.pt"
    if not os.path.exists(ckpt_path):
        print(f"Warning: {ckpt_path} not found, falling back to factorizer.pt")
        ckpt_path = "checkpoints/factorizer_microhubert_finetune/factorizer.pt"

    ckpt = torch.load(ckpt_path, map_location=device)
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}, strict=False)
    
    # MicroHuBERT
    micro = MicroHuBERT().to(device)
    ckpt = torch.load("checkpoints/microhubert/microhubert_ep95.pt", map_location=device)
    micro.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
    
    # Quantizers
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    # CONFIG-BASED INIT (Fixes the crash)
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    
    bitnet_dir = "checkpoints/checkpoints_stable/step_87000"
    sem_vq.load_state_dict(torch.load(f"{bitnet_dir}/sem_rfsq.pt", map_location=device))
    pro_vq.load_state_dict(torch.load(f"{bitnet_dir}/pro_rfsq.pt", map_location=device))
    spk_pq.load_state_dict(torch.load(f"{bitnet_dir}/spk_pq.pt", map_location=device))
    
    # Flow & Fuser
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    flow_model.load_state_dict(torch.load("checkpoints/checkpoints_flow_v2/flow_epoch20.pt", map_location=device))
    
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fuser.load_state_dict(torch.load("checkpoints/checkpoints_flow_v2/fuser_epoch20.pt", map_location=device))
    
    # Load Audio
    print("Loading audio...")
    # Using MACIEJ converted file
    wav, sr = sf.read("maciej_converted.wav")
    wav = torch.tensor(wav, dtype=torch.float32).to(device)
    if wav.dim() > 1: wav = wav.mean(dim=0)
    wav = wav / (wav.abs().max() + 1e-6)
    
    # 1. Ground Truth Mel
    print("Computing Ground Truth...")
    gt_mel = get_mel_spectrogram(wav.unsqueeze(0), hop_length=320)
    gt_mel = torch.clamp(gt_mel, min=-12.0, max=3.0)

    # 2. Generated Mel
    print("Computing Generated...")
    with torch.no_grad():
        feats = micro(wav.unsqueeze(0).unsqueeze(0)) # Add batch & channel
        sem, pro, spk = factorizer(feats)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        target_len = gt_mel.shape[2]
        cond = fuser(sem_z, pro_z, spk_z, target_len)
        
        # RK4, CFG 1.0 (Standard), Steps 50 (restored default)
        gen_mel = flow_model.solve_ode(cond, steps=50, solver='rk4', cfg_scale=1.0)
        
        # Denormalize
        gen_mel = gen_mel * 3.5 - 5.0
        gen_mel = torch.clamp(gen_mel, min=-12.0, max=3.0)
        
    # Plot
    print("Plotting...")
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.title("Ground Truth Mel (Hop 256)")
    plt.imshow(gt_mel.squeeze().cpu().numpy(), origin='lower', aspect='auto', cmap='magma', vmin=-12, vmax=3)
    plt.colorbar()
    
    plt.subplot(2, 1, 2)
    plt.title("Generated Mel (Flow RK4, CFG 1.0, 50 Steps) - Maciej")
    plt.imshow(gen_mel.squeeze().T.cpu().numpy(), origin='lower', aspect='auto', cmap='magma', vmin=-12, vmax=3)
    plt.colorbar()
    
    plt.savefig("outputs/debug_comparison.png")
    print("Saved outputs/debug_comparison.png")

if __name__ == "__main__":
    main()
