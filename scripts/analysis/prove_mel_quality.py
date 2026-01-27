
import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from train_flow_matching import ConditionFuser

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def gaussian_blur(x, kernel_size=9, sigma=2.0):
    x = x.transpose(1, 2)
    pad = kernel_size // 2
    k = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    k = torch.exp(-0.5 * (k / sigma)**2)
    k = k / k.sum()
    k = k.view(1, 1, -1).to(x.device)
    k = k.expand(x.shape[1], 1, -1)
    x_blur = F.conv1d(x, k, padding=pad, groups=x.shape[1])
    return x_blur.transpose(1, 2)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Files
    audio_path = "data/jakubie_16k.wav"
    config_path = "ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    flow_ckpt = "checkpoints/checkpoints_flow_v6/flow_epoch4.pt"
    fuser_ckpt = "checkpoints/checkpoints_flow_v6/fuser_epoch4.pt"
    factorizer_ckpt = "checkpoints/checkpoints_stable/step_87000/decoder.pt"
    
    # 1. Load Audio and Extract GT Mel
    print("Loading Audio & GT Mel...")
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1: audio = audio.mean(1)
    audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    if sr != 16000:
        audio_16k = torchaudio.functional.resample(audio_t, sr, 16000)
    else:
        audio_16k = audio_t
        
    # GT Mel Transform (Power=2.0 as per fixed training)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, # Target SR
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        power=2.0,
        center=False
    ).to(device)
    
    # Resample to 24k for GT Mel calculation
    audio_24k = torchaudio.functional.resample(audio_16k, 16000, 24000)
    gt_mel = mel_transform(audio_24k)
    gt_mel_log = torch.log(gt_mel.clamp(min=1e-5)).squeeze(0).transpose(0, 1) # (T, 100)
    
    # 2. Load Models
    print("Loading Models...")
    config = load_config(config_path)
    config['model']['decoder']['fusion_dim'] = 100
    
    factorizer = InformationFactorizerV2(config).to(device).eval()
    
    # Load Factorizer Weights
    ckpt_dir = os.path.dirname(factorizer_ckpt)
    def load_helper(name, obj):
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location=device)
            new_d = {k.replace("_orig_mod.", ""): v for k,v in d.items()}
            obj.load_state_dict(new_d)
    
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device).eval()
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device).eval()
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device).eval()
    
    load_helper("factorizer", factorizer)
    load_helper("sem_rfsq", sem_vq)
    load_helper("pro_rfsq", pro_vq)
    load_helper("spk_pq", spk_pq)
    
    flow_model = ConditionalFlowMatching(config).to(device).eval()
    flow_model.load_state_dict(torch.load(flow_ckpt, map_location=device))
    
    fuser = ConditionFuser(config['model']['semantic']['output_dim'], config['model']['prosody']['output_dim'], 256, 512).to(device).eval()
    fuser.load_state_dict(torch.load(fuser_ckpt, map_location=device))
    
    # 3. Pipeline Execution
    print("Run Pipeline...")
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    inputs = feature_extractor(audio_16k.squeeze(0).cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = hubert(inputs.input_values.to(device), output_hidden_states=True)
        hubert_out = outputs.hidden_states[9]
        sem, pro, spk = factorizer(hubert_out)
        sem_q, _, _ = sem_vq(sem)
        pro_q, _, _ = pro_vq(pro)
        spk_q, _, _ = spk_pq(spk)
        
        target_frames = gt_mel_log.shape[0]
        cond = fuser(sem_q, pro_q, spk_q, target_frames)
        
        # APPLY SMOOTHING FIX
        cond = gaussian_blur(cond, kernel_size=9, sigma=2.0)
        
        mel_pred_norm = flow_model.solve_ode(cond, steps=50, solver='euler')
        
        # Denormalize
        MEL_MEAN, MEL_STD = -3.2, 4.5
        mel_pred = mel_pred_norm * MEL_STD + MEL_MEAN # (1, T, 100)
        mel_pred = mel_pred.squeeze(0) # (T, 100)

    # 4. Compare & Plot
    min_len = min(gt_mel_log.shape[0], mel_pred.shape[0])
    gt = gt_mel_log[:min_len].cpu().numpy().T
    pred = mel_pred[:min_len].cpu().numpy().T
    
    print(f"\nStats Comparison:")
    print(f"GT   Mean: {gt.mean():.4f} | Std: {gt.std():.4f} | Range: [{gt.min():.2f}, {gt.max():.2f}]")
    print(f"Pred Mean: {pred.mean():.4f} | Std: {pred.std():.4f} | Range: [{pred.min():.2f}, {pred.max():.2f}]")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.imshow(gt, aspect='auto', origin='lower', cmap='viridis', vmin=-12, vmax=12)
    plt.colorbar()
    plt.title("Ground Truth Mel (Log-Power)")
    
    plt.subplot(3, 1, 2)
    plt.imshow(pred, aspect='auto', origin='lower', cmap='viridis', vmin=-12, vmax=12)
    plt.colorbar()
    plt.title("Generated Mel (Flow Model + Smoothing)")
    
    plt.subplot(3, 1, 3)
    diff = np.abs(gt - pred)
    plt.imshow(diff, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=5)
    plt.colorbar()
    plt.title("Difference (L1 Error)")
    
    plt.tight_layout()
    plt.savefig("proof_mel_comparison.png")
    print("Saved proof_mel_comparison.png")

if __name__ == "__main__":
    main()
