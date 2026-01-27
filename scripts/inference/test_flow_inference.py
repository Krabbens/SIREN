import torch
import torch.nn.functional as F
import yaml
import os
import sys
import soundfile as sf
import matplotlib.pyplot as plt
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching

# Define Fuser (Quick copy from train script, ideally should be in model file)
class ConditionFuser(torch.nn.Module):
    def __init__(self, sem_dim, pro_dim, spk_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Linear(sem_dim + pro_dim + spk_dim, out_dim)
    def forward(self, s, p, spk, target_len):
        s = s.transpose(1, 2)
        p = p.transpose(1, 2)
        s = F.interpolate(s, size=target_len, mode='linear', align_corners=False)
        p = F.interpolate(p, size=target_len, mode='linear', align_corners=False)
        s = s.transpose(1, 2)
        p = p.transpose(1, 2)
        spk = spk.unsqueeze(1).expand(-1, target_len, -1)
        cat = torch.cat([s, p, spk], dim=-1)
        return self.proj(cat)

def load_clean_state_dict(model, path, device):
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

def plot_spectrogram(spec, save_path):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Generated Mel Spectrogram (Flow Matching)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    config_path = "ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    ckpt_stable_dir = "checkpoints/checkpoints_stable/step_87000"
    output_dir = "flow_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['model']['decoder']['fusion_dim'] = 100 # Override for 24kHz Mel target
    
    # 2. Load Models
    print("Loading Models...")
    factorizer = InformationFactorizerV2(config).to(device).eval()
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device).eval()
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device).eval()
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device).eval()
    
    load_clean_state_dict(factorizer, os.path.join(ckpt_stable_dir, "factorizer.pt"), device)
    load_clean_state_dict(sem_vq, os.path.join(ckpt_stable_dir, "sem_rfsq.pt"), device)
    load_clean_state_dict(pro_vq, os.path.join(ckpt_stable_dir, "pro_rfsq.pt"), device)
    load_clean_state_dict(spk_pq, os.path.join(ckpt_stable_dir, "spk_pq.pt"), device)
    
    # Flow Model
    model = ConditionalFlowMatching(config).to(device).eval()
    fuser = ConditionFuser(
        config['model']['semantic']['output_dim'],
        config['model']['prosody']['output_dim'],
        256, 512 # Match 512 hidden_dim
    ).to(device).eval()
    
    # Load Latest Checkpoint
    ckpt_dir = "checkpoints/checkpoints_flow_new" # Updated dir
    if not os.path.exists(ckpt_dir):
        print(f"Directory {ckpt_dir} does not exist!")
        return
        
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("flow_epoch")]
    print(f"Found {len(ckpts)} checkpoints.")
    if not ckpts:
        print("No flow checkpoints found!")
        return
    ckpts.sort(key=lambda x: int(x.replace("flow_epoch", "").replace(".pt", "")))
    latest_flow = ckpts[-1]
    

    latest_fuser = f"fuser_epoch{latest_flow.replace('flow_epoch', '').replace('.pt', '')}.pt"
    
    print(f"Loading {latest_flow}...")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, latest_flow), map_location=device))
    fuser.load_state_dict(torch.load(os.path.join(ckpt_dir, latest_fuser), map_location=device))
    
    # 3. Input Audio (Ensure 24kHz for Mel resolution)
    input_wav = "data/audio/1246_1246_135815_000001_000000.wav"
    audio_orig, sr = sf.read(input_wav)
    audio_orig = torch.from_numpy(audio_orig).float().to(device)
    if audio_orig.dim() == 1: audio_orig = audio_orig.unsqueeze(0)
    
    # 24kHz for Mel/Vocos
    audio_24 = torchaudio.functional.resample(audio_orig, sr, 24000)
    # 16kHz for HuBERT
    audio_16 = torchaudio.functional.resample(audio_orig, sr, 16000)
    
    # 4. Prepare Tokens
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, 
        n_fft=1024, 
        win_length=1024, 
        hop_length=256, 
        n_mels=100
    ).to(device)
    
    with torch.no_grad():
        inputs = hubert_processor(audio_16.cpu().squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        outputs = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        features = outputs.hidden_states[9]
        
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        # Target length (match 24kHz audio)
        # Mel hop 256 -> T_mel = T_audio / 256
        target_len = audio_24.shape[1] // 256
        
        cond = fuser(sem_z, pro_z, spk_z, target_len)
        
        # 5. Generate
        print(f"Generating Mel Spectrogram (CFG=1.5)...")
        mel_hat_norm = model.solve_ode(cond, steps=128, solver='midpoint', cfg_scale=2.5) # Increased quality
        
        # Denormalize
        # Training: (x - (-5.0)) / 3.5
        # Inference: x = x_hat * 3.5 + (-5.0)
        # Note: Vocos expects LOG Mel, we are already in Log-Mel space due to training.
        mel_hat = mel_hat_norm * 3.5 - 5.0
        
        # --- DEBUG VIZ ---
        # Get GT Mel
        mel_gt = mel_transform(audio_24)
        mel_gt = torch.log(torch.clamp(mel_gt, min=1e-5)).transpose(1, 2)
        
        # Note: mel_gt is NOT normalized here, so we compare denormalized hat with raw GT. Correct.
        
        print(f"Mel GT Stats: Min={mel_gt.min():.2f}, Max={mel_gt.max():.2f}, Mean={mel_gt.mean():.2f}")
        print(f"Mel Hat Stats: Min={mel_hat.min():.2f}, Max={mel_hat.max():.2f}, Mean={mel_hat.mean():.2f}")
        print(f"Mel Hat Norm Stats: Min={mel_hat_norm.min():.2f}, Max={mel_hat_norm.max():.2f}")
        
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(mel_gt[0].transpose(0, 1).cpu().numpy(), aspect='auto', origin='lower')
        plt.title("Ground Truth Mel")
        plt.colorbar()
        
        plt.subplot(2, 1, 2)
        plt.imshow(mel_hat[0].transpose(0, 1).cpu().numpy(), aspect='auto', origin='lower')
        plt.title("Generated Mel (Flow)")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_epoch{latest_flow.replace('flow_epoch', '').replace('.pt', '')}.png"))
        plt.close()
        
    # Load Vocos
    print("Loading Vocos (High Quality 24kHz)...")
    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    
    # Transpose to (B, n_mels, T)
    mel_hat_transposed = mel_hat.transpose(1, 2)
    
    # Vocos expects Float32
    with torch.no_grad():
        audio_hat = vocos.decode(mel_hat_transposed.float())
    
    save_path = os.path.join(output_dir, f"audio_flow_{latest_flow}_vocos.wav")
    sf.write(save_path, audio_hat.cpu().squeeze().numpy().flatten(), 24000)
    print(f"Saved Vocos audio to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
