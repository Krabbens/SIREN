
import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
import yaml
import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ultra_low_bitrate_codec.models.micro_transformer import MicroTransformer
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer

def get_mel_spectrogram(wav, sr=16000, n_mels=80, hop_length=320):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default="data/jakubie.wav")
    parser.add_argument("--ckpt", default="checkpoints/microtransformer_distill/microtransformer_ep100.pt")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Config
    try:
        with open("src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml") as f:
            config = yaml.safe_load(f)
    except:
        config = {'model': {'decoder': {'fusion_dim': 80, 'hidden_dim': 512, 'fusion_heads': 8, 'dropout': 0.1}, 
                            'fsq_levels': [8,5,5,5], 'rfsq_num_levels': 1}}
        
    print("Loading models...")
    
    # 1. Student (MicroTransformer)
    student = MicroTransformer(
        hidden_dim=384, 
        num_layers=8,
        use_rvq=True,
        rvq_num_quantizers=8,
        rvq_dropout_p=0.0
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    student.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
    student.eval()

    # 2. Factorizer
    factorizer = InformationFactorizerV2(config).to(device)
    fq_path = "checkpoints/factorizer_microhubert_finetune_v4/factorizer.pt"
    if not os.path.exists(fq_path): fq_path = "checkpoints/factorizer_microhubert_finetune_v4/factorizer_best_step.pt"
    ckpt_f = torch.load(fq_path, map_location=device)
    if isinstance(ckpt_f, dict) and 'model_state_dict' in ckpt_f: ckpt_f = ckpt_f['model_state_dict']
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_f.items()}, strict=False)
    factorizer.eval()
    
    # 3. Quantizers
    sem_vq = ResidualFSQ(levels=[8,5,5,5], num_levels=1, input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=[8,5,5,5], num_levels=1, input_dim=8).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    # 4. Fuser
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fu_path = "checkpoints/checkpoints_flow_v2/fuser_epoch20.pt"
    fu_ckpt = torch.load(fu_path, map_location=device)
    if isinstance(fu_ckpt, dict) and 'model_state_dict' in fu_ckpt: fu_ckpt = fu_ckpt['model_state_dict']
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fu_ckpt.items()})
    fuser.eval()

    # 5. Flow
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    fl_path = "checkpoints/checkpoints_flow_v2/flow_epoch20.pt"
    fl_ckpt = torch.load(fl_path, map_location=device)
    if isinstance(fl_ckpt, dict) and 'model_state_dict' in fl_ckpt: fl_ckpt = fl_ckpt['model_state_dict']
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fl_ckpt.items()})
    flow_model.eval()

    # Load Audio
    print(f"Loading audio: {args.wav}")
    if not os.path.exists(args.wav):
        print("Audio file not found.")
        return

    wav, sr = sf.read(args.wav)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        w_t = torch.tensor(wav).float()
        w_t = torchaudio.functional.resample(w_t, sr, 16000)
        wav = w_t.numpy()
    
    wav = torch.tensor(wav, dtype=torch.float32).to(device)
    wav = (wav - wav.mean()) / (wav.std() + 1e-6)

    # 1. Ground Truth Mel
    print("Computing Ground Truth...")
    gt_mel = get_mel_spectrogram(wav.unsqueeze(0), hop_length=320, n_mels=80) # Revert to 80
    # Range of standard mel is usually -11 to 2 roughly after log.
    # Flow model expects roughly -2 to 2 if standardized, but here we check raw log mel match
    
    # 2. Generated Mel
    print("Computing Generated...")
    with torch.no_grad():
        # Student -> Features
        out = student(wav.unsqueeze(0).unsqueeze(0))
        if isinstance(out, tuple):
            features = out[0]
        else:
            features = out
            
        # Factorizer
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        target_len = gt_mel.shape[2]
        
        # Fuser
        cond = fuser(sem_z, pro_z, spk_z, target_len)
        
        # Flow (Output is Normalized)
        # Assuming training used (x - (-2.9)) / 4.3, output is N(0,1)-ish or whatever model learned
        gen_norm = flow_model.solve_ode(cond, steps=50, solver='euler', cfg_scale=1.0)
        
        # Post-process: Denormalize
        MEL_MEAN, MEL_STD = -2.9, 4.3
        gen_mel = gen_norm * MEL_STD + MEL_MEAN
        
        # Clamp
        gen_mel = torch.clamp(gen_mel, min=-12.0, max=3.0)
        
        print(f"Gen Mel Stats (Denorm): min={gen_mel.min():.2f}, max={gen_mel.max():.2f}, mean={gen_mel.mean():.2f}")

    # Plot
    print("Plotting...")
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.title("Ground Truth Mel (100 bands)")
    plt.imshow(gt_mel.squeeze().cpu().numpy(), origin='lower', aspect='auto', cmap='magma', vmin=-12, vmax=3)
    plt.colorbar()
    
    plt.subplot(2, 1, 2)
    plt.title("Reconstructed Mel (MicroTransformer + Flow) [Correct Norm]")
    plt.imshow(gen_mel.squeeze().T.cpu().numpy(), origin='lower', aspect='auto', cmap='magma', vmin=-12, vmax=3)
    plt.colorbar()
    
    out_path = "outputs/comparison_rvq_gt_fixed.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
