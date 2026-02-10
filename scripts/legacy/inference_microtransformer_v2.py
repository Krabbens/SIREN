
import os
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import yaml
import soundfile as sf
import argparse

# Models
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet
from ultra_low_bitrate_codec.models.micro_transformer import MicroTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/jakubie.wav")
    parser.add_argument("--output", default="outputs/ep5_distilled_fix.wav")
    parser.add_argument("--steps", type=int, default=50)
    
    # Checkpoints
    parser.add_argument("--student_ckpt", default="checkpoints/microtransformer_v2/microtransformer_ep5.pt")
    parser.add_argument("--factorizer_dir", default="checkpoints/factorizer_microhubert_finetune_v4")
    parser.add_argument("--flow_dir", default="checkpoints/checkpoints_flow_v2")
    parser.add_argument("--flow_suffix", default="epoch20")
    parser.add_argument("--use_rvq", action="store_true", help="Use RVQ bottleneck")
    parser.add_argument("--rvq_quantizers", type=int, default=8)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("MicroTransformer Distilled (EP5) Inference")
    print("=" * 60)
    
    # 1. Load Student
    print(f"1. Loading Student: {args.student_ckpt}")
    student = MicroTransformer(
        hidden_dim=384, 
        num_layers=8,
        use_rvq=args.use_rvq,
        rvq_num_quantizers=args.rvq_quantizers
    ).to(device)
    
    ckpt_s = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_s.items()})
    student.eval()
    
    # 2. Load Factorizer
    print(f"2. Loading Factorizer (v4): {args.factorizer_dir}")
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path) as f: config = yaml.safe_load(f)
    
    factorizer = InformationFactorizerV2(config).to(device)
    fq_path = os.path.join(args.factorizer_dir, "factorizer.pt")
    if not os.path.exists(fq_path): fq_path = os.path.join(args.factorizer_dir, "factorizer_best_step.pt")
    ckpt_f = torch.load(fq_path, map_location=device)
    if isinstance(ckpt_f, dict) and 'model_state_dict' in ckpt_f: ckpt_f = ckpt_f['model_state_dict']
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_f.items()}, strict=False)
    factorizer.eval()

    # 3. Load Fuser & Flow
    print(f"3. Loading Flow: {args.flow_suffix}")
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fu_path = os.path.join(args.flow_dir, f"fuser_{args.flow_suffix}.pt")
    fu_ckpt = torch.load(fu_path, map_location=device)
    if isinstance(fu_ckpt, dict) and 'model_state_dict' in fu_ckpt: fu_ckpt = fu_ckpt['model_state_dict']
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fu_ckpt.items()})
    fuser.eval()

    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    fl_path = os.path.join(args.flow_dir, f"flow_{args.flow_suffix}.pt")
    fl_ckpt = torch.load(fl_path, map_location=device)
    if isinstance(fl_ckpt, dict) and 'model_state_dict' in fl_ckpt: fl_ckpt = fl_ckpt['model_state_dict']
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fl_ckpt.items()})
    flow_model.eval()

    # 4. Quantizers
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)

    # 5. Vocoder
    vocoder = MelVocoderBitNet().to(device)
    voc_ckpt = torch.load("checkpoints/vocoder_mel/vocoder_latest.pt", map_location=device)
    if isinstance(voc_ckpt, dict) and 'model_state_dict' in voc_ckpt: voc_ckpt = voc_ckpt['model_state_dict']
    vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in voc_ckpt.items()})
    vocoder.eval()

    # 6. Process
    print(f"4. Processing {args.input}...")
    wav, sr = sf.read(args.input)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() > 1: wav = wav.mean(dim=0)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    
    # Normalization (Mean-Std to match student training)
    wav_norm = (wav - wav.mean()) / (wav.std() + 1e-6)
    target_mel_len = wav.shape[0] // 320
    
    with torch.no_grad():
        if args.use_rvq:
            features, _, rvq_indices = student(wav_norm.unsqueeze(0).to(device))
            print(f"   RVQ Indices shape: {rvq_indices.shape}")
        else:
            features = student(wav_norm.unsqueeze(0).to(device))
        
        # Factorizer -> Latents
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        # Fuser -> Cond
        cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
        
        # Flow -> Mel
        print("   Generating Mel...")
        mel = flow_model.solve_ode(cond, steps=args.steps, solver='rk4', cfg_scale=1.0)
        
        # Denorm
        mel = mel * 3.5 - 5.0
        
        # Save Spectrogram
        plt.figure(figsize=(10, 4))
        plt.imshow(mel.squeeze().T.cpu().numpy(), origin='lower', aspect='auto', cmap='viridis')
        plt.title(f"MicroTransformer EP5 (Distilled)")
        plt.savefig(args.output.replace(".wav", ".png"))
        
        # Vocode
        audio_out = vocoder(mel)
        sf.write(args.output, audio_out.squeeze().cpu().numpy(), 16000)
        print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
