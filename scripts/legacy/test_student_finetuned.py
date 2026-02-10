
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
    parser.add_argument("--output", default="outputs/student_finetuned_ep50.wav")
    parser.add_argument("--steps", type=int, default=50)
    
    # Checkpoints
    parser.add_argument("--student_ckpt", default="checkpoints/microtransformer_v2/microtransformer_ep75.pt")
    parser.add_argument("--factorizer_dir", default="checkpoints/factorizer_microhubert_finetune_v4")
    parser.add_argument("--flow_ckpt", default="checkpoints/flow_finetune_student/flow_ft_epoch50.pt")
    
    # Original Pipeline Checkpoints (for Fuser and Quantizers)
    parser.add_argument("--pipeline_dir", default="checkpoints/checkpoints_flow_v2")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("MicroTransformer + Finetuned Flow Inference")
    print("=" * 60)
    
    # 1. Load Student (EP75)
    print(f"1. Loading Student: {args.student_ckpt}")
    student = MicroTransformer(hidden_dim=384, num_layers=8).to(device)
    ckpt_s = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_s.items()}, strict=False)
    student.eval()
    
    # 2. Load Factorizer
    print(f"2. Loading Factorizer: {args.factorizer_dir}")
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path) as f: config = yaml.safe_load(f)
    
    factorizer = InformationFactorizerV2(config).to(device)
    fq_path = os.path.join(args.factorizer_dir, "factorizer.pt")
    if not os.path.exists(fq_path): fq_path = os.path.join(args.factorizer_dir, "factorizer_best_step.pt")
    ckpt_f = torch.load(fq_path, map_location=device)
    if isinstance(ckpt_f, dict) and 'model_state_dict' in ckpt_f: ckpt_f = ckpt_f['model_state_dict']
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_f.items()}, strict=False)
    factorizer.eval()

    # 3. Load Fuser (Original from Pipeline)
    print(f"3. Loading Fuser: fuser_epoch31.pt")
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fu_path = os.path.join(args.pipeline_dir, "fuser_epoch31.pt")
    if not os.path.exists(fu_path): fu_path = os.path.join(args.pipeline_dir, "fuser_epoch20.pt")
    fu_ckpt = torch.load(fu_path, map_location=device)
    if isinstance(fu_ckpt, dict) and 'model_state_dict' in fu_ckpt: fu_ckpt = fu_ckpt['model_state_dict']
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fu_ckpt.items()})
    fuser.eval()

    # 4. Load Finetuned Flow
    print(f"4. Loading Finetuned Flow: {args.flow_ckpt}")
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    
    fl_ckpt = torch.load(args.flow_ckpt, map_location=device)
    # Check if checkpoint is full state dict or just model
    if isinstance(fl_ckpt, dict) and 'model_state_dict' in fl_ckpt: 
        # Actually our finetune script saves model.state_dict() directly.
        # But let's be safe.
        fl_ckpt = fl_ckpt['model_state_dict']
    
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fl_ckpt.items()}, strict=False)
    flow_model.eval()

    # Quantizers
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
    print(f"Processing {args.input}...")
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    wav, sr = sf.read(args.input)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() > 1: wav = wav.mean(dim=0)
    if sr != 16000: 
        import torchaudio.functional as AF
        wav = AF.resample(wav, sr, 16000)
    
    # Normalization (Mean-Std)
    wav_norm = (wav - wav.mean()) / (wav.std() + 1e-6)
    target_mel_len = wav.shape[0] // 320
    
    with torch.no_grad():
        # Student
        features = student(wav_norm.unsqueeze(0).to(device))
        
        # Factorizer
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        # Fuser
        cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
        
        # Flow Generation
        print("Generating Mel...")
        mel = flow_model.solve_ode(cond, steps=args.steps, solver='rk4', cfg_scale=1.0)
        
        # Denorm (Mean=-5.0, Std=3.5)
        mel = mel * 3.5 - 5.0
        
        # Clamp for Vocoder sanity
        mel = torch.clamp(mel, min=-11.5, max=4.0)
        
        # Vocode
        audio_out = vocoder(mel)
        sf.write(args.output, audio_out.squeeze().cpu().numpy(), 16000)
        print(f"Saved {args.output}")
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.imshow(mel.squeeze().T.cpu().numpy(), origin='lower', aspect='auto', cmap='viridis')
        plt.title(f"Finetuned Flow (EP50)")
        plt.savefig(args.output.replace(".wav", ".png"))
        print(f"Saved Spectrogram")

if __name__ == "__main__":
    main()
