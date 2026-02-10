
import os
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import yaml
import soundfile as sf
import argparse
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# Models
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/jakubie.wav")
    parser.add_argument("--output", default="outputs/v5_distilhubert_fix.wav")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=1.0)
    
    # Checkpoint Dir
    default_ckpt = "checkpoints/factorizer_microhubert_finetune_v5"
    if not os.path.exists(default_ckpt):
         default_ckpt = "checkpoints/factorizer_microhubert_finetune_v4"
    parser.add_argument("--checkpoint_dir", default=default_ckpt)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Official DistilHuBERT Inference (Mismatch Fix)")
    print("=" * 60)
    
    # 1. Load DistilHuBERT (Official)
    print("1. Loading DistilHuBERT (ntu-spml/distilhubert)...")
    feature_extractor = AutoModel.from_pretrained("ntu-spml/distilhubert").to(device)
    feature_extractor.eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained("ntu-spml/distilhubert")
    
    # 2. Load Pipeline Models
    print("2. Loading Pipeline...")
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path) as f: config = yaml.safe_load(f)

    # Factorizer
    factorizer = InformationFactorizerV2(config).to(device)
    fq_path = os.path.join(args.checkpoint_dir, "factorizer.pt")
    if os.path.exists(os.path.join(args.checkpoint_dir, "factorizer_best_step.pt")):
        fq_path = os.path.join(args.checkpoint_dir, "factorizer_best_step.pt")
    
    print(f"   Factorizer: {fq_path}")
    ckpt = torch.load(fq_path, map_location=device)
    if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
    factorizer.eval()

    # Quantizers (Standard FSQ initialization usually sufficient if no learnable params, 
    # but strictly we should load if they have projections. RFSQ dim=8 input=8 has NO projections.
    # So init is fine.)
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)
    
    # Fuser
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fu_path = os.path.join(args.checkpoint_dir, "fuser.pt")
    if os.path.exists(os.path.join(args.checkpoint_dir, "fuser_best_step.pt")): fu_path = os.path.join(args.checkpoint_dir, "fuser_best_step.pt")
    
    fu_ckpt = torch.load(fu_path, map_location=device)
    if isinstance(fu_ckpt, dict) and 'model_state_dict' in fu_ckpt: fu_ckpt = fu_ckpt['model_state_dict']
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fu_ckpt.items()}, strict=False)
    fuser.eval()

    # Flow
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    fl_path = os.path.join(args.checkpoint_dir, "flow_model.pt")
    if os.path.exists(os.path.join(args.checkpoint_dir, "flow_best_step.pt")): fl_path = os.path.join(args.checkpoint_dir, "flow_best_step.pt")
    
    fl_ckpt = torch.load(fl_path, map_location=device)
    if isinstance(fl_ckpt, dict) and 'model_state_dict' in fl_ckpt: fl_ckpt = fl_ckpt['model_state_dict']
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fl_ckpt.items()}, strict=False)
    flow_model.eval()

    # Vocoder
    vocoder = MelVocoderBitNet().to(device)
    voc_ckpt = torch.load("checkpoints/vocoder_mel/vocoder_latest.pt", map_location=device)
    if isinstance(voc_ckpt, dict) and 'model_state_dict' in voc_ckpt: voc_ckpt = voc_ckpt['model_state_dict']
    vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in voc_ckpt.items()})
    vocoder.eval()

    # 3. Process Audio
    print(f"3. Processing {args.input}...")
    wav, sr = sf.read(args.input)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() > 1: wav = wav.mean(dim=0)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav / (wav.abs().max() + 1e-6)
    
    target_mel_len = wav.shape[0] // 320
    
    # Extract Features (DistilHuBERT)
    with torch.no_grad():
        # Processor normalization
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        
        outputs = feature_extractor(input_values)
        features = outputs.last_hidden_state # (1, T, 768)
        
        # Scale if requested (Sweep)
        features = features * args.scale
        
        print(f"   Features: {features.shape} (DistilHuBERT)")
        
        # Factorize
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        # Fuse
        cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
        
        # Flow
        print("   Generating Mel...")
        mel = flow_model.solve_ode(cond, steps=args.steps, solver='rk4', cfg_scale=1.0)
        
        # Denorm
        mel = mel * 3.5 - 5.0
        mel = torch.clamp(mel, min=-12.0, max=3.0)
        
        # Save Spectrogram
        plt.figure(figsize=(10, 4))
        plt.imshow(mel.squeeze().T.cpu().numpy(), origin='lower', aspect='auto', cmap='viridis')
        plt.title("DistilHuBERT Fix")
        plt.savefig(args.output.replace(".wav", ".png"))
        
        # Vocode
        # Transpose for Vocoder (B, 80, T) -> (B, T, 80) logic again?
        # Debug script used: vocoder(mel.transpose(1, 2)) because input was (B, 80, T).
        # Flow outputs (B, T, 80).
        # Vocoder expects (B, T, 80) (as per debug result which passed).
        # Wait, in debug script:
        # mel_gt was (B, 80, T). I transposed to (B, T, 80). And it worked.
        # Flow output 'mel' is (B, T, 80) (Time first).
        # So passes directly.
        audio_out = vocoder(mel)
        
        # Save
        audio_out = audio_out.squeeze().cpu()
        audio_out = audio_out / (audio_out.abs().max() + 1e-6)
        sf.write(args.output, audio_out.numpy(), 16000)
        print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
