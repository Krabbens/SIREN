#!/usr/bin/env python3
"""
Compare inference: Official HuBERT vs DistilHuBERT for Flow V2 pipeline.
Generates side-by-side spectrograms for quality comparison.
"""
import os
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import yaml
import soundfile as sf

from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet
from transformers import Wav2Vec2FeatureExtractor, HubertModel, AutoModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    bitnet_dir = "checkpoints/checkpoints_bitnet_v2/step_122000"
    flow_checkpoint = "checkpoints/checkpoints_flow_v2/flow_epoch20.pt"
    fuser_checkpoint = "checkpoints/checkpoints_flow_v2/fuser_epoch20.pt"
    vocoder_checkpoint = "checkpoints/vocoder_mel/vocoder_latest.pt"
    input_wav = "data/jakubie_16k.wav"
    output_dir = "outputs/comparison"
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # =========================================================================
    # Load Models (shared pipeline)
    # =========================================================================
    print("=" * 60)
    print("Loading shared pipeline...")
    
    # Factorizer + Quantizers
    factorizer = InformationFactorizerV2(config).to(device)
    sem_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'], 
        num_levels=config['model']['rfsq_num_levels'], 
        input_dim=config['model']['semantic']['output_dim']
    ).to(device)
    pro_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'], 
        num_levels=config['model']['rfsq_num_levels'], 
        input_dim=config['model']['prosody']['output_dim']
    ).to(device)
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    
    def load_part(name, obj):
        p = os.path.join(bitnet_dir, f"{name}.pt")
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                ckpt = ckpt['model_state_dict']
            ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
            obj.load_state_dict(ckpt, strict=False)
            print(f"  Loaded {name}")
    
    load_part("factorizer", factorizer)
    load_part("sem_rfsq", sem_vq)
    load_part("pro_rfsq", pro_vq)
    load_part("spk_pq", spk_pq)
    
    # Flow Model
    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    flow_ckpt = torch.load(flow_checkpoint, map_location=device)
    if isinstance(flow_ckpt, dict) and 'model_state_dict' in flow_ckpt:
        flow_ckpt = flow_ckpt['model_state_dict']
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt.items()}, strict=False)
    flow_model.eval()
    print("  Loaded Flow Model")
    
    # FuserV2
    fuser = ConditionFuserV2(
        sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512,
        sem_upsample=4, pro_upsample=8
    ).to(device)
    fuser_ckpt = torch.load(fuser_checkpoint, map_location=device)
    if isinstance(fuser_ckpt, dict) and 'model_state_dict' in fuser_ckpt:
        fuser_ckpt = fuser_ckpt['model_state_dict']
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fuser_ckpt.items()}, strict=False)
    fuser.eval()
    print("  Loaded FuserV2")
    
    # Vocoder
    vocoder = MelVocoderBitNet().to(device)
    voc_ckpt = torch.load(vocoder_checkpoint, map_location=device)
    if isinstance(voc_ckpt, dict) and 'model_state_dict' in voc_ckpt:
        voc_ckpt = voc_ckpt['model_state_dict']
    vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in voc_ckpt.items()})
    vocoder.eval()
    print("  Loaded Vocoder")
    
    # =========================================================================
    # Load Feature Extractors
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading Feature Extractors...")
    
    # 1. Official HuBERT (94.4M params)
    print("  Loading Official HuBERT (94.4M params, 360MB)...")
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    # 2. DistilHuBERT (23.5M params)
    print("  Loading DistilHuBERT (23.5M params, 90MB)...")
    distil_processor = Wav2Vec2FeatureExtractor.from_pretrained("ntu-spml/distilhubert")
    distil_model = AutoModel.from_pretrained("ntu-spml/distilhubert").to(device).eval()
    
    # =========================================================================
    # Load Audio
    # =========================================================================
    print(f"\nLoading audio: {input_wav}")
    wav_data, sr = sf.read(input_wav)
    wav = torch.tensor(wav_data, dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.T
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(device)
        wav = wav.to(device)
        wav = resampler(wav)
    else:
        wav = wav.to(device)
    
    wav = wav / (wav.abs().max() + 1e-6)
    target_mel_len = wav.shape[1] // 320
    print(f"  Audio: {wav.shape[1]} samples -> {target_mel_len} mel frames")
    
    # =========================================================================
    # Inference with Official HuBERT
    # =========================================================================
    print("\n" + "=" * 60)
    print("Inference: Official HuBERT")
    
    with torch.no_grad():
        wav_np = wav.cpu().numpy().squeeze()
        inputs = hubert_processor(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
        hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        features_hubert = hubert_out.hidden_states[9]  # Layer 9
        print(f"  Features: {features_hubert.shape}")
        
        sem, pro, spk = factorizer(features_hubert)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
        mel_hubert = flow_model.solve_ode(cond, steps=50, solver='euler')
        
        MEL_MEAN, MEL_STD = -5.0, 2.0
        mel_hubert = mel_hubert * MEL_STD + MEL_MEAN
        mel_hubert = torch.clamp(mel_hubert, min=-12.0, max=3.0)
        
        audio_hubert = vocoder(mel_hubert)
    
    # =========================================================================
    # Inference with DistilHuBERT
    # =========================================================================
    print("\n" + "=" * 60)
    print("Inference: DistilHuBERT")
    
    with torch.no_grad():
        inputs = distil_processor(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
        distil_out = distil_model(inputs.input_values.to(device), output_hidden_states=True)
        # DistilHuBERT only has 2 layers, use last hidden state
        features_distil = distil_out.last_hidden_state
        print(f"  Features: {features_distil.shape}")
        
        sem, pro, spk = factorizer(features_distil)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
        mel_distil = flow_model.solve_ode(cond, steps=50, solver='euler')
        
        mel_distil = mel_distil * MEL_STD + MEL_MEAN
        mel_distil = torch.clamp(mel_distil, min=-12.0, max=3.0)
        
        audio_distil = vocoder(mel_distil)
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("Saving results...")
    
    # Save audio
    audio_hubert_np = (audio_hubert / (audio_hubert.abs().max() + 1e-6)).squeeze().cpu().numpy()
    audio_distil_np = (audio_distil / (audio_distil.abs().max() + 1e-6)).squeeze().cpu().numpy()
    
    sf.write(f"{output_dir}/hubert_output.wav", audio_hubert_np, 16000)
    sf.write(f"{output_dir}/distilhubert_output.wav", audio_distil_np, 16000)
    print(f"  Saved audio files")
    
    # =========================================================================
    # Plot Comparison
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    mel1 = mel_hubert.squeeze().T.cpu().numpy()
    mel2 = mel_distil.squeeze().T.cpu().numpy()
    
    im1 = axes[0].imshow(mel1, origin='lower', aspect='auto', cmap='viridis', vmin=-12, vmax=3)
    axes[0].set_title('Official HuBERT (94.4M params, 360MB)', fontsize=14)
    axes[0].set_ylabel('Mel Bin')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(mel2, origin='lower', aspect='auto', cmap='viridis', vmin=-12, vmax=3)
    axes[1].set_title('DistilHuBERT (23.5M params, 90MB) - 75% smaller!', fontsize=14)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Mel Bin')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison.png", dpi=150)
    print(f"  Saved: {output_dir}/comparison.png")
    
    # Compute difference
    min_len = min(mel1.shape[1], mel2.shape[1])
    diff = abs(mel1[:, :min_len] - mel2[:, :min_len])
    
    plt.figure(figsize=(14, 4))
    plt.imshow(diff, origin='lower', aspect='auto', cmap='hot')
    plt.colorbar(label='Absolute Difference')
    plt.title('Difference: |HuBERT - DistilHuBERT|')
    plt.xlabel('Frame')
    plt.ylabel('Mel Bin')
    plt.savefig(f"{output_dir}/difference.png", dpi=150)
    print(f"  Saved: {output_dir}/difference.png")
    
    # Statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  L1 Difference: {diff.mean():.4f}")
    print(f"  Max Difference: {diff.max():.4f}")
    print(f"  Correlation: {((mel1[:,:min_len] * mel2[:,:min_len]).sum() / (mel1[:,:min_len].std() * mel2[:,:min_len].std() * min_len * 80)):.4f}")
    
    print("\n✅ Comparison complete!")
    print(f"   Results in: {output_dir}/")


if __name__ == "__main__":
    main()
