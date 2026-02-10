import os
import sys
import torch
import torchaudio
import soundfile as sf
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt


# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from train_flow_matching import ConditionFuser

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def log_stats(name, tensor):
    if isinstance(tensor, torch.Tensor):
        t = tensor.float().detach().cpu()
        print(f"[{name}] Shape: {list(t.shape)} | Min: {t.min():.4f} | Max: {t.max():.4f} | Mean: {t.mean():.4f} | Std: {t.std():.4f}")
    else:
        print(f"[{name}] Not a tensor: {type(tensor)}")

def save_plot(data, path, title="Plot"):
    plt.figure(figsize=(10, 4))
    if data.ndim == 2:
        plt.imshow(data.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
    elif data.ndim == 1:
        plt.plot(data)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", required=True, help="Path to input audio")
    parser.add_argument("--output_dir", default="debug_intermediates")
    # Using the checkpoints identified in run_inference_pipeline.py
    parser.add_argument("--flow_ckpt", default="checkpoints/checkpoints_flow_v6/flow_epoch4.pt") 
    parser.add_argument("--fuser_ckpt", default="checkpoints/checkpoints_flow_v6/fuser_epoch4.pt")
    parser.add_argument("--vocoder_ckpt", default="checkpoints/checkpoints_bitnet_mel_v3/mel_vocoder_epoch3.pt")
    parser.add_argument("--factorizer_ckpt", default="checkpoints/checkpoints_stable/step_87000/decoder.pt")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Config
    config = load_config(args.config)
    config['model']['decoder']['fusion_dim'] = 100 

    # 2. Load Models
    print("\n--- Loading Models ---")
    factorizer = InformationFactorizerV2(config).to(device).eval()
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device).eval()
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device).eval()
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device).eval()

    # Load Factorizer Weights
    ckpt_dir = os.path.dirname(args.factorizer_ckpt)
    def load_helper(name, obj):
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location=device)
            new_d = {k.replace("_orig_mod.", ""): v for k,v in d.items()}
            obj.load_state_dict(new_d)
            print(f"Loaded {name}")
    
    load_helper("factorizer", factorizer)
    load_helper("sem_rfsq", sem_vq)
    load_helper("pro_rfsq", pro_vq)
    load_helper("spk_pq", spk_pq)

    # Flow Matching
    flow_model = ConditionalFlowMatching(config).to(device).eval()
    flow_model.load_state_dict(torch.load(args.flow_ckpt, map_location=device))
    print(f"Loaded Flow Model")

    fuser = ConditionFuser(
        config['model']['semantic']['output_dim'],
        config['model']['prosody']['output_dim'],
        256, 
        512
    ).to(device).eval()
    fuser.load_state_dict(torch.load(args.fuser_ckpt, map_location=device))
    print(f"Loaded Fuser")

    # BitVocoder
    vocoder = BitVocoder(
        input_dim=100, 
        dim=512, 
        num_layers=12,
        num_res_blocks=2,
        hop_length=256 
    ).to(device).eval()
    vocoder.load_state_dict(torch.load(args.vocoder_ckpt, map_location=device))
    print(f"Loaded Vocoder")

    # 3. Process Audio
    print(f"\n--- Processing Audio: {args.input_audio} ---")
    audio, sr = sf.read(args.input_audio)
    if audio.ndim > 1: audio = audio.mean(1)
    
    # Resample
    audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    if sr != 16000:
        audio_16k = torchaudio.functional.resample(audio_t, sr, 16000)
    else:
        audio_16k = audio_t
    log_stats("Audio 16k", audio_16k)

    # 4. Extract HubERT
    print("\n--- Extracting HuBERT ---")
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()

    inputs = feature_extractor(audio_16k.squeeze(0).cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        hubert_out = hubert(inputs.input_values.to(device)).last_hidden_state
    log_stats("HuBERT Output", hubert_out)
    save_plot(hubert_out[0].cpu().numpy(), os.path.join(args.output_dir, "01_hubert.png"), "HuBERT Features")

    # 5. Factorize & Quantize
    print("\n--- Factorization & Quantization ---")
    with torch.no_grad():
        sem, pro, spk = factorizer(hubert_out)
        log_stats("Semantic (Raw)", sem)
        log_stats("Prosody (Raw)", pro)
        log_stats("Speaker (Raw)", spk)
        
        save_plot(sem[0].cpu().numpy(), os.path.join(args.output_dir, "02_sem_raw.png"), "Semantic Raw")
        save_plot(pro[0].cpu().numpy(), os.path.join(args.output_dir, "02_pro_raw.png"), "Prosody Raw")
        
        # DEBUG QUANTIZER
        print("\n--- Inspecting Sem FSQ ---")
        if hasattr(sem_vq, 'quantizers'):
             print(f"Levels: {sem_vq.quantizers[0]._levels}")
        
        if getattr(sem_vq, 'has_proj', False):
             print(f"Has Projection: True")
             log_stats("Input Proj Weight", sem_vq.input_proj.weight)
             log_stats("Input Proj Bias", sem_vq.input_proj.bias)
        else:
             print(f"Has Projection: False")
        
        # Quantize
        # FSQ returns: z_q, loss, indices
        sem_q, sem_loss, sem_indices = sem_vq(sem)
        pro_q, pro_loss, pro_indices = pro_vq(pro)
        spk_q, spk_loss, spk_indices = spk_pq(spk)

        log_stats("Semantic (Quant)", sem_q)
        log_stats("Prosody (Quant)", pro_q)
        log_stats("Speaker (Quant)", spk_q)

        # Check Code Usage
        # indices shape: (B, T, levels)
        print(f"Sem Indices Unique: {len(torch.unique(sem_indices))}")
        print(f"Pro Indices Unique: {len(torch.unique(pro_indices))}")
        print(f"Spk Indices Unique: {len(torch.unique(spk_indices))}")

    # 6. Flow Generation
    target_duration_sec = audio_16k.shape[1] / 16000
    target_frames = int(target_duration_sec * 24000 / 256)
    print(f"\n--- Flow Generation (Steps: 50, Target Frames: {target_frames}) ---")
    
    with torch.no_grad():
        # Fuse
        cond = fuser(sem_q, pro_q, spk_q, target_frames)
        log_stats("Fused Condition", cond)
        save_plot(cond[0].cpu().numpy(), os.path.join(args.output_dir, "03_cond_fused.png"), "Fused Condition")
        
        # Generate
        # Start from random noise
        x_t = torch.randn(1, target_frames, 100, device=device)
        steps = 50
        dt = 1.0 / steps
        
        # Euler Integration
        for i in range(steps):
            t = torch.tensor([i / steps], device=device).view(1)
            v_pred = flow_model(x_t, t, cond)
            x_t = x_t + v_pred * dt
        
        mel_pred = x_t
        log_stats("Mel Predicted (Normalized)", mel_pred)
        save_plot(mel_pred[0].cpu().numpy(), os.path.join(args.output_dir, "04_mel_pred_norm.png"), "Mel Predicted (Normalized)")

        # Denormalize (USING TRAINING STATS: -3.0, 4.0)
        MEL_MEAN_TRAIN, MEL_STD_TRAIN = -3.0, 4.0
        mel_denorm = mel_pred * MEL_STD_TRAIN + MEL_MEAN_TRAIN
        log_stats("Mel Denormalized (Stats: -3.0/4.0)", mel_denorm)
        save_plot(mel_denorm[0].cpu().numpy(), os.path.join(args.output_dir, "05_mel_denorm.png"), "Mel Denormalized")


    # 7. Vocode (Flow Output)
    print("\n--- Vocoding (Flow Output) ---")
    with torch.no_grad():
        mel_input = mel_denorm.transpose(1, 2)
        audio_pred = vocoder(mel_input)
        log_stats("Audio Pred (Flow)", audio_pred)

    output_path = os.path.join(args.output_dir, "reconstructed_flow.wav")
    sf.write(output_path, audio_pred.squeeze().cpu().numpy(), 24000)
    print(f"Saved Flow-Audio to {output_path}")

    # 8. Vocode (Ground Truth Mel)
    print("\n--- Vocoding (GT Mel - Sanity Check) ---")
    # We need to extract Mel exactly as precompute_flow_dataset.py does
    # Re-load audio at 24k
    audio_24k = torchaudio.functional.resample(audio_t, sr, 24000).to(device)
    audio_24k = audio_24k / (audio_24k.abs().max() + 1e-6) * 0.95
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, 
        n_fft=1024, 
        win_length=1024, 
        hop_length=256, 
        n_mels=100,
        power=2.0 # Match precompute default
    ).to(device)
    
    with torch.no_grad():
        mel_gt = mel_transform(audio_24k)
        mel_gt = torch.log(mel_gt + 1e-5) # (1, 100, T)
        
        # Vocoder expects (B, 100, T)
        log_stats("GT Mel", mel_gt)
        save_plot(mel_gt[0].transpose(0, 1).cpu().numpy(), os.path.join(args.output_dir, "00_gt_mel.png"), "GT Mel")
        
        audio_gt_recon = vocoder(mel_gt)
        log_stats("Audio Pred (GT)", audio_gt_recon)
        
    output_path_gt = os.path.join(args.output_dir, "reconstructed_gt_vocoder.wav")
    sf.write(output_path_gt, audio_gt_recon.squeeze().cpu().numpy(), 24000)
    print(f"Saved GT-Audio to {output_path_gt}")

if __name__ == "__main__":
    main()
