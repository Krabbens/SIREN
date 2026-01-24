
import os
import sys
import argparse
import torch
import torchaudio
import soundfile as sf
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.tiny_diffusion import TinyDiffusionEnhancer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_clean_state_dict(model, path, device):
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

def save_spectrogram(spec, path, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.cpu().numpy(), aspect='auto', origin='lower', vmin=-11.5, vmax=2.0)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--bitnet_ckpt", default="checkpoints_ultra200bps_large/step_14500")
    parser.add_argument("--diffusion_ckpt", default="checkpoints_diffusion_finetune/best_model.pt")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    parser.add_argument("--output_dir", default="final_results")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--strength", type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading models...")
    # 1. Load BitNet
    config = load_config(args.config)
    
    # Models
    factorizer = InformationFactorizerV2(config).to(device).eval()
    decoder = SpeechDecoderV2(config).to(device).eval()
    
    fsq_levels = config['model']['fsq_levels']
    rfsq_num_levels = config['model']['rfsq_num_levels']
    
    sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['semantic']['output_dim']).to(device).eval()
    pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['prosody']['output_dim']).to(device).eval()
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device).eval()

    # Weights
    ckpt_dir = args.bitnet_ckpt
    load_clean_state_dict(factorizer, os.path.join(ckpt_dir, "factorizer.pt"), device)
    load_clean_state_dict(decoder, os.path.join(ckpt_dir, "decoder.pt"), device)
    load_clean_state_dict(sem_vq, os.path.join(ckpt_dir, "sem_rfsq.pt"), device)
    load_clean_state_dict(pro_vq, os.path.join(ckpt_dir, "pro_rfsq.pt"), device)
    load_clean_state_dict(spk_pq, os.path.join(ckpt_dir, "spk_pq.pt"), device)
    
    # HuBERT
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()

    # 2. Load Diffusion
    print(f"Loading diffusion model from {args.diffusion_ckpt}")
    diff_model = TinyDiffusionEnhancer(n_mels=80, hidden_dim=64).to(device)
    diff_model.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device))
    diff_model.eval()

    # Process Audio
    print(f"Processing {args.input}...")
    wav_np, sr = sf.read(args.input)
    if wav_np.ndim > 1: wav_np = wav_np.mean(axis=1)
    wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    
    with torch.no_grad():
        # A. BitNet Forward Pass
        # Extract features
        inputs = hubert_processor(wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        features = hubert_out.hidden_states[config['model'].get('hubert_layer', 9)]
        
        # Quantize
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)

        # B. Vocoder Components Interception
        # 1. Feature Fusion
        # Access reconstructor from decoder
        reconstructor = decoder.reconstructor
        neural_vocoder = decoder.vocoder
        vocoder_model = neural_vocoder.model
        
        # Get fused features (feats: B, T, 512)
        feats = reconstructor(sem_z, pro_z, spk_z)
        
        # 2. Predict Vocoder Components (Linear Mag 513, Phase 513)
        mag_lin, phase, log_mag_lin = vocoder_model.predict_components(feats)
        
        # 3. Prepare for Diffusion (Linear -> Mel)
        # Diffusion expects Log Mel (B, 80, T)
        # We can compute Mel from the predicted Linear Mag, or from the audio 
        # (BUT we want to fix the artifacts inherent in the prediction BEFORE synthesis if possible).
        # Actually, the diffusion was trained on "Degraded Mel" from "Degraded Audio".
        # Degraded Audio came from `vocoder(feats)`.
        # So `mag_lin` -> `istft` -> `audio` -> `stft` -> `mel` is the exact domain.
        # But we can approximate `mag_lin` -> `mel` directly via filterbank.
        
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80, power=1.0
        ).to(device)
        
        # mag_lin is (B, T, F). MelTransform expects (B, F, T) usually?
        # Check torch docs: input (..., time), returns (..., n_mels, time)
        # But here we have magnitude ready. `MelScale` transform operates on spec.
        mel_scale = torchaudio.transforms.MelScale(
            n_mels=80, sample_rate=16000, n_stft=1024//2+1
        ).to(device)
        
        # mag_lin: (B, T, 513) -> (B, 513, T)
        mag_lin_T = mag_lin.transpose(1, 2)
        mel_pred = mel_scale(mag_lin_T) # (B, 80, T)
        
        log_mel_pred = torch.log(mel_pred + 1e-5)
        
        # 4. Diffusion Enhancement
        print("Enhancing spectrogram...")
        # (B, 80, T)
        enhanced_log_mel = diff_model.sample(log_mel_pred, num_steps=args.steps)
        enhanced_mel = torch.exp(enhanced_log_mel)
        
        # 5. Inverse Mel Projection (Mel 80 -> Linear 513)
        print("Projecting Mel back to Linear...")
        inv_mel_scale = torchaudio.transforms.InverseMelScale(
            n_stft=1024//2+1, n_mels=80, sample_rate=16000
        ).to(device)
        inv_mel_scale.max_iter = 50
        
        enhanced_mag_lin_T = inv_mel_scale(enhanced_mel)
        enhanced_mag_lin = enhanced_mag_lin_T.transpose(1, 2)
        
        # 6. Vocoder Synthesis (Enhanced Mag + Original Phase)
        print("Synthesizing with Vocoder Phase...")
        # (B, T, 513)
        audio_hat = vocoder_model.synthesize(enhanced_mag_lin, phase)

        # Save
        if audio_hat.dim() > 1: audio_hat = audio_hat.squeeze()
        basename = os.path.splitext(os.path.basename(args.input))[0]
        out_wav = os.path.join(args.output_dir, f"{basename}_vocoder_enhanced.wav")
        out_spec = os.path.join(args.output_dir, f"{basename}_vocoder_enhanced_spec.png")
        
        sf.write(out_wav, audio_hat.cpu().numpy(), 16000)
        save_spectrogram(enhanced_log_mel.squeeze(), out_spec, "Diffusion Enhanced Mel")
        
        print(f"Saved: {out_wav}")
        
        # Compare with baseline (No diffusion)
        audio_base = vocoder_model.synthesize(mag_lin, phase)
        if audio_base.dim() > 1: audio_base = audio_base.squeeze()
        sf.write(os.path.join(args.output_dir, f"{basename}_baseline.wav"), audio_base.cpu().numpy(), 16000)
        print(f"Saved baseline: {basename}_baseline.wav")

if __name__ == "__main__":
    main()

