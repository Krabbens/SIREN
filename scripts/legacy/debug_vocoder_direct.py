
import torch
import torchaudio
import soundfile as sf
import os
import matplotlib.pyplot as plt
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Vocoder on {device}...")
    
    # 1. Load Vocoder
    vocoder = MelVocoderBitNet().to(device)
    voc_path = "checkpoints/vocoder_mel/vocoder_latest.pt"
    if not os.path.exists(voc_path):
        print(f"Error: Vocoder checkpoint not found at {voc_path}")
        return
        
    ckpt = torch.load(voc_path, map_location=device)
    if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
    vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
    vocoder.eval()
    print("Vocoder loaded.")

    # 2. Load Audio & Compute Mel (Ground Truth)
    audio_path = "data/jakubie.wav"
    wav, sr = sf.read(audio_path)
    wav = torch.tensor(wav, dtype=torch.float32).to(device)
    if wav.dim() > 1: wav = wav.mean(dim=0) # Mono
    
    # Resample to 16k
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    
    # Normalize
    wav = wav / (wav.abs().max() + 1e-6)
    
    # Mel Transform (Exact training params)
    # n_fft=1024, hop_length=256, n_mels=80, center=False
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        center=False
    ).to(device)
    
    with torch.no_grad():
        # Compute Mel
        mel_gt = mel_transform(wav.unsqueeze(0)).clamp(min=1e-5).log()
        
        # Norm/Denorm check (Training does: (x - (-5.0)) / 3.5)
        # Vocoder expects NORMALIZED Mel?
        # Let's check train_vocoder.py...
        # ... usually Vocoders are trained on Normalized Mels or Log Mels.
        # This pipeline uses Normalization: (Mel - Mean) / Std
        
        MEL_MEAN = -5.0
        MEL_STD = 3.5
        
        # Test 1: Raw Log Mel (If vocoder handles raw)
        # Test 2: Normalized (If vocoder expects normalized)
        
        # We need to MATCH what the Flow model generates.
        # Flow model outputs NORMALIZED (approx N(0,1)).
        # Then we denormalize: mel = mel * 3.5 - 5.0
        # So Vocoder MUST expect RAW LOG MEL (approx -11 to 3).
        
        # BUT wait! If Flow generates Normalized, and we denormalize BEFORE Vocoder...
        # Then Vocoder expects Raw.
        # Let's verify inference_microhubert_pipeline.py line 214:
        # mel = mel * MEL_STD + MEL_MEAN (Denormalize)
        # audio = vocoder(mel)
        # YES. Vocoder expects RAW Log Mel.
        
        print(f"Ground Truth Mel: {mel_gt.shape}, min={mel_gt.min():.2f}, max={mel_gt.max():.2f}, mean={mel_gt.mean():.2f}")
        
        # 3. Vocode
        print("Synthesizing...")
        # Vocoder expects (B, T, 80) based on source code (it applies transpose(1,2) internally)
        # But wait, if it applies transpose(1,2), it turns (B,T,80) -> (B,80,T).
        # Conv1d needs (B,80,T).
        # My mel_gt is (B,80,T).
        # So mel_gt.transpose(1,2) would make it (B,T,80) -> Conv1d sees T channels -> FAIL.
        # So I need to pass (B,T,80).
        audio_out = vocoder(mel_gt.transpose(1, 2))
        
        # 4. Save
        path = "outputs/debug_vocoder_gt_raw.wav"
        os.makedirs("outputs", exist_ok=True)
        
        audio_out = audio_out.squeeze().cpu()
        audio_out = audio_out / (audio_out.abs().max() + 1e-6)
        sf.write(path, audio_out.numpy(), 16000)
        print(f"Saved {path}")
        
        # Spectrogram comparison
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(mel_gt.squeeze().cpu().numpy(), aspect='auto', origin='lower', vmin=-11, vmax=4)
        plt.title("Ground Truth Mel (Input to Vocoder)")
        plt.colorbar()
        
        # Create Mel from Generated Audio to verify "round trip"
        mel_gen = mel_transform(audio_out.to(device).unsqueeze(0)).clamp(min=1e-5).log()
        plt.subplot(2, 1, 2)
        plt.imshow(mel_gen.squeeze().cpu().numpy(), aspect='auto', origin='lower', vmin=-11, vmax=4)
        plt.title("Mel from Vocoded Audio (Output)")
        plt.colorbar()
        
        plt.savefig("outputs/debug_vocoder_check.png")
        print("Saved outputs/debug_vocoder_check.png")

if __name__ == "__main__":
    main()
