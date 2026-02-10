import torch
import torchaudio
import soundfile as sf
import numpy as np

def main():
    # Load reference
    path = "outputs/flow_v2_e20_fixed.wav"
    wav, sr = sf.read(path)
    print(f"Loaded {path}, sr={sr}")
    
    # Resample to 24k if needed (Vocos usually 24k) or 16k?
    # Flow V2 usually 24k? Let's check sr.
    # But Mel calc depends on SR.
    # Assuming 16k based on MicroHuBERT pipeline default?
    # Or 24k? 
    # Let's try both.
    
    wav_t = torch.tensor(wav).float()
    if wav_t.dim() > 1: wav_t = wav_t.mean(0)
    
    # 1. 16k Mel (Hubert style)
    if sr != 16000:
        wav_16 = torchaudio.functional.resample(wav_t, sr, 16000)
    else:
        wav_16 = wav_t
        
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, win_length=1024, hop_length=256, n_mels=80
    )
    mel = mel_transform(wav_16)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    
    print(f"\n--- Analysis (Assuming 16kHz, 80 band) ---")
    print(f"Mean: {log_mel.mean():.4f}")
    print(f"Std:  {log_mel.std():.4f}")
    print(f"Max:  {log_mel.max():.4f}")
    print(f"Min:  {log_mel.min():.4f}")

    # 2. 24k Mel (Vocos style - just in case)
    if sr != 24000:
        wav_24 = torchaudio.functional.resample(wav_t, sr, 24000)
    else:
        wav_24 = wav_t
        
    mel_transform_24 = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mels=100
    )
    mel_24 = mel_transform_24(wav_24)
    log_mel_24 = torch.log(torch.clamp(mel_24, min=1e-5))
    
    print(f"\n--- Analysis (Assuming 24kHz, 100 band) ---")
    print(f"Mean: {log_mel_24.mean():.4f}")
    print(f"Std:  {log_mel_24.std():.4f}")
    print(f"Max:  {log_mel_24.max():.4f}")
    print(f"Min:  {log_mel_24.min():.4f}")

if __name__ == "__main__":
    main()
