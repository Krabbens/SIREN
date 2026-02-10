
import os
import torch
import soundfile as sf
import torchaudio
import argparse
import sys

# Imports
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input wav file")
    parser.add_argument("--output", default="output_vocoded_gt.wav", help="Output wav file")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Vocoder
    print("Loading Vocoder...")
    vocoder = MelVocoderBitNet().to(device)
    try:
        voc_ckpt = torch.load("checkpoints/vocoder_mel/vocoder_latest.pt", map_location=device)
        if isinstance(voc_ckpt, dict) and 'model_state_dict' in voc_ckpt: 
            voc_ckpt = voc_ckpt['model_state_dict']
        vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in voc_ckpt.items()})
    except Exception as e:
        print(f"Failed to load vocoder: {e}")
        return

    # 2. Load Audio & Compute GT Mel
    print(f"Processing {args.input}...")
    wav, sr = sf.read(args.input)
    wav = torch.tensor(wav, dtype=torch.float32)
    
    # Resample to 16k
    if sr != 16000:
        import torchaudio.functional as F_audio
        wav = F_audio.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    
    wav = wav / (wav.abs().max() + 1e-6)
    wav = wav.unsqueeze(0).to(device)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        n_mels=80,
        f_min=0,
        f_max=8000
    ).to(device)
    
    with torch.no_grad():
        gt_mel = mel_transform(wav)
        # Log-scale + Normalize (approximate the training stats if known, or just log)
        # The vocoder likely expects log-mels. 
        # In inference_siamese.py we used: pred_denorm = pred * STD + MEAN
        # Here we just use log(mel). 
        # Check if we need to apply the SAME normalization as training?
        # Usually vocoder is robust, but ideally we match.
        # Let's try standard log-mel first.
        gt_log_mel = torch.log(torch.clamp(gt_mel, min=1e-5))
        
        # Verify shape for vocoder: Expects (B, T, C) or (B, C, T)?
        # MelVocoderBitNet forward: x = mel.transpose(1, 2) -> then conv.
        # So it expects (B, T, 80).
        # gt_log_mel is (B, 80, T).
        gt_log_mel_t = gt_log_mel.transpose(1, 2)
        
        print(f"Vocoding GT Mel shape: {gt_log_mel_t.shape}")
        
        audio_out = vocoder(gt_log_mel_t)
        
    # Save
    audio_out = audio_out.squeeze().cpu().numpy()
    sf.write(args.output, audio_out, 16000)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
