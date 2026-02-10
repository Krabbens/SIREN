
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

# Import MelVocoder for Vocoding the GT Mel
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def get_mel_transform(device):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        n_mels=80,
        center=True,
        power=2.0
    ).to(device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Files
    gt_audio = "data/jakubie.wav"
    teacher_audio = "outputs/teacher_ref_flow31.wav"
    student_audio = "outputs/distilled_ep75_flow31.wav"
    
    output_dir = "outputs/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Vocoder (to test GT limit)
    vocoder = MelVocoderBitNet().to(device)
    try:
        voc_ckpt = torch.load("checkpoints/vocoder_mel/vocoder_latest.pt", map_location=device)
        vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in voc_ckpt.items()})
        vocoder.eval()
    except Exception as e:
        print(f"Warning: Could not load Vocoder for GT Test: {e}")
        vocoder = None

    # 2. Process GT
    print("Processing Ground Truth...")
    wav_gt, sr = sf.read(gt_audio)
    wav_gt = torch.tensor(wav_gt, dtype=torch.float32)
    if wav_gt.dim() > 1: wav_gt = wav_gt.mean(1) # fix axis for sf.read
    if sr != 16000: 
        import torchaudio.functional as AF
        wav_gt = AF.resample(wav_gt, sr, 16000)
    
    mel_transform = get_mel_transform(device)
    wav_gt_torch = wav_gt.unsqueeze(0).to(device)
    
    # Compute GT Mel
    # Ensure normalization matches training? Flow dataset uses raw audio -> mel -> log -> clamp
    # But usually audio is peak normalized? Not explicitly in precompute...
    # Wait, in precompute: wav = wav (no norm mentioned before mel?)
    # NO, wait. In precompute_flow_dataset.py line 40-70, there is NO normalization (like wav / max).
    # BUT, in `train_flow_matching.py` (not checked but assumed), it depends.
    # Let's assume standard torchaudio mel.
    
    mel_gt = mel_transform(wav_gt_torch)
    mel_gt = torch.log(torch.clamp(mel_gt, min=1e-5))
    
    # Normalize for Vocoder (Flow matches this range)
    # Flow outputs are usually trained to match this Log Mel.
    # The Vocoder was trained on Mel that was likely:
    # (mel - mean) / std ? OR
    # Just Log Mel?
    # Looking at `inference_microtransformer_v2`:
    # mel = mel * 3.5 - 5.0 (after flow gen)
    # This implies Flow generates roughly [0, 1] or similar standardized range?
    # No, Flow Matching usually learns data distribution directly.
    # If we trained Flow on (LogMel - Mean) / Std, then we must denorm.
    # The denorm `mel * 3.5 - 5.0` suggests the Flow generates data in roughly [0.?, 2.?] 
    # to result in typical LogMel values [-11, +2].
    # Typical LogMel of silence is ~-11.5 (log(1e-5)).
    # Peak is log(something).
    
    # For GT, we have the Raw Log Mel.
    # If we feed this to Vocoder, it should work IF Vocoder was trained on Raw Log Mel.
    # If Vocoder was trained on Norm Mel, we scale.
    # Usually BitVocoder is trained on Raw Log Mel (range [-11, 2]).
    
    # Vocode GT
    if vocoder:
        with torch.no_grad():
            # Range check
            print(f"GT Mel Range: {mel_gt.min():.2f} to {mel_gt.max():.2f}")
            # Clamp for stability
            mel_gt_clamped = torch.clamp(mel_gt, min=-12.0, max=4.0)
            
            # Vocoder forward
            # Input: (B, 80, T) or (B, T, 80)? 
            # MelVocoderBitNet forward expects (B, T, 80) in forward signature: `def forward(self, mel): x = mel.transpose(1, 2)`
            # BUT wait, Step 2175: `x = mel.transpose(1, 2) # (B, 80, T); x = self.input_conv(x)`
            # This confirms input MUST be (B, T, 80).
            # My mel_gt is (1, 80, T).
            mel_gt_voc = mel_gt.transpose(1, 2)
            
            audio_gt_voc = vocoder(mel_gt_voc).squeeze().cpu()
            
            sf.write(f"{output_dir}/gt_vocoded.wav", audio_gt_voc.numpy(), 16000)
            print(f"Saved {output_dir}/gt_vocoded.wav")

    # 3. Load & Plot Comparisons
    def load_mel_from_file(path):
        if not os.path.exists(path): return None
        w, s = sf.read(path)
        w = torch.tensor(w).float()
        if w.ndim > 1: w = w.mean(1)
        # Assuming 16k
        mel = mel_transform(w.unsqueeze(0).to(device))
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze().cpu().numpy()

    mel_teacher = load_mel_from_file(teacher_audio)
    mel_student = load_mel_from_file(student_audio)
    mel_gt_np = mel_gt.squeeze().cpu().numpy()

    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.title("Ground Truth (Mel -> Vocoder)")
    plt.imshow(mel_gt_np, origin='lower', aspect='auto', cmap='viridis', vmin=-11.5, vmax=2.5)
    plt.colorbar()
    
    if mel_teacher is not None:
        plt.subplot(3, 1, 2)
        plt.title("Teacher (DistilHuBERT -> Factorizer -> Flow)")
        plt.imshow(mel_teacher, origin='lower', aspect='auto', cmap='viridis', vmin=-11.5, vmax=2.5)
        plt.colorbar()
        
    if mel_student is not None:
        plt.subplot(3, 1, 3)
        plt.title("Student (MicroTransformer -> Factorizer -> Flow)")
        plt.imshow(mel_student, origin='lower', aspect='auto', cmap='viridis', vmin=-11.5, vmax=2.5)
        plt.colorbar()
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_gt_student.png")
    print(f"Saved {output_dir}/comparison_gt_student.png")

if __name__ == "__main__":
    main()
