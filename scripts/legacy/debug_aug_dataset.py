
import torch
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
import sys

# Copy-paste the class
class AugmentedAudioDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=16000, max_len=48000):
        self.files = []
        for root, dirs, files in os.walk(audio_dir):
            for f in files:
                if f.endswith(('.wav', '.flac', '.mp3')):
                    self.files.append(os.path.join(root, f))
        self.sr = sample_rate
        self.max_len = max_len
        print(f"Dataset size: {len(self.files)}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            import soundfile as sf
            wav, sr = sf.read(path)
            wav = torch.tensor(wav, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(self.max_len), torch.zeros(self.max_len)

        if wav.dim() > 1:
            wav = wav.mean(dim=-1)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
            
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / (peak + 1e-6)
            
        actual_len = wav.shape[0]
        if actual_len > self.max_len:
            start = torch.randint(0, actual_len - self.max_len, (1,)).item()
            wav = wav[start:start + self.max_len]
        elif actual_len < self.max_len:
            wav = F.pad(wav, (0, self.max_len - actual_len))
            
        orig_wav = wav.clone()
        aug_wav = wav.clone()
        
        # Augmentation
        semitones = torch.randint(-4, 5, (1,)).item()
        if semitones != 0:
            speed_factor = 2.0 ** (semitones / 12.0)
            new_sr = int(self.sr * speed_factor)
            # print(f"Resampling from {self.sr} to {new_sr} (factor {speed_factor:.2f})")
            aug_wav = torchaudio.functional.resample(aug_wav, self.sr, new_sr)
            
            if aug_wav.shape[0] > self.max_len:
                aug_wav = aug_wav[:self.max_len]
            else:
                aug_wav = F.pad(aug_wav, (0, self.max_len - aug_wav.shape[0]))

        return aug_wav, orig_wav

if __name__ == "__main__":
    dataset = AugmentedAudioDataset("data/audio", max_len=48000)
    loader = DataLoader(dataset, batch_size=16, num_workers=0)
    
    print("Testing data loading speed...")
    start = time.time()
    for i, batch in enumerate(loader):
        print(f"Batch {i} loaded. Shape: {batch[0].shape}")
        if i >= 5: break
    end = time.time()
    print(f"Loaded 5 batches in {end-start:.2f}s")
