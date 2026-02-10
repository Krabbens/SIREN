import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
from tqdm import tqdm
import os
import glob
import argparse
from scipy.io import wavfile
import sys

from ultra_low_bitrate_codec.models.tiny_hubert import TinyHubert
from transformers import HubertModel, Wav2Vec2FeatureExtractor

class DistillDataset(Dataset):
    def __init__(self, audio_dir, target_len=16000*5): # 5 sec chunks
        # Find all audio files
        self.files = []
        for ext in ["wav", "m4a", "flac", "mp3"]:
             self.files.extend(glob.glob(os.path.join(audio_dir, f"**/*.{ext}"), recursive=True))
        
        self.target_len = target_len
        print(f"Found {len(self.files)} files for distillation")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            # Try torchaudio first
            try:
                wav, sr = torchaudio.load(self.files[idx])
            except:
                # Fallback to scipy for wav
                if self.files[idx].endswith(".wav"):
                    sr, wav_data = wavfile.read(self.files[idx])
                    wav = torch.tensor(wav_data, dtype=torch.float32).unsqueeze(0)
                else:
                    raise ValueError("Unsupported format or corrupted file")

            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            
            # Pad or crop to target_len
            if wav.shape[-1] < self.target_len:
                wav = F.pad(wav, (0, self.target_len - wav.shape[-1]))
            else:
                start = torch.randint(0, wav.shape[-1] - self.target_len + 1, (1,)).item()
                wav = wav[:, start:start+self.target_len]
            
            return wav.squeeze(0)
        except Exception as e:
            # Return silence on error to avoid crashing the whole epoch
            return torch.zeros(self.target_len)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="data/audio")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--resume", help="Path to student checkpoint")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    log_file = open("checkpoints/distill.log", "a")
    
    def log_print(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        sys.stdout.flush()
    
    log_print("Training session started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Teacher
    print("Loading Teacher (HuBERT)...")
    teacher = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. Load Student
    print(f"Initializing Student (TinyHubert {args.num_layers}L, {args.hidden_dim}D)...")
    student = TinyHubert(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming student from {args.resume}")
        student.load_state_dict(torch.load(args.resume, map_location=device))
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    dataset = DistillDataset(args.audio_dir)
    # Using 0 workers because of the complex torchaudio/scipy mix and potential file issues
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0
        
        for i, wav in enumerate(loader):
            wav = wav.to(device)
            # Add channel dim if missing
            if wav.dim() == 2:
                pass # Already (B, T)
            
            # Get Teacher features (Layer 9)
            with torch.no_grad():
                outputs = teacher(wav, output_hidden_states=True)
                teacher_feat = outputs.hidden_states[9] # (B, T_feat, 768)
            
            # Student forward
            student_feat = student(wav)
            
            # Match lengths
            min_t = min(teacher_feat.shape[1], student_feat.shape[1])
            t_feat = teacher_feat[:, :min_t, :]
            s_feat = student_feat[:, :min_t, :]
            
            # Multi-part Loss
            mse_loss = F.mse_loss(s_feat, t_feat)
            cos_loss = 1.0 - F.cosine_similarity(s_feat, t_feat, dim=-1).mean()
            
            # We want to match the direction of vectors strongly
            loss = mse_loss + 5.0 * cos_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

            # Robust logging: print and flush
            actual_n = i + 1
            if actual_n % 50 == 0 or actual_n == len(loader):
                msg = f"Distill Epoch {epoch} | Batch {actual_n}/{len(loader)} | Loss: {loss.item():.4f} | MSE: {mse_loss.item():.4f}"
                log_print(msg)
        avg_loss = epoch_loss / len(loader)
        log_print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
        scheduler.step()
        
        # Save every epoch but also explicitly "best"
        torch.save(student.state_dict(), f"checkpoints/tiny_hubert_latest.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), f"checkpoints/tiny_hubert_best.pt")
            print(f"New best model saved! (Loss: {best_loss:.4f})")
            
        # Periodic checkpoint
        if epoch % 5 == 0:
            torch.save(student.state_dict(), f"checkpoints/tiny_hubert_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
