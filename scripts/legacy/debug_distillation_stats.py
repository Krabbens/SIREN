
import torch
import soundfile as sf
import os
from transformers import AutoModel
from ultra_low_bitrate_codec.models.micro_transformer import MicroTransformer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_path = "data/jakubie.wav"
    student_ckpt = "checkpoints/microtransformer_v2/microtransformer_ep75.pt"
    
    # 1. Load Audio
    wav, sr = sf.read(audio_path)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() > 1: wav = wav.mean(0)
    
    # 2. Get Teacher Features (WITH MAX NORM to match precompute script)
    print("Loading Teacher (DistilHuBERT)...")
    teacher = AutoModel.from_pretrained("ntu-spml/distilhubert").to(device)
    teacher.eval()
    
    wav_t = wav / (wav.abs().max() + 1e-6)
    
    with torch.no_grad():
        t_out = teacher(wav_t.unsqueeze(0).to(device), output_hidden_states=True)
        t_feat = t_out.last_hidden_state.squeeze(0).cpu()
        
    # 3. Get Student Features (WITH MEAN-STD NORM to match student training)
    print("Loading Student (MicroTransformer)...")
    student = MicroTransformer(hidden_dim=384, num_layers=8).to(device)
    ckpt = torch.load(student_ckpt, map_location=device)
    student.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
    student.eval()
    
    wav_s = (wav - wav.mean()) / (wav.std() + 1e-6)
    
    with torch.no_grad():
        s_feat = student(wav_s.unsqueeze(0).to(device)).squeeze(0).cpu()
    
    # 4. Compare Stats
    print("\n" + "="*40)
    print("FIXED STATS COMPARISON")
    print("="*40)
    print(f"Teacher (MaxNorm)  | Std: {t_feat.std():.4f} | Mean: {t_feat.mean():.4f}")
    print(f"Student (MeanStd)  | Std: {s_feat.std():.4f} | Mean: {s_feat.mean():.4f}")
    
    min_len = min(t_feat.shape[0], s_feat.shape[0])
    cos = torch.nn.functional.cosine_similarity(t_feat[:min_len], s_feat[:min_len], dim=-1).mean()
    print(f"Mean Cosine Similarity: {cos:.4f}")

if __name__ == "__main__":
    main()
