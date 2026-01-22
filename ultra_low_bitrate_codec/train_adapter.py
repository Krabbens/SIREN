"""
Train Feature Adapter to bridge MicroEncoder and BitVocoder.
Miminimzes Audio Reconstruction Loss (end-to-end training).

Flow: Audio -> MicroEncoder(Frozen) -> Adapter(Train) -> BitVocoder(Frozen) -> Audio_Rec
Loss: MelSpectrogramLoss(Audio, Audio_Rec)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import glob
import random
from tqdm import tqdm
import yaml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.micro_encoder import MicroEncoder
from models.bit_vocoder import BitVocoder
from models.adapter import FeatureAdapter

# Import loss from standard library or define simple one
class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=320, n_mels=80, power=1.0
        )
        
    def forward(self, pred, target):
        # pred, target: (B, T)
        mel_pred = self.mel(pred)
        mel_target = self.mel(target)
        # Log L1 loss
        loss = F.l1_loss(torch.log(mel_pred + 1e-5), torch.log(mel_target + 1e-5))
        return loss

class SimpleAudioDataset(Dataset):
    def __init__(self, audio_dir, segment_length=32000):
        self.files = glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True)
        self.files = [f for f in self.files if os.path.getsize(f) > 50000]
        self.segment_length = segment_length
        print(f"Found {len(self.files)} files")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        try:
            audio, sr = torchaudio.load(self.files[idx])
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            audio = audio.mean(0)
            
            if len(audio) > self.segment_length:
                start = random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start+self.segment_length]
            else:
                audio = F.pad(audio, (0, self.segment_length - len(audio)))
            return audio
        except:
            return torch.zeros(self.segment_length)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro_checkpoint", default="checkpoints_micro_encoder/best_model.pt")
    parser.add_argument("--vocoder_checkpoint", default="checkpoints_bitnet/best_model.pt")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/sub100bps.yaml")
    parser.add_argument("--audio_dir", default="data/audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load MicroEncoder (Frozen)
    print("Loading MicroEncoder...")
    micro = MicroEncoder().to(device)
    ckpt = torch.load(args.micro_checkpoint, map_location=device)
    micro.load_state_dict(ckpt['model_state_dict'])
    micro.eval()
    for p in micro.parameters(): p.requires_grad = False
    
    # 2. Load BitVocoder (Frozen)
    print("Loading BitVocoder...")
    # Read config for vocoder params
    with open(args.config) as f:
        conf = yaml.safe_load(f)
    voc_conf = conf['model']['vocoder']
    
    vocoder = BitVocoder(
        input_dim=512,
        dim=256, # Matching checkpoint
        n_fft=1024,
        hop_length=320,
        num_layers=voc_conf.get('num_convnext_layers', 8),
        num_res_blocks=voc_conf.get('num_res_blocks', 3)
    ).to(device)
    
    voc_ckpt = torch.load(args.vocoder_checkpoint, map_location=device)
    state = voc_ckpt['model_state_dict']
    new_state = {}
    for k, v in state.items():
        if k.startswith('model.'): new_state[k[6:]] = v
        else: new_state[k] = v
    vocoder.load_state_dict(new_state, strict=False)
    vocoder.eval()
    for p in vocoder.parameters(): p.requires_grad = False
    
    # 3. Adapter (Trainable)
    print("Initializing Adapter...")
    adapter = FeatureAdapter(in_dim=768, out_dim=512).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    
    # Loss
    criterion = MelSpectrogramLoss().to(device)
    
    # Data
    dataset = SimpleAudioDataset(args.audio_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    writer = SummaryWriter("checkpoints_adapter/tensorboard")
    os.makedirs("checkpoints_adapter", exist_ok=True)
    
    print("Starting Training...")
    global_step = 0
    
    for epoch in range(args.epochs):
        adapter.train()
        avg_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            audio = batch.to(device) # (B, T)
            
            with torch.no_grad():
                # MicroEncoder features
                audio_in = audio.unsqueeze(1) if audio.dim() == 2 else audio
                feats = micro(audio_in) # (B, Frames, 768)
                
            # Adapter
            feats_adapted = adapter(feats) # (B, Frames, 512)
            
            # Vocoder
            # Vocoder expects (B, 512, T) or (B, T, 512)
            # We have (B, T, 512). BitVocoder forward handles transpose if dim match input_dim (512).
            # If feats_adapted has shape (B, Frames, 512), dim 2 is 512. Match!
            # BUT wait, the bug from before?
            # Before: (1, 251, 512). 512 == 512. Transpose -> (1, 512, 251).
            # Then BitConv1d does transpose AGAIN to (1, 251, 512).
            # Then RMSNorm(512) works on 512.
            # So (B, Frames, 512) should be correct!
            
            # However, Vocoder forward logic:
            # if x.shape[2] == self.input_dim (512): transpose (1, 2) -> (B, 512, Frames).
            # if input is (B, Frames, 512), shape[2] is 512. So it becomes (B, 512, Frames).
            
            # If BitVocoder forward:
            # BitConv1d expects (B, C, T) = (B, 512, Frames).
            # Inside BitConv1d: x.transpose(1, 2) -> (B, Frames, 512).
            # Norm works on last dim (512). Correct.
            
            # So passing (B, Frames, 512) is correct.
            
            # BUT quantization STE allows gradient flow through BitVocoder?
            # Yes, STE allows backward. But BitVocoder is frozen here?
            # We want gradients to flow through BitVocoder back to Adapter?
            # Wait, if BitVocoder is frozen, does autograd graph path exist?
            # Yes, as long as operations are differentiable. 
            # Quantization (round) is non-differentiable but STE patches it.
            # So yes, we can backprop through frozen Vocoder!
            
            # Forward vocoder
            audio_rec = vocoder(feats_adapted.unsqueeze(1) if feats_adapted.dim()==2 else feats_adapted)
            
            # Loss
            # Ensure lengths match
            min_len = min(audio.shape[-1], audio_rec.shape[-1])
            loss = criterion(audio_rec[..., :min_len], audio[..., :min_len])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            global_step += 1
            
            if global_step % 100 == 0:
                writer.add_scalar("loss", loss.item(), global_step)
                
        print(f"Epoch {epoch} Loss: {avg_loss / len(dataloader):.4f}")
        torch.save(adapter.state_dict(), f"checkpoints_adapter/checkpoint_epoch{epoch}.pt")
        
    torch.save(adapter.state_dict(), "checkpoints_adapter/final_model.pt")

if __name__ == "__main__":
    main()
