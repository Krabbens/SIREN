import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np

from ultra_low_bitrate_codec.models.encoder import SpeechEncoder
from ultra_low_bitrate_codec.models.quantizers import VectorQuantizer, ProductQuantizer
from ultra_low_bitrate_codec.models.decoder import SpeechDecoder
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from ultra_low_bitrate_codec.data.dataset import AudioDataset

import torchaudio

def mel_spectrogram(y, n_fft=1024, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        hop_length=hop_size,
        win_length=win_size,
        f_min=fmin,
        f_max=fmax
    ).to(y.device)
    
    spec = mel(y)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec

class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.encoder = SpeechEncoder(self.config).to(self.device)
        
        quant_type = self.config['model'].get('quantizer_type', 'vq')
        if quant_type == 'fsq':
            from ultra_low_bitrate_codec.models.quantizers import FSQ
            levels = self.config['model']['fsq_levels']
            self.sem_vq = FSQ(levels).to(self.device)
            self.pro_vq = FSQ(levels).to(self.device) # Using same structure for prosody
        else:
            self.sem_vq = VectorQuantizer(
                dim=self.config['model']['semantic']['output_dim'],
                vocab_size=self.config['model']['semantic']['vocab_size']
            ).to(self.device)
            
            self.pro_vq = VectorQuantizer(
                dim=self.config['model']['prosody']['output_dim'],
                vocab_size=self.config['model']['prosody']['vocab_size']
            ).to(self.device)
        
        self.spk_pq = ProductQuantizer(
            input_dim=self.config['model']['speaker']['embedding_dim'],
            num_groups=self.config['model']['speaker']['num_groups'],
            codes_per_group=self.config['model']['speaker']['codes_per_group']
        ).to(self.device)
        
        self.decoder = SpeechDecoder(self.config).to(self.device)
        self.entropy_model = EntropyModel(self.config).to(self.device)
        
        # Optimizer
        params = list(self.encoder.parameters()) + \
                 list(self.sem_vq.parameters()) + \
                 list(self.pro_vq.parameters()) + \
                 list(self.spk_pq.parameters()) + \
                 list(self.decoder.parameters()) + \
                 list(self.entropy_model.parameters())
                 
        self.optimizer = optim.AdamW(params, lr=float(self.config['training']['learning_rate']))
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Data
        self.train_ds = AudioDataset(
            self.config['data']['train_manifest'],
            sample_rate=self.config['data']['sample_rate'],
            max_duration=self.config['data']['max_duration']
        )
        self.train_dl = DataLoader(
            self.train_ds, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True, 
            num_workers=4
        )
        
    def train_step(self, audio):
        audio = audio.to(self.device)
        # Ensure (B, T)
        if audio.dim() > 2:
            audio = audio.view(audio.shape[0], -1) 
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # 1. Encode
        with torch.cuda.amp.autocast():
            sem, pro, spk = self.encoder(audio)
            
            # 2. Quantize
            sem_z, sem_loss, sem_idx = self.sem_vq(sem)
            pro_z, pro_loss, pro_idx = self.pro_vq(pro)
            spk_z, spk_loss, spk_idx = self.spk_pq(spk)
            
            vq_loss = sem_loss + pro_loss + spk_loss
            
            # 3. Decode
            audio_hat = self.decoder(sem_z, pro_z, spk_z)
            
            # 4. Reconstruction Loss
            min_len = min(audio.shape[1], audio_hat.shape[1])
            audio = audio[:, :min_len]
            audio_hat = audio_hat[:, :min_len]
            
            l1_loss = F.l1_loss(audio_hat, audio)
            
            orig_spec = mel_spectrogram(audio)
            recon_spec = mel_spectrogram(audio_hat)
            spec_loss = F.l1_loss(recon_spec, orig_spec)
            
            recon_loss = l1_loss + spec_loss
            
            # 5. Entropy Loss
            sem_bits, pro_bits = self.entropy_model.estimate_bits(sem_idx, pro_idx)
            duration = min_len / 16000.0
            total_bits = sem_bits.sum() + pro_bits.sum() + 64 
            bps = total_bits / (duration * audio.shape[0] + 1e-6)
            
            entropy_loss = (sem_bits.mean() + pro_bits.mean()) * 0.01
        
        # Total Loss
        loss = recon_loss + vq_loss + entropy_loss
        
        self.optimizer.zero_grad()
        
        # AMP Backward
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "bps": bps.item()
        }

    def train_loop(self):
        steps = 0
        max_steps = self.config['training']['max_steps']
        
        pbar = tqdm(total=max_steps)
        
        while steps < max_steps:
            for audio in self.train_dl:
                metrics = self.train_step(audio)
                
                steps += 1
                pbar.update(1)
                pbar.set_postfix(metrics)
                
                if steps % self.config['training']['save_every'] == 0:
                    print(f"Saving checkpoint at step {steps}...")
                    torch.save(self.encoder.state_dict(), f"encoder_{steps}.pt")
                    torch.save(self.decoder.state_dict(), f"decoder_{steps}.pt")
                    
                    # Save Spectrogram
                    try:
                        self.save_spectrogram_comparison(steps)
                    except Exception as e:
                        print(f"Failed to save spectrogram: {e}")
                    
                if steps >= max_steps:
                    break

    def save_spectrogram_comparison(self, step):
        # Get one batch
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            try:
                audio = next(iter(self.train_dl)).to(self.device)
            except StopIteration:
                # Reload iterator if needed or just skip
                return
                
            if audio.dim() > 2:
                audio = audio.view(audio.shape[0], -1) 
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
                
            # Forward
            sem, pro, spk = self.encoder(audio)
            sem_z, _, _ = self.sem_vq(sem)
            pro_z, _, _ = self.pro_vq(pro)
            spk_z, _, _ = self.spk_pq(spk)
            audio_hat = self.decoder(sem_z, pro_z, spk_z)
            
            # Spectrograms
            # Take first item in batch
            orig = audio[0].cpu()
            recon = audio_hat[0].cpu()
            
            # Simple STFT for viz
            orig_spec = torch.log(torch.clamp(torch.abs(torch.stft(orig, n_fft=1024, return_complex=True)), min=1e-5)).numpy()
            recon_spec = torch.log(torch.clamp(torch.abs(torch.stft(recon, n_fft=1024, return_complex=True)), min=1e-5)).numpy()
            
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.imshow(orig_spec, origin='lower', aspect='auto', vmin=-10, vmax=5)
            plt.title(f"Original (Step {step})")
            plt.colorbar()
            
            plt.subplot(2, 1, 2)
            plt.imshow(recon_spec, origin='lower', aspect='auto', vmin=-10, vmax=5)
            plt.title(f"Reconstructed (Step {step})")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(f"spectrograms/step_{step}.png")
            plt.close()
            
        self.encoder.train()
        self.decoder.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train_loop()
