"""
Fast Trainer using precomputed features.
Skips HuBERT computation entirely for ~10x speedup.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - prevents Tkinter crash
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import yaml
import argparse

from ultra_low_bitrate_codec.models.encoder import InformationFactorizer
from ultra_low_bitrate_codec.models.quantizers import VectorQuantizer, ProductQuantizer, FSQ
from ultra_low_bitrate_codec.models.decoder import SpeechDecoder
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from ultra_low_bitrate_codec.models.discriminator import HiFiGANDiscriminator
from ultra_low_bitrate_codec.data.feature_dataset import PrecomputedFeatureDataset

import torchaudio

class MelSpecComputation(nn.Module):
    def __init__(self, n_fft=1024, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            hop_length=hop_size,
            win_length=win_size,
            f_min=fmin,
            f_max=fmax
        )
    
    def forward(self, y):
        # mel usually has no parameters, but has buffers (filterbanks)
        # We can just call .to(y.device) - it's a no-op if already there
        self.mel = self.mel.to(y.device)
        spec = self.mel(y)
        spec = torch.log(torch.clamp(spec, min=1e-5))
        return spec

class FastTrainer:
    def __init__(self, config_path, feature_dir):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models (NO HuBERT - features are precomputed)
        self.factorizer = InformationFactorizer(self.config).to(self.device)
        
        quant_type = self.config['model'].get('quantizer_type', 'vq')
        if quant_type == 'fsq':
            from ultra_low_bitrate_codec.models.quantizers import FSQ
            levels = self.config['model']['fsq_levels']
            self.sem_vq = FSQ(levels).to(self.device)
            self.pro_vq = FSQ(levels).to(self.device)
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
        self.discriminator = HiFiGANDiscriminator().to(self.device)
        
        # torch.compile disabled - causes hangs on this system
        # Use cudnn.benchmark instead for some speedup
        torch.backends.cudnn.benchmark = True
        
        # Optimizer
        params = list(self.factorizer.parameters()) + \
                 list(self.sem_vq.parameters()) + \
                 list(self.pro_vq.parameters()) + \
                 list(self.spk_pq.parameters()) + \
                 list(self.decoder.parameters()) + \
                 list(self.entropy_model.parameters())
                 
        self.optimizer = optim.AdamW(params, lr=float(self.config['training']['learning_rate']), betas=(0.8, 0.99))
        self.optimizer_d = optim.AdamW(self.discriminator.parameters(), lr=float(self.config['training']['learning_rate']), betas=(0.8, 0.99))
        
        # Scheduler
        def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
                )
            return LambdaLR(optimizer, lr_lambda, last_epoch)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=self.config['training']['max_steps']
        )
        self.scheduler_d = get_linear_schedule_with_warmup(
            self.optimizer_d,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=self.config['training']['max_steps']
        )
        
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Losses
        from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss, feature_matching_loss, discriminator_loss, generator_loss
        self.mr_stft = MultiResolutionSTFTLoss().to(self.device)
        self.mel_fn = MelSpecComputation().to(self.device)
        
        # Ensure directory exists
        import os
        os.makedirs("spectrograms", exist_ok=True)
        
        # Data
        self.train_ds = PrecomputedFeatureDataset(
            feature_dir=feature_dir,
            manifest_path=self.config['data']['train_manifest'],
            max_frames=100  # Reduced for speed (~2s audio)
        )
        self.train_dl = DataLoader(
            self.train_ds, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True, 
            num_workers=4,  # Reduced - GPU is bottleneck not CPU
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=4,  # Prefetch more batches
        )
        
    def train_step(self, batch):
        features, audio = batch
        features = features.to(self.device)  # (B, T, 768)
        audio = audio.to(self.device)        # (B, T_audio)
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1) # (B, 1, T_audio)
        
        # ==========================
        # 1. Generator Forward
        # ==========================
        with torch.amp.autocast('cuda'):
            # Factorize
            sem, pro, spk = self.factorizer(features)
            
            # Quantize
            sem_z, sem_loss, sem_idx = self.sem_vq(sem)
            pro_z, pro_loss, pro_idx = self.pro_vq(pro)
            spk_z, spk_loss, spk_idx = self.spk_pq(spk)
            
            vq_loss = sem_loss + pro_loss + spk_loss
            
            # Decode (Generative)
            audio_hat = self.decoder(sem_z, pro_z, spk_z) # (B, T_out)
            
            if audio_hat.dim() == 2:
                audio_hat = audio_hat.unsqueeze(1) # (B, 1, T_out)
            
            # Match lengths
            min_len = min(audio.shape[2], audio_hat.shape[2])
            audio = audio[..., :min_len]
            audio_hat = audio_hat[..., :min_len]
            
        # ==========================
        # 2. Discriminator Step
        # ==========================
        from ultra_low_bitrate_codec.training.losses import discriminator_loss, generator_loss, feature_matching_loss
        
        self.optimizer_d.zero_grad()
        with torch.amp.autocast('cuda'):
            # D forward with detached fake
            mpd_res_d, mrd_res_d = self.discriminator(audio, audio_hat.detach())
            loss_d_mpd, _, _ = discriminator_loss(mpd_res_d[0], mpd_res_d[1])
            loss_d_mrd, _, _ = discriminator_loss(mrd_res_d[0], mrd_res_d[1])
            loss_d = loss_d_mpd + loss_d_mrd
        
        self.scaler.scale(loss_d).backward()
        self.scaler.step(self.optimizer_d)
        
        # ==========================
        # 3. Generator Step
        # ==========================
        self.optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            # G forward (no detach - need gradients)
            mpd_res, mrd_res = self.discriminator(audio, audio_hat)
            
            # GAN Loss (Fooling D)
            loss_g_mpd, _ = generator_loss(mpd_res[1])
            loss_g_mrd, _ = generator_loss(mrd_res[1])
            loss_gen_gan = loss_g_mpd + loss_g_mrd
            
            # Feature Matching Loss
            loss_fm_mpd = feature_matching_loss(mpd_res[2], mpd_res[3])
            loss_fm_mrd = feature_matching_loss(mrd_res[2], mrd_res[3])
            loss_fm = loss_fm_mpd + loss_fm_mrd
            
            # Mel Spectrogram Loss (Recon)
            orig_spec = self.mel_fn(audio.squeeze(1))
            recon_spec = self.mel_fn(audio_hat.squeeze(1))
            mel_loss = F.l1_loss(recon_spec, orig_spec) * 45.0
            
            # STFT Loss
            sc_loss, mag_loss = self.mr_stft(audio.squeeze(1), audio_hat.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * 1.0 # Tune weight if needed
            
            # Entropy
            sem_bits, pro_bits = self.entropy_model.estimate_bits(sem_idx, pro_idx)
            entropy_loss = (sem_bits.mean() + pro_bits.mean()) * 0.01
            
            # Total G Loss
            loss_g = vq_loss + entropy_loss + mel_loss + loss_gen_gan + loss_fm + stft_loss
            
            # BPS calc
            total_bits = sem_bits.sum() + pro_bits.sum() + 64 
            duration = min_len / 16000.0
            bps = total_bits / (duration * features.shape[0] + 1e-6)
        
        self.scaler.scale(loss_g).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        cur_lr = self.scheduler.get_last_lr()[0]
        
        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "vq_loss": vq_loss.item(),
            "mel_loss": mel_loss.item(),
            "stft_loss": stft_loss.item(),
            "gan_loss": loss_gen_gan.item(),
            "fm_loss": loss_fm.item(),
            "bps": bps.item(),
            "lr": cur_lr
        }

    def train_loop(self):
        steps = 0
        max_steps = self.config['training']['max_steps']
        
        pbar = tqdm(total=max_steps)
        
        while steps < max_steps:
            for batch in self.train_dl:
                metrics = self.train_step(batch)
                
                steps += 1
                self.scheduler.step()
                self.scheduler_d.step()
                pbar.update(1)
                pbar.set_postfix(metrics)
                
                if steps % 200 == 0:
                     self.save_spectrogram_comparison(steps)
                
                if steps % self.config['training']['save_every'] == 0:
                    torch.save(self.factorizer.state_dict(), f"factorizer_{steps}.pt")
                    torch.save(self.decoder.state_dict(), f"decoder_{steps}.pt")
                    torch.save(self.spk_pq.state_dict(), f"spk_pq_{steps}.pt")
                    torch.save(self.discriminator.state_dict(), f"discriminator_{steps}.pt")
                    if not isinstance(self.sem_vq, nn.Module) or isinstance(self.sem_vq, FSQ):
                         # FSQ save
                         torch.save(self.sem_vq.state_dict(), f"sem_vq_{steps}.pt")
                         torch.save(self.pro_vq.state_dict(), f"pro_vq_{steps}.pt")
                    
                if steps >= max_steps:
                    break
                    
        pbar.close()
        print("Training complete!")
        
    def save_spectrogram_comparison(self, step):
        self.factorizer.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            batch = next(iter(self.train_dl))
            features, audio = batch
            features = features.to(self.device)
            audio = audio.to(self.device)
            
            if audio.dim() == 2:
                 audio = audio.unsqueeze(1)
            
            # Forward Quantized
            sem, pro, spk = self.factorizer(features)
            sem_z, _, _ = self.sem_vq(sem)
            pro_z, _, _ = self.pro_vq(pro)
            spk_z, _, _ = self.spk_pq(spk)
            
            # Continuous
            audio_hat_quant = self.decoder(sem_z, pro_z, spk_z)
            audio_hat_cont = self.decoder(sem, pro, spk_z)
            
            if audio_hat_quant.dim() == 2: audio_hat_quant = audio_hat_quant.unsqueeze(1)
            if audio_hat_cont.dim() == 2: audio_hat_cont = audio_hat_cont.unsqueeze(1)
            
            # Viz
            orig = audio[0, 0].cpu()
            recon_q = audio_hat_quant[0, 0, :orig.shape[0]].cpu()
            recon_c = audio_hat_cont[0, 0, :orig.shape[0]].cpu()
            
            orig_spec = torch.log(torch.clamp(torch.abs(torch.stft(orig, n_fft=1024, return_complex=True)), min=1e-5)).numpy()
            q_spec = torch.log(torch.clamp(torch.abs(torch.stft(recon_q, n_fft=1024, return_complex=True)), min=1e-5)).numpy()
            c_spec = torch.log(torch.clamp(torch.abs(torch.stft(recon_c, n_fft=1024, return_complex=True)), min=1e-5)).numpy()
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 12))
            
            plt.subplot(3, 1, 1)
            plt.imshow(orig_spec, origin='lower', aspect='auto', vmin=-10, vmax=5)
            plt.title(f"Original (Step {step})")
            
            plt.subplot(3, 1, 2)
            plt.imshow(c_spec, origin='lower', aspect='auto', vmin=-10, vmax=5)
            plt.title("Continuous/Oracle (No Quantization)")
            
            plt.subplot(3, 1, 3)
            plt.imshow(q_spec, origin='lower', aspect='auto', vmin=-10, vmax=5)
            plt.title(f"Quantized Reconstructed (~68 bps)")
            
            plt.tight_layout()
            plt.savefig(f"spectrograms/step_{step}_comparison.png")
            plt.close()
            
            # Save audio samples
            sr = self.config['data']['sample_rate']
            torchaudio.save(f"spectrograms/step_{step}_orig.wav", orig.unsqueeze(0), sr)
            torchaudio.save(f"spectrograms/step_{step}_recon_q.wav", recon_q.unsqueeze(0), sr)
            torchaudio.save(f"spectrograms/step_{step}_recon_c.wav", recon_c.unsqueeze(0), sr)
            
        self.factorizer.train()
        self.decoder.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='ultra_low_bitrate_codec/configs/default.yaml')
    parser.add_argument('--features', type=str, default='/home/sperm/diff/data/features_train')
    args = parser.parse_args()
    
    trainer = FastTrainer(args.config, args.features)
    trainer.train_loop()
