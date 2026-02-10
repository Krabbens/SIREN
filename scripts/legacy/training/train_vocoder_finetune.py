
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import glob
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_spectrogram(y, y_hat, save_path):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
    db_transform = torchaudio.transforms.AmplitudeToDB()
    
    # Original
    if y.dim() == 1: y = y.unsqueeze(0)
    spec = db_transform(mel_transform(y.cpu())).squeeze().numpy()
    axs[0].imshow(spec, aspect='auto', origin='lower', vmin=-80, vmax=20)
    axs[0].set_title("Real")
    
    # Generated
    if y_hat.dim() == 1: y_hat = y_hat.unsqueeze(0)
    spec_hat = db_transform(mel_transform(y_hat.cpu())).squeeze().numpy()
    axs[1].imshow(spec_hat, aspect='auto', origin='lower', vmin=-80, vmax=20)
    axs[1].set_title("Generated")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# Models
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder
from ultra_low_bitrate_codec.models.discriminator import HiFiGANDiscriminator
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer

# Losses
from ultra_low_bitrate_codec.training.losses import MultiResolutionSTFTLoss

# Transformers (for HuBERT Loss)
from transformers import Wav2Vec2FeatureExtractor, HubertModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class AudioDataset(Dataset):
    def __init__(self, data_dir, segment_length=16000*2): # 2 seconds
        # Recursive search for wav files
        self.files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
        # Filter small files
        self.files = [f for f in self.files if os.path.getsize(f) > 32000] # Min 1 sec
        self.segment_length = segment_length
        print(f"Found {len(self.files)} files")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        try:
            wav, sr = torchaudio.load(self.files[idx])
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.mean(0, keepdim=True)
            
            if wav.shape[1] < self.segment_length:
                pad = self.segment_length - wav.shape[1]
                wav = F.pad(wav, (0, pad))
            else:
                start = random.randint(0, wav.shape[1] - self.segment_length)
                wav = wav[:, start:start+self.segment_length]
            
            # Normalize volume
            wav = wav / (wav.abs().max() + 1e-6) * 0.95
                
            return wav.squeeze(0)
        except:
            return torch.zeros(self.segment_length)

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
        
    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1-dg)**2)
        loss += l
        gen_losses.append(l.item())
    return loss, gen_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Full BitNet checkpoint (decoder)")
    parser.add_argument("--data_dir", default="data/audio_cv_pl", help="Path to audio training data")
    parser.add_argument("--output_dir", default="checkpoints/checkpoints_ultra200bps_gan")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4) # Generator LR
    parser.add_argument("--d_lr", type=float, default=1e-4) # Discriminator LR
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    
    config = load_config(args.config)
    
    print("Initializing Models...")
    
    # 1. Generator (Pipeline)
    model = SpeechDecoderV2(config).to(device)
    
    # MONKEY PATCH: Replace internal Vocoder with BitVocoder (SnakeBeta)
    print("Monkey-patching SpeechDecoderV2 with BitVocoder...")
    bit_vocoder = BitVocoder(
        input_dim=config['model']['decoder']['fusion_dim'],
        dim=256, # Compact BitNet dimension
        n_fft=1024,
        hop_length=320,
        num_layers=4,
        num_res_blocks=1
    ).to(device)
    model.vocoder = bit_vocoder # Hotswap!
    
    # Load Weights for Feature Reconstructor (Encoder/Decoder Backbone)
    print(f"Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Careful loading: Partial load for Reconstructor, Ignore old Vocoder
    model_state = model.state_dict()
    new_state = {}
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    print(f"Loaded {len(new_state)} keys.")
    
    # 2. Discriminator (HiFi-GAN)
    discriminator = HiFiGANDiscriminator().to(device)
    
    # 3. Helpers (Encoder, Quantizers) - Needed to create tokens
    factorizer = InformationFactorizerV2(config).to(device)
    
    fsq_levels = config['model']['fsq_levels']
    rfsq_num_levels = config['model']['rfsq_num_levels']
    sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device)

    # Load Helper Weights from same checkpoint folder if possible
    ckpt_dir = os.path.dirname(args.checkpoint)
    
    def load_helper(name, obj):
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location=device)
            # handle keys
            new_d = {}
            for k,v in d.items(): 
                new_d[k.replace("_orig_mod.", "")] = v
            obj.load_state_dict(new_d)
            print(f"Loaded {name}")
        else:
            print(f"WARNING: {name} not found! Training will be garbage if encoder is random.")
            
    load_helper("factorizer", factorizer)
    load_helper("sem_rfsq", sem_vq)
    load_helper("pro_rfsq", pro_vq)
    load_helper("spk_pq", spk_pq)
    
    # Freeze Helpers + Reconstructor (Train ONLY BitVocoder)
    print("Freezing everything except BitVocoder for finetuning...")
    factorizer.eval().requires_grad_(False)
    sem_vq.eval().requires_grad_(False)
    pro_vq.eval().requires_grad_(False)
    spk_pq.eval().requires_grad_(False)
    model.reconstructor.eval().requires_grad_(False)
    
    # Optimizers
    opt_g = torch.optim.AdamW(model.vocoder.parameters(), lr=args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr, betas=(0.8, 0.99))
    
    # Perceptual Loss Helpers
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    # HuBERT for Feature Loss
    print("Loading HuBERT for Perceptual Feedback...")
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    # Dataset
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    global_step = 0
    
    print("Starte Direct BitNet Training!")
    for epoch in range(args.epochs):
        model.vocoder.train() # Make sure Vocoder is training
        discriminator.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for wav in pbar:
            wav = wav.to(device)
            wav = wav.unsqueeze(1) # (B, 1, T)
            
            # --- 1. Prepare Tokens (No Grad) ---
            with torch.no_grad():
                # HuBERT features for Encoder
                # Must be very careful with shapes/SR
                wav_np = wav.squeeze(1).cpu().numpy()
                inputs = hubert_processor(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
                hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
                features = hubert_out.hidden_states[9] # Layer 9
                
                # Encode & Quantize
                sem, pro, spk = factorizer(features)
                sem_z, _, _ = sem_vq(sem)
                pro_z, _, _ = pro_vq(pro)
                spk_z, _, _ = spk_pq(spk)
                
                # Feature Reconstruction (Decoder Backbone)
                # This gives us the "Features" that enter the Vocoder
                vocoder_input_features = model.reconstructor(sem_z, pro_z, spk_z)
            
            # --- 2. Train Discriminator ---
            if epoch >= 2:
                # Forward Generator (Detach)
                wav_fake = model.vocoder(vocoder_input_features) # (B, T_fake)
                # Match lengths
                min_len = min(wav.shape[2], wav_fake.shape[1])
                wav_real_crop = wav[:, :, :min_len]
                wav_fake_crop = wav_fake[:, :min_len].unsqueeze(1) # (B, 1, T)
                
                # Disc Forward
                mpd_real, mrd_real = discriminator(wav_real_crop, wav_real_crop) 
                mpd_fake, mrd_fake = discriminator(wav_real_crop, wav_fake_crop.detach())
                
                # Loss
                loss_d_mpd, _, _ = discriminator_loss(mpd_real[0], mpd_fake[0])
                loss_d_mrd, _, _ = discriminator_loss(mrd_real[1], mrd_fake[1]) 
                loss_d = loss_d_mpd + loss_d_mrd
                
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
            else:
                # Needed for G forward pass logic even if D off? 
                # Actually, G needs wav_fake for STFT. So we must generate it.
                wav_fake = model.vocoder(vocoder_input_features)
                min_len = min(wav.shape[2], wav_fake.shape[1])
                wav_real_crop = wav[:, :, :min_len]
                wav_fake_crop = wav_fake[:, :min_len].unsqueeze(1)
                
                loss_d = torch.tensor(0.0, device=device)
            
            # --- 3. Train Generator ---
            # --- 3. Train Generator ---
            # GAN Warmup: Only use STFT loss for first few epochs to stabilize BitVocoder
            if epoch < 2:
                loss_adv = torch.tensor(0.0, device=device)
                loss_fm = torch.tensor(0.0, device=device)
                stft_weight = 45.0 # Strong guidance
            else:
                # Re-run Disc (Backprop to G)
                mpd_res, mrd_res = discriminator(wav_real_crop, wav_fake_crop)
                
                # Advertarial Loss
                loss_g_mpd, _ = generator_loss(mpd_res[0])
                loss_g_mrd, _ = generator_loss(mrd_res[1])
                loss_adv = loss_g_mpd + loss_g_mrd
                
                # Feature Matching Loss (Disc Internals)
                loss_fm = feature_loss(mpd_res[2], mpd_res[3]) + feature_loss(mrd_res[2], mrd_res[3])
                stft_weight = 5.0 # Relaxed guidance to allow GAN texture
            
            # Mel Loss (Auxiliary)
            sc_loss, mag_loss = stft_loss(wav_fake_crop.squeeze(1), wav_real_crop.squeeze(1))
            loss_stft = (sc_loss + mag_loss) * stft_weight
            
            # HuBERT Feature Loss (Perceptual Content Consistency)
            # Verify generated audio content matches input
            # We run HuBERT on generated audio
            # inputs_fake = hubert_processor(wav_fake_crop.squeeze(1).detach().cpu().numpy().flatten(), sampling_rate=16000, return_tensors="pt", padding=True)
            # This is expensive inside loop... maybe do it every n steps?
            # Or skip for maximum speed now.
            # Let's skip explicit HuBERT loss for now to save VRAM/Speed and rely on FM Loss.
            
            loss_g_total = loss_adv + loss_fm + loss_stft
            
            opt_g.zero_grad()
            loss_g_total.backward()
            opt_g.step()
            
            # Logs
            if global_step % 20 == 0:
                pbar.set_postfix({'D': f"{loss_d.item():.3f}", 'G': f"{loss_g_total.item():.3f}", 'STFT': f"{loss_stft.item():.3f}"})
                writer.add_scalar("loss/d_total", loss_d.item(), global_step)
                writer.add_scalar("loss/g_adv", loss_adv.item(), global_step)
                writer.add_scalar("loss/g_fm", loss_fm.item(), global_step)
                writer.add_scalar("loss/g_stft", loss_stft.item(), global_step)
                
            if global_step % 200 == 0: # Visualize often
                img_path = os.path.join(args.output_dir, "spectrograms", f"step_{global_step}.png")
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                plot_spectrogram(wav_real_crop[0, 0].detach(), wav_fake_crop[0, 0].detach(), img_path)
                
            global_step += 1
        
        # Save
        if epoch % 1 == 0:
            torch.save(model.vocoder.state_dict(), os.path.join(args.output_dir, f"bitvocoder_epoch{epoch}.pt"))
            torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f"disc_epoch{epoch}.pt"))
            
if __name__ == "__main__":
    main()
