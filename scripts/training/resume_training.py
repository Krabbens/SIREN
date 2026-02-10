#!/usr/bin/env python3
"""
Resume training from checkpoint - with VALIDATION on unseen data
"""
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import yaml
import sys
import os
import random

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ultra_low_bitrate_codec'))

from models.encoder import InformationFactorizerV2 as InformationFactorizer
from models.quantizers import ResidualFSQ, ProductQuantizer
from models.decoder import SpeechDecoderV2 as SpeechDecoder
from models.entropy_coding import EntropyModel
from models.discriminator import HiFiGANDiscriminator
from data.feature_dataset import PrecomputedFeatureDataset
from training.losses import MultiResolutionSTFTLoss, feature_matching_loss, discriminator_loss, generator_loss
import torchaudio
import matplotlib.pyplot as plt
from transformers import HubertModel, Wav2Vec2FeatureExtractor

CONFIG_PATH = os.path.join(PROJECT_ROOT, "ultra_low_bitrate_codec/configs/improved_ljspeech.yaml")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/checkpoints_v2")
RESUME_STEP = 16500  # Najnowszy checkpoint V2

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

print("="*70)
print(f"🚀 WZNOWIENIE TRENINGU OD STEP {RESUME_STEP}")
print("   Z WALIDACJĄ NA NIEWIDZIANYCH DANYCH")
print("="*70)

# ============================================================================
# ZNAJDŹ PLIKI TESTOWE (spoza training set)
# ============================================================================
features_dir = os.path.join(PROJECT_ROOT, "data/features_train")
wav_dir = os.path.join(PROJECT_ROOT, "data/LJSpeech-1.1/wavs")

training_files = set(f.replace('.pt', '') for f in os.listdir(features_dir) if f.endswith('.pt'))
all_wavs = set(f.replace('.wav', '') for f in os.listdir(wav_dir) if f.endswith('.wav'))
test_files = sorted(list(all_wavs - training_files))

print(f"📊 Pliki testowe (spoza training set): {len(test_files)}")

# ============================================================================
# MODELE
# ============================================================================
factorizer = InformationFactorizer(config).to(device)
levels = config['model']['fsq_levels']
num_levels = config['model'].get('rfsq_num_levels', 1)

sem_vq = ResidualFSQ(levels, num_levels=num_levels).to(device)
pro_vq = ResidualFSQ(levels, num_levels=num_levels).to(device)
spk_pq = ProductQuantizer(256, 8, 256).to(device)
decoder = SpeechDecoder(config).to(device)
entropy_model = EntropyModel(config).to(device)
discriminator = HiFiGANDiscriminator().to(device)

# Załaduj checkpointy
factorizer.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/factorizer_{RESUME_STEP}.pt", map_location=device))
sem_vq.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/sem_rfsq_{RESUME_STEP}.pt", map_location=device))
pro_vq.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/pro_rfsq_{RESUME_STEP}.pt", map_location=device))
spk_pq.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/spk_pq_{RESUME_STEP}.pt", map_location=device))
decoder.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/decoder_{RESUME_STEP}.pt", map_location=device))
discriminator.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/discriminator_{RESUME_STEP}.pt", map_location=device))

print("✅ Wszystkie checkpointy załadowane")

# HuBERT dla walidacji
hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
hubert_model.eval()
print("✅ HuBERT załadowany dla walidacji")

# ============================================================================
# OPTYMIZATORY
# ============================================================================
params = list(factorizer.parameters()) + \
         list(sem_vq.parameters()) + \
         list(pro_vq.parameters()) + \
         list(spk_pq.parameters()) + \
         list(decoder.parameters()) + \
         list(entropy_model.parameters())

optimizer = optim.AdamW(params, lr=float(config['training']['learning_rate']), betas=(0.8, 0.99))
optimizer_d = optim.AdamW(discriminator.parameters(), lr=float(config['training']['learning_rate']), betas=(0.8, 0.99))

max_steps = 50000

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

scheduler = get_linear_schedule_with_warmup(optimizer, config['training']['warmup_steps'], max_steps)
scheduler_d = get_linear_schedule_with_warmup(optimizer_d, config['training']['warmup_steps'], max_steps)

for _ in range(RESUME_STEP):
    scheduler.step()
    scheduler_d.step()

scaler = torch.amp.GradScaler('cuda')

# ============================================================================
# DATA
# ============================================================================
train_ds = PrecomputedFeatureDataset(
    feature_dir=os.path.join(PROJECT_ROOT, "data/features_train"),
    manifest_path=config['data']['train_manifest'],
    max_frames=100
)
train_dl = DataLoader(
    train_ds, 
    batch_size=config['training']['batch_size'],
    shuffle=True, 
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

# Losses
mr_stft = MultiResolutionSTFTLoss().to(device)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256, win_length=1024
).to(device)

def mel_fn(y):
    spec = mel_transform(y)
    return torch.log(torch.clamp(spec, min=1e-5))

os.makedirs("spectrograms", exist_ok=True)
os.makedirs("spectrograms_val", exist_ok=True)

# ============================================================================
# FUNKCJA WALIDACJI NA NIEWIDZIANYCH DANYCH
# ============================================================================
def validate_on_unseen(step):
    """Testuj na losowym pliku spoza training set"""
    factorizer.eval()
    decoder.eval()
    
    # Wybierz losowy plik testowy
    test_file = random.choice(test_files)
    wav_path = os.path.join(wav_dir, f"{test_file}.wav")
    
    # Załaduj audio
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    
    # Ogranicz do 2 sekund
    waveform = waveform[:, :32000]
    
    with torch.no_grad():
        # Ekstrahuj HuBERT features
        inputs = hubert_processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = hubert_model(**inputs, output_hidden_states=True)
        features = outputs.hidden_states[9]
        
        # Forward przez model
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        audio_hat = decoder(sem_z, pro_z, spk_z)
        if audio_hat.dim() == 2:
            audio_hat = audio_hat.unsqueeze(1)
        
        # Wizualizacja
        orig = waveform[0].cpu()
        recon = audio_hat[0, 0, :orig.shape[0]].cpu()
        
        orig_spec = torch.log(torch.clamp(torch.abs(torch.stft(orig, n_fft=1024, return_complex=True)), min=1e-5)).numpy()
        recon_spec = torch.log(torch.clamp(torch.abs(torch.stft(recon, n_fft=1024, return_complex=True)), min=1e-5)).numpy()
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(orig_spec, origin='lower', aspect='auto', vmin=-10, vmax=5)
        plt.title(f"VALIDATION (niewidziane) - Original: {test_file}")
        plt.subplot(2, 1, 2)
        plt.imshow(recon_spec, origin='lower', aspect='auto', vmin=-10, vmax=5)
        plt.title(f"Reconstructed (Step {step})")
        plt.tight_layout()
        plt.savefig(f"spectrograms_val/step_{step}_val.png")
        plt.close()
        
        torchaudio.save(f"spectrograms_val/step_{step}_orig.wav", orig.unsqueeze(0), 16000)
        torchaudio.save(f"spectrograms_val/step_{step}_recon.wav", recon.unsqueeze(0), 16000)
    
    factorizer.train()
    decoder.train()
    
    return test_file

print(f"📊 Dataset: {len(train_ds)} samples")
print(f"📊 Max steps: {max_steps}")
print(f"📊 Resuming from step: {RESUME_STEP}")

# ============================================================================
# TRAINING LOOP
# ============================================================================
steps = RESUME_STEP
pbar = tqdm(total=max_steps - RESUME_STEP, desc=f"Training from {RESUME_STEP}")

log_file = open(os.path.join(PROJECT_ROOT, "training_resumed.log"), "a")
log_file.write(f"\n=== Resumed from step {RESUME_STEP} with validation ===\n")
log_file.flush()

while steps < max_steps:
    for batch in train_dl:
        features, audio = batch
        features = features.to(device)
        audio = audio.to(device)
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        with torch.amp.autocast('cuda'):
            sem, pro, spk = factorizer(features)
            sem_z, sem_loss, sem_idx = sem_vq(sem)
            pro_z, pro_loss, pro_idx = pro_vq(pro)
            spk_z, spk_loss, spk_idx = spk_pq(spk)
            vq_loss = sem_loss + pro_loss + spk_loss
            
            audio_hat = decoder(sem_z, pro_z, spk_z)
            if audio_hat.dim() == 2:
                audio_hat = audio_hat.unsqueeze(1)
            
            min_len = min(audio.shape[2], audio_hat.shape[2])
            audio = audio[..., :min_len]
            audio_hat = audio_hat[..., :min_len]
        
        # Discriminator
        optimizer_d.zero_grad()
        with torch.amp.autocast('cuda'):
            mpd_res_d, mrd_res_d = discriminator(audio, audio_hat.detach())
            loss_d_mpd, _, _ = discriminator_loss(mpd_res_d[0], mpd_res_d[1])
            loss_d_mrd, _, _ = discriminator_loss(mrd_res_d[0], mrd_res_d[1])
            loss_d = loss_d_mpd + loss_d_mrd
        
        scaler.scale(loss_d).backward()
        scaler.step(optimizer_d)
        
        # Generator
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            mpd_res, mrd_res = discriminator(audio, audio_hat)
            loss_g_mpd, _ = generator_loss(mpd_res[1])
            loss_g_mrd, _ = generator_loss(mrd_res[1])
            loss_gen_gan = loss_g_mpd + loss_g_mrd
            
            loss_fm_mpd = feature_matching_loss(mpd_res[2], mpd_res[3])
            loss_fm_mrd = feature_matching_loss(mrd_res[2], mrd_res[3])
            loss_fm = loss_fm_mpd + loss_fm_mrd
            
            orig_spec = mel_fn(audio.squeeze(1))
            recon_spec = mel_fn(audio_hat.squeeze(1))
            mel_loss = F.l1_loss(recon_spec, orig_spec) * 45.0
            
            sc_loss, mag_loss = mr_stft(audio.squeeze(1), audio_hat.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * 1.0
            
            sem_bits, pro_bits = entropy_model.estimate_bits(sem_idx, pro_idx)
            entropy_loss = (sem_bits.mean() + pro_bits.mean()) * 0.01
            
            loss_g = vq_loss + entropy_loss + mel_loss + loss_gen_gan + loss_fm + stft_loss
            
            total_bits = sem_bits.sum() + pro_bits.sum() + 64
            duration = min_len / 16000.0
            bps = total_bits / (duration * features.shape[0] + 1e-6)
        
        scaler.scale(loss_g).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        scheduler_d.step()
        
        steps += 1
        pbar.update(1)
        
        metrics = {"g": f"{loss_g.item():.2f}", "d": f"{loss_d.item():.2f}", "mel": f"{mel_loss.item():.1f}", "bps": f"{bps.item():.0f}"}
        pbar.set_postfix(metrics)
        
        if steps % 100 == 0:
            log_line = f"Step {steps}: loss_g={loss_g.item():.3f}, mel={mel_loss.item():.3f}, bps={bps.item():.1f}\n"
            log_file.write(log_line)
            log_file.flush()
            print(log_line.strip())
        
        # Checkpoints + WALIDACJA
        if steps % 200 == 0:
            torch.save(factorizer.state_dict(), f"{CHECKPOINT_DIR}/factorizer_{steps}.pt")
            torch.save(decoder.state_dict(), f"{CHECKPOINT_DIR}/decoder_{steps}.pt")
            torch.save(spk_pq.state_dict(), f"{CHECKPOINT_DIR}/spk_pq_{steps}.pt")
            torch.save(discriminator.state_dict(), f"{CHECKPOINT_DIR}/discriminator_{steps}.pt")
            torch.save(sem_vq.state_dict(), f"{CHECKPOINT_DIR}/sem_rfsq_{steps}.pt")
            torch.save(pro_vq.state_dict(), f"{CHECKPOINT_DIR}/pro_rfsq_{steps}.pt")
            
            # Training set comparison
            with torch.no_grad():
                orig_viz = audio[0, 0].cpu()
                recon_viz = audio_hat[0, 0, :orig_viz.shape[0]].cpu()
                orig_s = torch.log(torch.clamp(torch.abs(torch.stft(orig_viz, n_fft=1024, return_complex=True)), min=1e-5)).numpy()
                recon_s = torch.log(torch.clamp(torch.abs(torch.stft(recon_viz, n_fft=1024, return_complex=True)), min=1e-5)).numpy()
                
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.imshow(orig_s, origin='lower', aspect='auto', vmin=-10, vmax=5)
                plt.title(f"TRAINING DATA - Original (Step {steps})")
                plt.subplot(2, 1, 2)
                plt.imshow(recon_s, origin='lower', aspect='auto', vmin=-10, vmax=5)
                plt.title("Reconstructed")
                plt.tight_layout()
                plt.savefig(f"spectrograms/step_{steps}_comparison.png")
                plt.close()
            
            # WALIDACJA na niewidzianych danych
            val_file = validate_on_unseen(steps)
            print(f"  📊 Validation on unseen: {val_file}")
        
        if steps >= max_steps:
            break

pbar.close()
log_file.close()
print("\n✅ Training complete!")
