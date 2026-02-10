
import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
import glob
import random
from tqdm import tqdm
import numpy as np

# Models
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.micro_transformer import MicroTransformer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class AudioDataset(Dataset):
    def __init__(self, data_dir, segment_length=16000*3): # 3s chunks
        self.files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
        # Filter tiny files
        self.files = [f for f in self.files if os.path.getsize(f) > 10000]
        self.segment_length = segment_length
        print(f"Found {len(self.files)} files")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        try:
            fname = self.files[idx]
            import soundfile as sf
            wav, sr = sf.read(fname)
            wav = torch.tensor(wav, dtype=torch.float32)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0) # (1, T)
            else:
                wav = wav.t() # (C, T)
                wav = wav.mean(0, keepdim=True) # Mix to mono
            
            if sr != 16000:
                import torchaudio.functional as AF
                wav = AF.resample(wav, sr, 16000)

            # Pad if too short
            if wav.shape[1] < self.segment_length:
                pad = self.segment_length - wav.shape[1]
                wav = F.pad(wav, (0, pad))
            
            # Random Crop
            if wav.shape[1] > self.segment_length:
                start = random.randint(0, wav.shape[1] - self.segment_length)
                wav = wav[:, start:start+self.segment_length]
                
            return wav.squeeze(0), os.path.basename(fname)
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(self.segment_length), "error.wav"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_ckpt", required=True, help="MicroTransformer Checkpoint (EP75)")
    parser.add_argument("--pipeline_ckpt_dir", required=True, help="Dir containing factorizer.pt, etc.")
    parser.add_argument("--data_dir", default="data/audio")
    parser.add_argument("--output_dir", default="data/flow_dataset_student")
    parser.add_argument("--config", default="src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    parser.add_argument("--num_samples", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = load_config(args.config)
    
    # STFT Params
    n_fft = 1024
    hop_length = 320
    win_length = 1024
    
    # 1. Load Student
    print("Loading Student (MicroTransformer)...")
    student = MicroTransformer(hidden_dim=384, num_layers=8).to(device)
    ckpt_s = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_s.items()}, strict=False)
    student.eval()
    
    # 2. Load Pipeline Models
    print("Loading Pipeline Models...")
    model = SpeechDecoderV2(config).to(device) # Contains reconstructor
    
    factorizer = InformationFactorizerV2(config).to(device)
    f_path = os.path.join(args.pipeline_ckpt_dir, "factorizer.pt")
    if not os.path.exists(f_path): f_path = os.path.join(args.pipeline_ckpt_dir, "factorizer_best_step.pt")
    ckpt_f = torch.load(f_path, map_location=device)
    if 'model_state_dict' in ckpt_f: ckpt_f = ckpt_f['model_state_dict']
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_f.items()}, strict=False)
    factorizer.eval()
    
    fsq_levels = config['model']['fsq_levels']
    rfsq_num_levels = config['model']['rfsq_num_levels']
    sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device)
    
    # Load Weights for Reconstructor (from full decoder checkpoint if available, or fuser?)
    # Wait, SpeechDecoderV2 contains `reconstructor`. The precompute_flow_dataset script loaded `decoder`.
    # Let's assume we have `decoder.pt` in pipeline dir or we used `factorizer_microhubert_finetune_v4`.
    # Wait, `precompute_flow_dataset.py` used `decoder` key. 
    # Let's check `SpeechDecoderV2` structure. It's essentially the Fuser logic? 
    # Actually, `ConditionFuserV2` IS the reconstructor logic usually.
    # In `precompute_flow_dataset.py` line 201: `cond = model.reconstructor(sem_z, pro_z, spk_z)`
    # So `SpeechDecoderV2` wraps the reconstructor.
    # Let's assume we can load it from `fuser.pt` if we map keys correctly, or just use `ConditionFuserV2` directly if we know the class capable of taking z and outputting cond.
    
    # SIMPLIFICATION: Use ConditionFuserV2 directly, as in inference scripts.
    from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fu_path = "checkpoints/checkpoints_flow_v2/fuser_epoch31.pt" # Use the ONE we want to finetune FROM (or the best one)
    if not os.path.exists(fu_path): fu_path = "checkpoints/checkpoints_flow_v2/fuser_epoch20.pt"
    print(f"Loading Fuser: {fu_path}")
    fu_ckpt = torch.load(fu_path, map_location=device)
    if 'model_state_dict' in fu_ckpt: fu_ckpt = fu_ckpt['model_state_dict']
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fu_ckpt.items()})
    fuser.eval()

    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    print("Starting Precomputation (Student)...")
    count = 0
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=80,
        center=True,
        power=2.0
    ).to(device)

    with torch.no_grad():
        for i, (wav, fname) in enumerate(tqdm(dataloader)):
            if count >= args.num_samples: break
            if fname[0] == "error.wav": continue
            
            wav = wav.to(device)
            
            # 1. Compute Log-Mel Spectrogram (Target)
            mel = mel_transform(wav) # (1, 80, T)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            target = mel
            
            # 2. Extract Student Features
            # Normalize for Student (Mean-Std)
            wav_s = (wav - wav.mean()) / (wav.std() + 1e-6)
            features = student(wav_s.unsqueeze(0)) # (1, T_feat, 768)
            
            # 3. Factorize
            sem, pro, spk = factorizer(features)
            sem_z, _, _ = sem_vq(sem)
            pro_z, _, _ = pro_vq(pro)
            spk_z, _, _ = spk_pq(spk)
            
            # 4. Fuse (Conditioning)
            target_mel_len = target.shape[2]
            cond = fuser(sem_z, pro_z, spk_z, target_mel_len)
            
            # 5. Save
            save_path = os.path.join(args.output_dir, f"{count:05d}.pt")
            torch.save({
                'mel': target.cpu().squeeze(0), # (80, T)
                'cond': cond.cpu().squeeze(0), # (512, T)
                # Save features optionally?
            }, save_path)
            
            count += 1
            
if __name__ == "__main__":
    main()
