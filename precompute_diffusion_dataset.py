
import os
import sys
import soundfile as sf
import torch
import torchaudio
import glob
import argparse
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_clean_state_dict(model, path, device):
    try:
        state_dict = torch.load(path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading {path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/audio")
    parser.add_argument("--output_dir", default="data/diffusion_pairs_bitnet")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    parser.add_argument("--limit", type=int, default=1000, help="Number of files to process")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Config & Models
    config = load_config(args.config)
    
    print("Loading BitNet models...")
    factorizer = InformationFactorizerV2(config).to(device).eval()
    decoder = SpeechDecoderV2(config).to(device).eval()
    
    fsq_levels = config['model']['fsq_levels']
    rfsq_num_levels = config['model']['rfsq_num_levels']
    
    sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['semantic']['output_dim']).to(device).eval()
    pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['prosody']['output_dim']).to(device).eval()
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device).eval()

    # Load Weights
    ckpt_dir = args.checkpoint_dir
    load_clean_state_dict(factorizer, os.path.join(ckpt_dir, "factorizer.pt"), device)
    load_clean_state_dict(decoder, os.path.join(ckpt_dir, "decoder.pt"), device)
    load_clean_state_dict(sem_vq, os.path.join(ckpt_dir, "sem_rfsq.pt"), device)
    load_clean_state_dict(pro_vq, os.path.join(ckpt_dir, "pro_rfsq.pt"), device)
    load_clean_state_dict(spk_pq, os.path.join(ckpt_dir, "spk_pq.pt"), device)
    
    # HuBERT
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    print("Loading HuBERT...")
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    # Mel Transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80, power=1.0
    ).to(device)
    
    # Prepare Output
    os.makedirs(os.path.join(args.output_dir, "clean"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "degraded"), exist_ok=True)
    
    # Process Files
    files = glob.glob(os.path.join(args.data_dir, "**", "*.wav"), recursive=True)
    files = [f for f in files if os.path.getsize(f) > 50000] # Min size
    
    print(f"Found {len(files)} files. Processing {args.limit}...")
    
    count = 0
    with torch.no_grad():
        for fpath in tqdm(files[:args.limit]):
            try:
                # Load Audio
                wav_np, sr = sf.read(fpath)
                if wav_np.ndim > 1:
                    wav_np = wav_np.mean(axis=1)
                
                wav = torch.from_numpy(wav_np).float().unsqueeze(0) # (1, T)
                
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                    
                wav = wav.to(device)
                
                # Make length multiple of 320 (hop size) and enough for hubert
                if wav.shape[1] < 16000: continue
                
                # 1. Get Clean Mel
                clean_mel = mel_transform(wav)
                clean_log_mel = torch.log(clean_mel + 1e-5)
                
                # 2. Run BitNet Inference
                # Extract HuBERT
                inputs = hubert_processor(wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
                hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
                features = hubert_out.hidden_states[config['model'].get('hubert_layer', 9)]
                
                # Encode/Quantize/Decode
                sem, pro, spk = factorizer(features)
                sem_z, _, _ = sem_vq(sem)
                pro_z, _, _ = pro_vq(pro)
                spk_z, _, _ = spk_pq(spk)
                rec_wav = decoder(sem_z, pro_z, spk_z)
                
                # 3. Get Degraded Mel
                # Ensure length match
                min_len_wav = min(wav.shape[1], rec_wav.shape[1])
                rec_wav = rec_wav[:, :min_len_wav]
                
                # Note: Decoder output might be slightly shifted or shorter due to upsampling/downsampling differences
                # But Mel transform hop size should align broadly.
                
                degraded_mel = mel_transform(rec_wav)
                degraded_log_mel = torch.log(degraded_mel + 1e-5)
                
                # Crop Mels to match
                min_frames = min(clean_log_mel.shape[-1], degraded_log_mel.shape[-1])
                clean_log_mel = clean_log_mel[..., :min_frames]
                degraded_log_mel = degraded_log_mel[..., :min_frames]
                
                # Save
                basename = os.path.splitext(os.path.basename(fpath))[0]
                torch.save(clean_log_mel.cpu(), os.path.join(args.output_dir, "clean", f"{basename}.pt"))
                torch.save(degraded_log_mel.cpu(), os.path.join(args.output_dir, "degraded", f"{basename}.pt"))
                
                count += 1
                if count >= args.limit:
                    break
                    
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                continue
                
    print(f"Saved {count} pairs to {args.output_dir}")

if __name__ == "__main__":
    main()
