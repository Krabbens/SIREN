
import os
import sys
import argparse
import yaml
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from transformers import Wav2Vec2FeatureExtractor, HubertModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_spectrogram(audio, path, title):
    # audio: (T,)
    spec = torch.stft(
        audio, n_fft=1024, hop_length=320, 
        win_length=1024, window=torch.hann_window(1024).to(audio.device),
        return_complex=True
    ).abs()
    spec = torch.log(spec + 1e-5)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.cpu().numpy(), aspect='auto', origin='lower', vmin=-11.5, vmax=2.0)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default="jakubie_16k.wav")
    parser.add_argument("--output", default="intermediate_epoch2.png")
    parser.add_argument("--config", default="ultra_low_bitrate_codec/configs/ultra200bps_large.yaml")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    # 1. Load Vocoder (Fine-tuned)
    print(f"Loading finetuned decoder from {args.checkpoint}...")
    model = SpeechDecoderV2(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict) # Should be strict compatible now
    model.eval()
    
    # 2. Load Encoder/Quantizers (Original)
    # We need these to get the codes to feed the decoder
    print("Loading original encoder components...")
    
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    
    factorizer = InformationFactorizerV2(config).to(device).eval()
    
    fsq_levels = config['model']['fsq_levels']
    rfsq_num_levels = config['model']['rfsq_num_levels']
    sem_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['semantic']['output_dim']).to(device).eval()
    pro_vq = ResidualFSQ(levels=fsq_levels, num_levels=rfsq_num_levels, input_dim=config['model']['prosody']['output_dim']).to(device).eval()
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device).eval()
    
    # Load weights for encoder parts (assuming they are in checkpoints_ultra200bps_large/step_14500)
    orig_ckpt_dir = "checkpoints_ultra200bps_large/step_14500"
    
    def load_part(obj, name):
        p = os.path.join(orig_ckpt_dir, f"{name}.pt")
        d = torch.load(p, map_location=device)
        new_d = {k.replace("_orig_mod.", ""): v for k, v in d.items()}
        obj.load_state_dict(new_d)
        
    load_part(factorizer, "factorizer")
    load_part(sem_vq, "sem_rfsq")
    load_part(pro_vq, "pro_rfsq")
    load_part(spk_pq, "spk_pq")

    # 3. Process
    print(f"Processing {args.input}...")
    wav, sr = sf.read(args.input)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    
    with torch.no_grad():
        inputs = hubert_processor(wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        hubert_out = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        features = hubert_out.hidden_states[config['model'].get('hubert_layer', 9)]
        
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        # Decode with finetuned vocoder
        rec_wav = model(sem_z, pro_z, spk_z)
        
        # Save Audio
        rec_wav = rec_wav.squeeze().cpu()
        out_wav = args.output.replace(".png", ".wav")
        sf.write(out_wav, rec_wav.numpy(), 16000)
        print(f"Saved audio to {out_wav}")
        
        # Save Spectrogram
        save_spectrogram(rec_wav.to(device), args.output, f"Epoch 2 Finetune (Phase Head)")
        print(f"Saved spectrogram to {args.output}")

if __name__ == "__main__":
    main()
