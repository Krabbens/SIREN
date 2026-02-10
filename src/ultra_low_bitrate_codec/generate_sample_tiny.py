import torch
import torchaudio
import yaml
import os
import soundfile as sf
from ultra_low_bitrate_codec.models.tiny_hubert import TinyHubert
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer

def fix_state_dict_keys(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        # Remove torch.compile prefix if present
        k = k.replace("_orig_mod.", "")
        new_sd[k] = v
    return new_sd

def main():
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_tiny_adapt.yaml"
    ckpt_dir = "checkpoints/checkpoints_factorizer_tiny_frozen/step_46000"
    input_wav = "data/jakubie_16k.wav"
    output_wav = "checkpoints/checkpoints_factorizer_tiny_frozen/sample_step_46000.wav"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    print(f"Loading TinyHubert...")
    hubert = TinyHubert(out_dim=768, hidden_dim=384, num_layers=12).to(device)
    hubert.load_state_dict(torch.load("checkpoints/tiny_hubert_best.pt", map_location=device))
    hubert.eval()
    
    print("Loading Factorizer & Decoder...")
    factorizer = InformationFactorizerV2(config).to(device)
    decoder = SpeechDecoderV2(config).to(device)
    
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device)
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device)
    
    # Load weights
    factorizer.load_state_dict(fix_state_dict_keys(torch.load(f"{ckpt_dir}/factorizer.pt", map_location=device)))
    decoder.load_state_dict(fix_state_dict_keys(torch.load(f"{ckpt_dir}/decoder.pt", map_location=device)))
    sem_vq.load_state_dict(fix_state_dict_keys(torch.load(f"{ckpt_dir}/sem_rfsq.pt", map_location=device)))
    pro_vq.load_state_dict(fix_state_dict_keys(torch.load(f"{ckpt_dir}/pro_rfsq.pt", map_location=device)))
    spk_pq.load_state_dict(fix_state_dict_keys(torch.load(f"{ckpt_dir}/spk_pq.pt", map_location=device)))
    
    factorizer.eval()
    decoder.eval()
    
    print(f"Processing {input_wav}...")
    wav, sr = sf.read(input_wav)
    wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
        wav = resampler(wav)
        
    with torch.no_grad():
        # TinyHubert -> Factorizer -> Decoder
        features = hubert(wav)
        sem, pro, spk = factorizer(features)
        
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        audio_hat = decoder(sem_z, pro_z, spk_z)
        
    audio_hat = audio_hat.cpu().squeeze().numpy()
    sf.write(output_wav, audio_hat, 16000)
    print(f"Saved result to {output_wav}")

if __name__ == "__main__":
    main()
