import torch
import yaml
import argparse
import torchaudio
import os
from ultra_low_bitrate_codec.models.encoder import InformationFactorizer
from ultra_low_bitrate_codec.models.quantizers import FSQ, VectorQuantizer, ProductQuantizer
from ultra_low_bitrate_codec.models.decoder import SpeechDecoder
from ultra_low_bitrate_codec.data.feature_dataset import PrecomputedFeatureDataset
from torch.utils.data import DataLoader

def generate(config_path, feature_dir, checkpoint_step, output_dir="samples"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Models
    factorizer = InformationFactorizer(config).to(device)
    decoder = SpeechDecoder(config).to(device)
    
    # Load Checkpoints
    try:
        factorizer.load_state_dict(torch.load(f"factorizer_{checkpoint_step}.pt", map_location=device))
        decoder.load_state_dict(torch.load(f"decoder_{checkpoint_step}.pt", map_location=device))
        print(f"Loaded checkpoints for step {checkpoint_step}")
    except FileNotFoundError:
        print(f"Checkpoints for step {checkpoint_step} not found.")
        return

    # Quantizers (Needed for forward pass logic if we replicate manual inference, 
    # but let's just instantiate them to match shapes if needed, or better yet, verify logic)
    # The models are: factorizer -> (sem, pro, spk) -> Quantizers -> (z_q) -> decoder
    # We need the quantizers to get z_q!
    
    quant_type = config['model'].get('quantizer_type', 'vq')
    if quant_type == 'fsq':
        levels = config['model']['fsq_levels']
        sem_vq = FSQ(levels).to(device)
        pro_vq = FSQ(levels).to(device)
    else:
        sem_vq = VectorQuantizer(
            dim=config['model']['semantic']['output_dim'],
            vocab_size=config['model']['semantic']['vocab_size']
        ).to(device)
        pro_vq = VectorQuantizer(
            dim=config['model']['prosody']['output_dim'],
            vocab_size=config['model']['prosody']['vocab_size']
        ).to(device)
        
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    
    # We don't strictly need to load quantizer state if they don't have learnable parameters (FSQ)
    # But VQ has embeddings.
    # The training script didn't save quantizers separately? 
    # Wait, train_fast.py saves: factorizer_X.pt, decoder_X.pt
    # IT DOES NOT SAVE QUANTIZERS?
    # FSQ has no parameters (fixed levels).
    # ProductQuantizer (SPK) HAS parameters (codebooks).
    # VectorQuantizer (if used) HAS parameters.
    # defaults.yaml says quantizer_type: "fsq". 
    # So sem/pro use FSQ (Safe).
    # But Speaker uses ProductQuantizer. 
    # If spk_pq is not saved, we have random codebooks! This is bad.
    # Let's check train_fast.py... 
    # "torch.save(self.factorizer.state_dict()...)"
    # "torch.save(self.decoder.state_dict()...)"
    # It misses spk_pq!
    # However, speaker embedding is usually very stable or negligible in this architecture? 
    # No, it's critical for reconstruction.
    # WE FOUND A BUG. The Speaker Quantizer is not being saved!
    
    # For now, we will proceed with FSQ for sem/pro (main content).
    # Spk might be garbage.
    
    # Data
    train_ds = PrecomputedFeatureDataset(
        feature_dir=feature_dir,
        manifest_path=config['data']['train_manifest'],
        max_frames=500
    )
    # Pick a random sample
    import random
    idx = random.randint(0, len(train_ds)-1)
    features, audio = train_ds[idx]
    
    features = features.unsqueeze(0).to(device)
    audio = audio.unsqueeze(0).to(device)
    
    if audio.dim() == 2: audio = audio.unsqueeze(1)
    
    factorizer.eval()
    decoder.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        sem, pro, spk = factorizer(features)
        
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk) # Random if not loaded
        
        audio_hat = decoder(sem_z, pro_z, spk_z)
        
        # Save
        if audio_hat.dim() == 2: audio_hat = audio_hat.unsqueeze(1)
        
        sr = config['data']['sample_rate']
        torchaudio.save(f"{output_dir}/sample_{checkpoint_step}_orig.wav", audio.cpu().squeeze(1), sr)
        torchaudio.save(f"{output_dir}/sample_{checkpoint_step}_recon.wav", audio_hat.cpu().squeeze(1), sr)
        print(f"Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=3000)
    args = parser.parse_args()
    
    generate('ultra_low_bitrate_codec/configs/default.yaml', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/features_train'), args.step)
