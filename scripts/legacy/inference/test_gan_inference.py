import torch
import torchaudio
import yaml
import os
import sys
import soundfile as sf
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder

def load_clean_state_dict(model, path, device, exclude_prefix=None, require_prefix=None):
    print(f"LOADING FILE: {path}")
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    debug_count = 0
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            key = k[10:]
        else:
            key = k
            
        if debug_count < 5 and require_prefix:
             print(f"[{path}] DEBUG: Key '{k}' -> '{key}' (Prefix: '{require_prefix}')")
             debug_count += 1
            
        if exclude_prefix and key.startswith(exclude_prefix):
            continue
            
        if require_prefix:
            if not key.startswith(require_prefix):
                continue
            key = key[len(require_prefix):] # Strip prefix
            
        new_state_dict[key] = v
        
    # Check if we loaded anything
    if len(new_state_dict) == 0:
        print(f"WARNING: No keys loaded for {path} with require_prefix={require_prefix}")
        
    model.load_state_dict(new_state_dict, strict=False) 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Configs
    config_path = "ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    ckpt_stable_dir = "checkpoints/checkpoints_stable/step_87000"
    ckpt_gan_path = "checkpoints/checkpoints_ultra200bps_gan/bitvocoder_epoch2.pt"
    input_wav = "data/audio/1246_1246_135815_000001_000000.wav"
    output_dir = "gan_test_results"
    
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Load Foundation Models (Step 87k)
    print("Loading Foundation Models (Step 87k)...")
    factorizer = InformationFactorizerV2(config).to(device).eval()
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device).eval()
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device).eval()
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device).eval()
    
    # Load original decoder purely to get the backbone weights
    decoder_base = SpeechDecoderV2(config).to(device).eval()

    load_clean_state_dict(factorizer, os.path.join(ckpt_stable_dir, "factorizer.pt"), device)
    load_clean_state_dict(sem_vq, os.path.join(ckpt_stable_dir, "sem_rfsq.pt"), device)
    load_clean_state_dict(pro_vq, os.path.join(ckpt_stable_dir, "pro_rfsq.pt"), device)
    load_clean_state_dict(spk_pq, os.path.join(ckpt_stable_dir, "spk_pq.pt"), device)
    # Exclude 'vocoder.' keys as that's the part we're replacing and it mismatches
    load_clean_state_dict(decoder_base, os.path.join(ckpt_stable_dir, "decoder.pt"), device, exclude_prefix="vocoder.")

    # 2. Load BitVocoder (Epoch 2)
    print("Loading BitVocoder (Epoch 2)...")
    bit_vocoder = BitVocoder(
        input_dim=config['model']['decoder']['fusion_dim'],
        dim=256,
        n_fft=1024,
        hop_length=320,
        num_layers=4,
        num_res_blocks=1
    ).to(device).eval()
    
    # Load ONLY vocoder keys from the GAN checkpoint, and strip the 'vocoder.' prefix
    load_clean_state_dict(bit_vocoder, ckpt_gan_path, device, require_prefix="vocoder.model.")

    # 3. Surgical Transplant
    print("Transplanting BitVocoder into SpeechDecoder...")
    # SpeechDecoderV2 has self.vocoder (NeuralVocoderV2). We replace it.
    decoder_base.vocoder = bit_vocoder

    
    # We also need to make sure the forward pass works. 
    # SpeechDecoderV2.forward calls self.backbone then self.decoder.
    # BitVocoder forward takes (x, g=None). SimpleDecoderV2 took (x).
    # We might need to wrap it if signatures differ, but BitVocoder was designed to fit.
    # Let's check signatures. 
    # SimpleDecoderV2: forward(x)
    # BitVocoder: forward(x, g=None) -> should be compatible if called with 1 arg.

    # 4. Load Audio & HuBERT
    print("Loading Audio & HuBERT...")
    audio, sr = sf.read(input_wav)
    audio = torch.from_numpy(audio).float().to(device)
    if audio.dim() == 1: audio = audio.unsqueeze(0)
    if sr != 16000: audio = torchaudio.functional.resample(audio, sr, 16000)
    
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()

    with torch.no_grad():
        inputs = hubert_processor(audio.cpu().squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        outputs = hubert_model(inputs.input_values.to(device), output_hidden_states=True)
        features = outputs.hidden_states[9] 

        # 5. Run Pipeline
        print("Running Inference...")
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)

        # Decode (uses modified decoder with BitVocoder)
        audio_hat = decoder_base(sem_z, pro_z, spk_z)

    # 6. Save
    output_path = os.path.join(output_dir, "test_output_epoch2.wav")
    sf.write(output_path, audio_hat.cpu().squeeze().numpy(), 16000)
    print(f"Saved to {output_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Original")
    plt.specgram(audio.cpu().squeeze().numpy(), Fs=16000)
    plt.subplot(2, 1, 2)
    plt.title("Generated (BitVocoder Epoch 2)")
    plt.specgram(audio_hat.cpu().squeeze().numpy(), Fs=16000)
    plt.savefig(os.path.join(output_dir, "spectrogram_cmp.png"))
    print("Saved spectrogram comparison.")

if __name__ == "__main__":
    main()
