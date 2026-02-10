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
    
    # Debug info
    keys = list(state_dict.keys())
    print(f"DEBUG: Found {len(keys)} keys in {os.path.basename(path)}")
    if len(keys) > 0:
        print(f"DEBUG: First key: {keys[0]}")
        
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            key = k[10:]
        else:
            key = k
            
        if exclude_prefix and key.startswith(exclude_prefix):
            continue
            
        if require_prefix:
            if not key.startswith(require_prefix):
                continue
            key = key[len(require_prefix):] # Strip prefix
            
        new_state_dict[key] = v
        
    if len(new_state_dict) == 0:
        print(f"WARNING: No keys loaded for {path} with require_prefix={require_prefix}")
        
    model.load_state_dict(new_state_dict, strict=False) 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Configs
    config_path = "ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    ckpt_stable_dir = "checkpoints/checkpoints_stable/step_87000"
    
    # Find latest checkpoint
    gan_dir = "checkpoints/checkpoints_ultra200bps_gan"
    ckpts = [f for f in os.listdir(gan_dir) if f.startswith("bitvocoder_epoch") and f.endswith(".pt")]
    if not ckpts:
        print("No checkpoints found!")
        return
        
    # Sort by epoch
    ckpts.sort(key=lambda x: int(x.replace("bitvocoder_epoch", "").replace(".pt", "")))
    latest_ckpt = ckpts[-1]
    ckpt_gan_path = os.path.join(gan_dir, latest_ckpt)
    print(f"Selected Checkpoint: {latest_ckpt}")

    input_wav = "data/audio/1246_1246_135815_000001_000000.wav" # Ensure this exists
    if not os.path.exists(input_wav): # Fallback
        # Find any wav
        import glob
        wavs = glob.glob("data/audio/**/*.wav", recursive=True)
        if wavs:
            input_wav = wavs[0]
            print(f"Fallback input: {input_wav}")
            
    output_dir = "gan_test_results_epoch13"
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Load Foundation Models
    print("Loading Foundation Models...")
    factorizer = InformationFactorizerV2(config).to(device).eval()
    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['semantic']['output_dim']).to(device).eval()
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=config['model']['prosody']['output_dim']).to(device).eval()
    spk_pq = ProductQuantizer(input_dim=config['model']['speaker']['embedding_dim'], num_groups=config['model']['speaker']['num_groups'], codes_per_group=config['model']['speaker']['codes_per_group']).to(device).eval()
    
    decoder_base = SpeechDecoderV2(config).to(device).eval()

    load_clean_state_dict(factorizer, os.path.join(ckpt_stable_dir, "factorizer.pt"), device)
    load_clean_state_dict(sem_vq, os.path.join(ckpt_stable_dir, "sem_rfsq.pt"), device)
    load_clean_state_dict(pro_vq, os.path.join(ckpt_stable_dir, "pro_rfsq.pt"), device)
    load_clean_state_dict(spk_pq, os.path.join(ckpt_stable_dir, "spk_pq.pt"), device)
    
    # Exclude vocoder from base decoder
    load_clean_state_dict(decoder_base, os.path.join(ckpt_stable_dir, "decoder.pt"), device, exclude_prefix="vocoder.")

    # 2. Load BitVocoder (Latest)
    print(f"Loading {latest_ckpt}...")
    bit_vocoder = BitVocoder(
        input_dim=config['model']['decoder']['fusion_dim'],
        dim=256,
        n_fft=1024,
        hop_length=320,
        num_layers=4,
        num_res_blocks=1
    ).to(device).eval()
    
    # Based on previous debugging, we know keys are likely under 'vocoder.model.' or just direct if saved from module.
    # Training script saves: torch.save(model.vocoder.state_dict(), ...)
    # So keys should be 'conv_in.weight', etc. (NO PREFIX)
    # But wait, earlier I had to use 'vocoder.model.'? No, earlier I was loading from a FULL decoder checkpoint.
    # Now I am loading specific 'bitvocoder_epochX.pt' saved by `torch.save(model.vocoder.state_dict())`.
    # So it should be NO prefix.
    
    # Try loading without prefix first.
    try:
        load_clean_state_dict(bit_vocoder, ckpt_gan_path, device)
    except Exception as e:
        print(f"Direct load failed: {e}. Trying with prefix...")
        # If that fails (maybe wrapped?), try stripping
        pass

    # 3. Surgical Transplant
    print("Transplanting BitVocoder into SpeechDecoder...")
    decoder_base.vocoder = bit_vocoder # Hotswap!

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
    output_path = os.path.join(output_dir, f"test_output_{latest_ckpt.replace('.pt', '')}.wav")
    sf.write(output_path, audio_hat.cpu().squeeze().numpy(), 16000)
    print(f"Saved to {output_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Original")
    plt.specgram(audio.cpu().squeeze().numpy(), Fs=16000)
    plt.subplot(2, 1, 2)
    plt.title(f"Generated ({latest_ckpt})")
    plt.specgram(audio_hat.cpu().squeeze().numpy(), Fs=16000)
    plt.savefig(os.path.join(output_dir, "spectrogram_cmp.png"))
    print("Saved spectrogram comparison.")

if __name__ == "__main__":
    main()
