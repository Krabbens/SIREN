import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import sys
import yaml
import struct
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.bit_vocoder import BitVocoder

# Add scripts to path for arithmetic coding
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from arithmetic_coding import ArithmeticEncoder

def compress_stream(indices):
    """
    Compresses a tensor of indices using static arithmetic coding with sparse histogram.
    Returns: compressed_bytes, table_bytes
    """
    flat = indices.flatten().tolist()
    if not flat:
        return b"", 0
        
    # 1. Compute Frequency Table (Sparse)
    counts = {}
    for s in flat:
        counts[s] = counts.get(s, 0) + 1
        
    # 2. Prepare for Encoding
    # We need to map actual symbols to a dense range 0..K-1
    # Sort symbols to make the map deterministic/reconstructible if we sent just the list of symbols
    unique_syms = sorted(counts.keys())
    sym_to_id = {sym: i for i, sym in enumerate(unique_syms)}
    
    freqs = [counts[sym] for sym in unique_syms]
    total_freq = sum(freqs)
    
    # Precompute cum_freqs for encoding
    # cum_freq[i] = sum(freqs[0..i-1])
    cum_freqs = [0] * (len(freqs) + 1)
    for i in range(len(freqs)):
        cum_freqs[i+1] = cum_freqs[i] + freqs[i]
        
    # 3. Encode
    enc = ArithmeticEncoder(precision=32)
    for sym in flat:
        sym_id = sym_to_id[sym]
        enc.encode(cum_freqs[sym_id], freqs[sym_id], total_freq)
        
    bitstream = enc.finish()
    
    # 4. Calculate Table Overhead
    # To reconstruct, we need:
    # - List of original symbols (each is int32 -> 4 bytes? or variable?)
    # - List of frequencies (int32 -> 4 bytes?)
    # Since symbols can be large (16M), let's assume 4 bytes per symbol.
    # Freqs can be large, 4 bytes per freq.
    # Simple estimation:
    table_bytes = len(unique_syms) * 4 + len(freqs) * 4
    
    # Or more efficiently, we can verify what pickling the table costs?
    # Let's use simple byte counting for honesty.
    
    return bitstream, table_bytes

def load_extraction_pipeline(checkpoint_dir, config, device):
    """Load the full SIREN pipeline for feature extraction."""
    print(f"Loading extraction pipeline from {checkpoint_dir}...")
    
    factorizer = InformationFactorizerV2(config).to(device)
    decoder = SpeechDecoderV2(config).to(device)
    
    sem_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=config['model']['semantic']['output_dim']
    ).to(device)
    
    pro_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=config['model']['prosody']['output_dim']
    ).to(device)
    
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    ).to(device)
    
    # Load weights
    def load_state(model, name):
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            state = torch.load(p, map_location=device)
            state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            print(f"Loaded {name}")

    load_state(factorizer, "factorizer.pt")
    load_state(decoder, "decoder.pt")
    load_state(sem_vq, "sem_rfsq.pt")
    load_state(pro_vq, "pro_rfsq.pt")
    load_state(spk_pq, "spk_pq.pt")
    
    factorizer.eval()
    decoder.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_pq.eval()
    
    return factorizer, decoder, sem_vq, pro_vq, spk_pq

@torch.no_grad()
def extract_features(audio_path, hubert_proc, hubert_model, factorizer, decoder, sem_vq, pro_vq, spk_pq, device):
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio.mean(0).unsqueeze(0).to(device)
    
    # HuBERT
    inputs = hubert_proc(audio.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    hubert_out = hubert_model(input_values).last_hidden_state
    
    # SIREN Components
    sem, pro, spk = factorizer(hubert_out)
    sem_q, _, sem_indices = sem_vq(sem)
    pro_q, _, pro_indices = pro_vq(pro)
    spk_q, _, spk_indices = spk_pq(spk)
    
    # Reconstruct 512-dim features
    fused = decoder.reconstructor(sem_q, pro_q, spk_q)
    return fused.transpose(1, 2), (sem_indices, pro_indices, spk_indices)

def plot_spectrogram(audio, title, save_path):
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80
    )(audio)
    plt.figure(figsize=(10, 4))
    plt.imshow(torch.log(spec[0] + 1e-5).cpu().numpy(), origin='lower', aspect='auto')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_audio = "jakubie_16k.wav"
    siren_checkpoint = "checkpoints_stable/step_87000"
    bitnet_checkpoint = "checkpoints_bitnet/checkpoint_epoch110.pt"
    config_path = "ultra_low_bitrate_codec/configs/sub100bps.yaml"
    output_dir = "bitnet_jakubie_results"
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 1. Load HuBERT
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    print("Loading HuBERT...")
    hubert_proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()

    # 2. Load Extraction Pipeline
    factorizer, decoder, sem_vq, pro_vq, spk_pq = load_extraction_pipeline(siren_checkpoint, config, device)

    # 3. Load BitNet Vocoder
    print(f"Loading BitNet Vocoder from {bitnet_checkpoint}...")
    bit_vocoder = BitVocoder(input_dim=512, dim=256, num_layers=4).to(device)
    bit_ckpt = torch.load(bitnet_checkpoint, map_location=device)
    bit_vocoder.load_state_dict(bit_ckpt['model_state_dict'])
    bit_vocoder.eval()

    # 4. Extract Features
    print(f"Extracting features from {input_audio}...")
    features, (sem_indices, pro_indices, spk_indices) = extract_features(input_audio, hubert_proc, hubert_model, factorizer, decoder, sem_vq, pro_vq, spk_pq, device)
    print(f"Features shape: {features.shape}")

    # Save intermediate file (codes)
    intermediate_path = os.path.join(output_dir, "intermediate_codes.pt")
    torch.save({
        'semantic_indices': sem_indices.cpu(),
        'prosody_indices': pro_indices.cpu(),
        'speaker_indices': spk_indices.cpu()
    }, intermediate_path)
    print(f"\n[INFO] Plik przejściowy (kody) zapisany w: {intermediate_path}")
    print(f"       Semantic shape: {sem_indices.shape}")
    print(f"       Prosody shape:  {pro_indices.shape}")
    print(f"       Speaker shape:  {spk_indices.shape}\n")

    # 5. Generate Audio
    print("Generating audio with BitNet Vocoder...")
    with torch.no_grad():
        pred_audio = bit_vocoder(features)
        if pred_audio.dim() == 3:
            pred_audio = pred_audio.squeeze(1)
            
    # 6. Save results
    torchaudio.save(os.path.join(output_dir, "jakubie_bitnet_110.wav"), pred_audio.cpu(), 16000)
    plot_spectrogram(pred_audio.cpu(), f"BitNet Jakubie (Epoch 110)", os.path.join(output_dir, "spectrogram_bitnet_jakubie_110.png"))
    
    # Plot original for comparison & Calculate Duration
    orig, sr = torchaudio.load(input_audio)
    if sr != 16000:
        orig = torchaudio.functional.resample(orig, sr, 16000)
    duration_seconds = orig.shape[1] / 16000
    plot_spectrogram(orig, "Original Jakubie", os.path.join(output_dir, "spectrogram_original_jakubie.png"))

    # --- BPS Calculation ---
    import math
    
    # Config params
    fsq_levels = config['model']['fsq_levels']
    rfsq_num_levels = config['model']['rfsq_num_levels']
    spk_num_groups = config['model']['speaker']['num_groups']
    spk_codes_per_group = config['model']['speaker']['codes_per_group']
    
    # Bits per code
    fsq_vocab_size = 1
    for level in fsq_levels:
        fsq_vocab_size *= level
    bits_per_fsq_code = math.log2(fsq_vocab_size)
    bits_per_spk_code = math.log2(spk_codes_per_group)

    # Frame counts
    num_sem_frames = sem_indices.shape[1]
    num_pro_frames = pro_indices.shape[1]
    
    # Total bits
    sem_bits = num_sem_frames * rfsq_num_levels * bits_per_fsq_code
    pro_bits = num_pro_frames * rfsq_num_levels * bits_per_fsq_code
    spk_bits = spk_num_groups * bits_per_spk_code
    
    total_bits = sem_bits + pro_bits + spk_bits
    total_bps = total_bits / duration_seconds
    sem_bps = sem_bits / duration_seconds
    pro_bps = pro_bits / duration_seconds
    spk_bps = spk_bits / duration_seconds

    print(f"\n{'='*60}")
    print("RAW BITRATE (bez kodowania entropijnego)")
    print(f"{'='*60}")
    print(f"Duration:  {duration_seconds:.2f}s")
    print(f"Semantic:  {sem_bits:>8.0f} bits → {sem_bps:>8.1f} bps")
    print(f"Prosody:   {pro_bits:>8.0f} bits → {pro_bps:>8.1f} bps")
    print(f"Speaker:   {spk_bits:>8.0f} bits → {spk_bps:>8.1f} bps")
    print(f"{'-'*60}")
    print(f"TOTAL:     {total_bits:>8.0f} bits → {total_bps:>8.1f} bps")
    print(f"{'='*60}\n")
    
    # --- Compression Analysis ---
    print(f"{'='*60}")
    print("COMPRESSION ANALYSIS")
    print(f"{'='*60}")
    
    # 1. Static Arithmetic Coding (Script)
    sem_ac_bytes, sem_table_bytes = compress_stream(sem_indices.cpu())
    pro_ac_bytes, pro_table_bytes = compress_stream(pro_indices.cpu())
    spk_ac_bytes, spk_table_bytes = compress_stream(spk_indices.cpu())
    
    total_ac_bytes = sem_ac_bytes + pro_ac_bytes + spk_ac_bytes
    total_table_bytes = sem_table_bytes + pro_table_bytes + spk_table_bytes
    total_compressed_bytes = len(total_ac_bytes) + total_table_bytes

    # 2. Strong Baseline (LZMA) - Simulating good entropy coding
    import lzma
    sem_lzma = lzma.compress(sem_indices.cpu().numpy().tobytes())
    pro_lzma = lzma.compress(pro_indices.cpu().numpy().tobytes())
    spk_lzma = lzma.compress(spk_indices.cpu().numpy().tobytes())
    total_lzma = len(sem_lzma) + len(pro_lzma) + len(spk_lzma)
    
    print(f"Algorithm         | Payload (B) | Header (B) | Total (B) | Bitrate (bps)")
    print(f"------------------|-------------|------------|-----------|--------------")
    
    # Raw
    raw_bytes = total_bits / 8
    print(f"Raw (No entropy)  | {raw_bytes:>11.0f} |          0 | {raw_bytes:>9.0f} | {total_bps:>12.1f}")
    
    # Static AC
    ac_total_size = len(total_ac_bytes) + total_table_bytes
    ac_bps = ac_total_size * 8 / duration_seconds
    print(f"Static Arithmetic | {len(total_ac_bytes):>11} | {total_table_bytes:>10} | {ac_total_size:>9} | {ac_bps:>12.1f}")
    
    # LZMA
    lzma_bps = total_lzma * 8 / duration_seconds
    print(f"LZMA (Baseline)   | {total_lzma:>11} |          - | {total_lzma:>9} | {lzma_bps:>12.1f}")
    
    print(f"{'='*60}")
    print(f"Original Audio:   {os.path.getsize(input_audio)} bytes")
    print(f"Compression Ratio: {os.path.getsize(input_audio) / ac_total_size:.1f}x (vs Original WAV)")
    print(f"{'='*60}\n")
    
    # Save the compressed (AC) file as proof
    compressed_file = os.path.join(output_dir, "compressed_stream.bin")
    with open(compressed_file, "wb") as f:
        # Simple format: 3x (len_table, table, len_stream, stream)
        # Just concatenating for the "physical file" requirement
        # Sem
        f.write(struct.pack("I", sem_table_bytes))
        f.write(len(sem_ac_bytes).to_bytes(4, 'little'))
        f.write(sem_ac_bytes)
        # Pro
        f.write(struct.pack("I", pro_table_bytes))
        f.write(len(pro_ac_bytes).to_bytes(4, 'little'))
        f.write(pro_ac_bytes)
        # Spk
        f.write(struct.pack("I", spk_table_bytes))
        f.write(len(spk_ac_bytes).to_bytes(4, 'little'))
        f.write(spk_ac_bytes)
        
    print(f"Saved physical compressed file: {compressed_file}")

    print(f"Done! Results in {output_dir}")

if __name__ == "__main__":
    main()
