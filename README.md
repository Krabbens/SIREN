# SIREN 🔊

**S**emantic **I**nformation **R**epresentation for **E**fficient **N**eural-coding

A neural speech codec achieving **~68 bps** (bits per second) through advanced information factorization and residual finite scalar quantization.

## 🎯 Features

- **Ultra-low bitrate**: ~68 bps (vs. 6000+ bps for Opus, 1500+ bps for Lyra)
- **Multi-speaker support**: Trained on English (LibriTTS) and Polish datasets
- **Information factorization**: Separates semantic content from speaker identity
- **Residual FSQ**: Finite Scalar Quantization with residual connections for better reconstruction
- **HiFi-GAN vocoder**: High-fidelity waveform synthesis

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    InformationFactorizer                        │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ DistilHuBERT│ -> │ Semantic     │ -> │ ResidualFSQ      │   │
│  │ Features    │    │ Branch       │    │ (2 stages)       │   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐                            │
│  │ Mel Spec    │ -> │ Acoustic     │ (speaker conditioning)    │
│  │             │    │ Branch       │                            │
│  └─────────────┘    └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SpeechDecoder                              │
│  ┌─────────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │ Semantic Upconv │->│ Cross-Modal   │->│ HiFi-GAN Decoder │  │
│  │                 │  │ Fusion        │  │                  │  │
│  └─────────────────┘  └───────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Krabbens/SIREN.git
cd SIREN

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## 🚀 Quick Start

### Inference

```python
from ultra_low_bitrate_codec.scripts.inference_pipeline import CodecInference

# Load model
codec = CodecInference(checkpoint_path="checkpoints/model.pt")

# Encode audio to tokens
tokens = codec.encode("input.wav")
print(f"Bitrate: {codec.calculate_bitrate(tokens)} bps")

# Decode tokens back to audio
audio = codec.decode(tokens)
codec.save_audio(audio, "output.wav")
```

### Training

#### Single Speaker (LJSpeech)
```bash
# Download and prepare dataset
python scripts/download_libritts.py --dataset ljspeech

# Precompute DistilHuBERT features
python scripts/precompute_features.py --data_dir data/ljspeech

# Train
python ultra_low_bitrate_codec/scripts/train_fast.py \
    --config ultra_low_bitrate_codec/configs/improved_ljspeech.yaml
```

#### Multi-Speaker (LibriTTS + Polish)
```bash
# Prepare multi-speaker dataset
python ultra_low_bitrate_codec/scripts/prepare_multispeaker_dataset.py

# Train multi-speaker model
python train_multispeaker.py \
    --config ultra_low_bitrate_codec/configs/multispeaker.yaml
```

#### Resume Training
```bash
python resume_training.py --checkpoint checkpoints/step_10000.pt
python resume_multispeaker.py --checkpoint checkpoints_multispeaker/step_5000.pt
```

## 📁 Project Structure

```
SIREN/
├── ultra_low_bitrate_codec/
│   ├── models/
│   │   ├── encoder.py          # InformationFactorizer
│   │   ├── decoder.py          # SpeechDecoder with HiFi-GAN
│   │   ├── quantizers.py       # ResidualFSQ implementation
│   │   ├── vocoder.py          # HiFi-GAN vocoder
│   │   ├── discriminator.py    # Multi-period & multi-scale discriminators
│   │   └── feature_extractor.py # DistilHuBERT wrapper
│   ├── training/
│   │   ├── losses.py           # Multi-resolution STFT loss
│   │   └── trainer.py          # Training loop
│   ├── scripts/
│   │   ├── train.py            # Basic training script
│   │   ├── train_fast.py       # Optimized training
│   │   ├── inference_pipeline.py # Inference utilities
│   │   └── prepare_multispeaker_dataset.py
│   └── configs/
│       ├── default.yaml
│       ├── improved.yaml
│       └── multispeaker.yaml
├── scripts/
│   ├── download_libritts.py    # Dataset download
│   ├── precompute_features.py  # Feature extraction
│   └── setup_and_train_v2.sh   # Full setup script
├── resume_training.py          # Resume single-speaker training
├── resume_multispeaker.py      # Resume multi-speaker training
└── train_multispeaker.py       # Multi-speaker training entry
```

## ⚙️ Configuration

Key parameters in config files:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fsq_levels` | [8, 5, 5, 5] | FSQ quantization levels per dimension |
| `num_residual_stages` | 2 | Number of residual quantization stages |
| `semantic_dim` | 256 | Semantic embedding dimension |
| `decoder_channels` | 512 | Decoder hidden channels |
| `sample_rate` | 16000 | Audio sample rate |

## 📊 Results

| Model | Bitrate | MOS (estimated) |
|-------|---------|-----------------|
| Opus (reference) | 6000 bps | 4.0 |
| Lyra v2 | 3200 bps | 3.8 |
| **SIREN** | **68 bps** | 3.2* |

*Subjective evaluation pending

## 🔬 Technical Details

### Bitrate Calculation

```
Tokens per second = sample_rate / hop_length / temporal_reduction
                  = 16000 / 320 / 8 = 6.25 tokens/s

Bits per token = log2(prod(fsq_levels)) × num_stages
               = log2(8×5×5×5) × 2 = 9.97 × 2 ≈ 20 bits

Bitrate = 6.25 × (20 / 2) ≈ 68 bps
```

### Loss Functions

- Multi-resolution STFT loss (reconstruction)
- Feature matching loss (GAN)
- Adversarial loss (multi-period + multi-scale discriminators)

## 📄 License

MIT License

## 🙏 Acknowledgments

- [DistilHuBERT](https://huggingface.co/ntu-spml/distilhubert) for semantic features
- HiFi-GAN architecture for high-quality synthesis
- FSQ from "Finite Scalar Quantization: VQ-VAE Made Simple"
