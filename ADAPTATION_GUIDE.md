# SIREN Adaptive Training Guide (TinyHubert)

## 🎯 Objective
Adapt the **Factorizer** to work with **TinyHubert** features while preserving compatibility with the pre-trained **Flow Matching** and **Fuser** models.

## 🚩 Problem & Solution
**The Issue:**
Previous attempts to adapt the Factorizer (`train_factorizer_tiny.py`) allowed the **Decoder** and **Quantizers** to update their weights. This caused the "semantic language" (codebooks) to drift. While the model could reconstruct audio (autoencoder), the codes it produced were no longer understandable by the frozen Flow/Fuser (which expect the original "language").

**The Solution:**
We perform **Constrained Adaptation**:
1. **Freeze** the Decoder, Quantizers (ResidualFSQ), and Entropy Model.
2. **Train ONLY the Factorizer**.
3. Force the Factorizer to learn how to map TinyHubert features to the *existing, frozen* semantic codes.

## 📁 Key Files
- `src/ultra_low_bitrate_codec/train_factorizer_tiny.py`: The training script (modified to freeze downstream components).
- `src/ultra_low_bitrate_codec/configs/ultra200bps_tiny_adapt.yaml`: Configuration for adaptation.
- `checkpoints/checkpoints_factorizer_tiny_frozen/`: directory where the new, correct model is training.

## 🚀 How to Run (On New Machine)

### 1. Training
To start or resume the adaptive training:

```bash
./run_adaptation.sh
```

Or manually:
```bash
python3 src/ultra_low_bitrate_codec/train_factorizer_tiny.py \
    --config src/ultra_low_bitrate_codec/configs/ultra200bps_tiny_adapt.yaml \
    --checkpoint_dir checkpoints/checkpoints_factorizer_tiny_frozen \
    --pretrained_checkpoint checkpoints/checkpoints_stable/step_87000
```
*Note: Remove `--fresh` if resuming from a checkpoint in `checkpoints_factorizer_tiny_frozen`.*

### 2. Inference (Verification)
Once training converges (Loss < 300, Entropy matches baseline), run the verification script:

```bash
python3 src/ultra_low_bitrate_codec/inference_tiny_v2.py \
    --config src/ultra_low_bitrate_codec/configs/ultra200bps_tiny_adapt.yaml \
    --factorizer_dir checkpoints/checkpoints_factorizer_tiny_frozen/step_XXXXXX \
    --flow_checkpoint checkpoints/checkpoints_flow_v2/flow_epoch20.pt \
    --input_wav data/jakubie.wav \
    --output_wav outputs/test_adaptation.wav
```

## 📊 Baseline Configuration (Verified)
The "Golden Standard" pipeline (Original Hubert) uses:
- **Factorizer**: `step_87000` (Original)
- **Flow**: `checkpoints_flow_v2/flow_epoch20.pt` (Epoch 20)
- **Fuser**: `checkpoints_flow_v2/fuser_epoch20.pt` (V2, ~11MB)
- **Vocoder**: `checkpoints/vocoder_mel/vocoder_latest.pt` (80-dim input)
