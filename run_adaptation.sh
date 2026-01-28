#!/bin/bash
# Script to run SIREN TinyHubert Factorizer Adaptation (Frozen Decoder)

# Ensure we are in the project root
cd "$(dirname "$0")"

echo "🚀 Starting SIREN Adaptation Training..."
echo "Target: checkpoints/checkpoints_factorizer_tiny_frozen"
echo "Frozen Components: Decoder, Quantizers, EntropyModel"

# Check if we should resume or start fresh
if [ -d "checkpoints/checkpoints_factorizer_tiny_frozen/step_*" ]; then
    echo "🔄 Resuming from latent checkpoint..."
    ARGS=""
else
    echo "✨ Starting FRESH from pretrained step_87000..."
    ARGS="--fresh --pretrained_checkpoint checkpoints/checkpoints_stable/step_87000"
fi

# Run training
python3 src/ultra_low_bitrate_codec/train_factorizer_tiny.py \
    --config src/ultra_low_bitrate_codec/configs/ultra200bps_tiny_adapt.yaml \
    --checkpoint_dir checkpoints/checkpoints_factorizer_tiny_frozen \
    $ARGS

