#!/bin/bash
set -e

# Configuration
# Configuration
# Use current directory
BASE_DIR="$(pwd)"
DATA_DIR="$BASE_DIR/data"
FEATURE_DIR="$BASE_DIR/data/features_mixed"
CONFIG_PATH="$BASE_DIR/ultra_low_bitrate_codec/configs/sub100bps.yaml"
CHECKPOINT_DIR="$BASE_DIR/checkpoints_sub100bps_mfcc"

echo "==================================================="
echo "🚀 STARTING SETUP AND TRAINING PIPELINE"
echo "==================================================="

# Activate Venv
source .venv/bin/activate || echo "⚠️ Could not activate .venv"

# 1. Download Datasets
echo "📥 1. Downloading Datasets..."

# LibriTTS
echo "   - LibriTTS (train-clean-100)..."
# python3 ultra_low_bitrate_codec/download_data.py --output-dir "$DATA_DIR" --dataset libritts --subset train-clean-100

# Common Voice PL
echo "   - Common Voice PL..."
# python3 ultra_low_bitrate_codec/download_data.py --output-dir "$DATA_DIR" --dataset common_voice_pl

# 2. Merge Manifests
echo "🔄 2. Merging Manifests..."
python3 -c "
import json
import os

data_dir = '$DATA_DIR'
train_files = ['libritts_train.json', 'cv_pl_train.json']
val_files = ['libritts_val.json', 'cv_pl_val.json']

def merge(files, out_name):
    merged = []
    for f in files:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            with open(p) as fp:
                merged.extend(json.load(fp))
        else:
            print(f'Warning: {p} not found')
    
    with open(os.path.join(data_dir, out_name), 'w') as fp:
        json.dump(merged, fp, indent=2)
    print(f'Merged {len(merged)} items into {out_name}')

merge(train_files, 'mixed_train.json')
merge(val_files, 'mixed_val.json')
"

# 3. Preprocess (Extract Features)
echo "⚙️ 3. Preprocessing Features (CUDA)..."
# We run preprocessing on the merged manifest
# Assuming preprocess_data.py takes a manifest JSON as input
python3 ultra_low_bitrate_codec/preprocess_data.py \
    --input "$DATA_DIR/mixed_train.json" \
    --output-dir "$FEATURE_DIR" \
    --device cuda \
    --batch-size 16

python3 ultra_low_bitrate_codec/preprocess_data.py \
    --input "$DATA_DIR/mixed_val.json" \
    --output-dir "$FEATURE_DIR" \
    --device cuda \
    --batch-size 16

# 4. Update Config (Optional, but we pass paths via args usually? No, yaml is hardcoded mostly)
# We need to sed the config or just rely on the user editing it. 
# But the user asked us to do it.
# Let's use Sed to update the config file to point to new manifests and feature dir.

echo "📝 4. Updating Config Paths..."
sed -i "s|train_manifest: .*|train_manifest: \"$DATA_DIR/mixed_train.json\"|g" "$CONFIG_PATH"
sed -i "s|val_manifest: .*|val_manifest: \"$DATA_DIR/mixed_val.json\"|g" "$CONFIG_PATH"
sed -i "s|feature_dir: .*|feature_dir: \"$FEATURE_DIR\"|g" "$CONFIG_PATH"

# 5. Run Training
echo "🔥 5. Starting Training..."
python3 ultra_low_bitrate_codec/train.py \
    --config "$CONFIG_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR"

