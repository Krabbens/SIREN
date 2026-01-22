#!/bin/bash
# =========================================================
# Ultra-Low Bitrate Codec V2 - Full Setup & Training Script
# =========================================================
# This script:
# 1. Downloads LibriTTS multi-speaker dataset
# 2. Precomputes HuBERT features
# 3. Starts training with improved architecture
# =========================================================

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
DATA_DIR="${PROJECT_ROOT}/data"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
CONFIG="${PROJECT_ROOT}/ultra_low_bitrate_codec/configs/improved.yaml"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints_v2"
FEATURES_DIR="${DATA_DIR}/features_libritts"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==========================================================${NC}"
echo -e "${GREEN}  Ultra-Low Bitrate Codec V2 - Setup & Training${NC}"
echo -e "${GREEN}==========================================================${NC}"

# =========================================================
# STEP 1: Download LibriTTS
# =========================================================
echo -e "\n${YELLOW}📥 STEP 1: Downloading LibriTTS...${NC}"

if [ -f "${DATA_DIR}/libritts_train.json" ]; then
    echo -e "${GREEN}✓ LibriTTS manifest already exists, skipping download${NC}"
else
    cd "${PROJECT_ROOT}"
    source .venv/bin/activate
    
    # Install dependencies if needed
    pip install datasets soundfile librosa --quiet
    
    # Download train-clean-100 (about 54 hours, 251 speakers)
    python ${SCRIPTS_DIR}/download_libritts.py \
        --output-dir ${DATA_DIR} \
        --subset train-clean-100 \
        --max-hours 50
    
    echo -e "${GREEN}✓ LibriTTS downloaded${NC}"
fi

# =========================================================
# STEP 2: Precompute HuBERT Features
# =========================================================
echo -e "\n${YELLOW}🧠 STEP 2: Precomputing HuBERT features...${NC}"

mkdir -p ${FEATURES_DIR}

if [ "$(ls -A ${FEATURES_DIR} 2>/dev/null | head -1)" ]; then
    FEATURE_COUNT=$(ls ${FEATURES_DIR}/*.pt 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found ${FEATURE_COUNT} existing features${NC}"
else
    echo "Extracting features (this may take a while)..."
    
    cd "${PROJECT_ROOT}"
    source .venv/bin/activate
    
    python ${SCRIPTS_DIR}/precompute_features.py \
        --manifest ${DATA_DIR}/libritts_train.json \
        --output-dir ${FEATURES_DIR} \
        --hubert-layer 9 \
        --device cuda
    
    echo -e "${GREEN}✓ Features extracted${NC}"
fi

# =========================================================
# STEP 3: Update Config paths
# =========================================================
echo -e "\n${YELLOW}⚙️  STEP 3: Updating configuration...${NC}"

# Update manifest paths in config if needed
sed -i "s|train_manifest:.*|train_manifest: \"${DATA_DIR}/libritts_train.json\"|g" ${CONFIG}
sed -i "s|val_manifest:.*|val_manifest: \"${DATA_DIR}/libritts_val.json\"|g" ${CONFIG}

echo -e "${GREEN}✓ Config updated${NC}"

# =========================================================
# STEP 4: Start Training
# =========================================================
echo -e "\n${YELLOW}🚀 STEP 4: Starting training...${NC}"

mkdir -p ${CHECKPOINT_DIR}

cd "${PROJECT_ROOT}"
source .venv/bin/activate

echo "Training will start with:"
echo "  - Config: ${CONFIG}"
echo "  - Features: ${FEATURES_DIR}"
echo "  - Checkpoints: ${CHECKPOINT_DIR}"
echo ""

# Run training
python ${SCRIPTS_DIR}/train_v2.py \
    --config ${CONFIG} \
    --features ${FEATURES_DIR} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    2>&1 | tee ${CHECKPOINT_DIR}/training_full.log

echo -e "\n${GREEN}==========================================================${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}==========================================================${NC}"
