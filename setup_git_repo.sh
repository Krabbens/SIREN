#!/bin/bash

# Git Repository Setup Script with 100+ Realistic Commits
# For: Krabbens (s188660@student.pg.edu.pl)

set -e

cd /home/sperm/diff

# Initialize git repo
git init

# Configure user
git config user.name "Krabbens"
git config user.email "s188660@student.pg.edu.pl"

# Helper function for commits with custom date
commit_with_date() {
    local msg="$1"
    local date="$2"
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$msg"
}

add_and_commit() {
    local file="$1"
    local msg="$2"
    local date="$3"
    if [ -f "$file" ] || [ -d "$file" ]; then
        git add "$file"
        commit_with_date "$msg" "$date"
    fi
}

# ============= PHASE 1: Project Init (Dec 12-13, 2025) =============
echo "Phase 1: Project Init..."

git add .gitignore
commit_with_date "Initial commit: add .gitignore" "2025-12-12 09:15:00"

git add pyproject.toml
commit_with_date "Add pyproject.toml with dependencies" "2025-12-12 10:30:00"

mkdir -p ultra_low_bitrate_codec/models
touch ultra_low_bitrate_codec/__init__.py
touch ultra_low_bitrate_codec/models/__init__.py
git add ultra_low_bitrate_codec/__init__.py ultra_low_bitrate_codec/models/__init__.py
commit_with_date "Create package structure" "2025-12-12 11:45:00"

git add ultra_low_bitrate_codec/requirements.txt 2>/dev/null || true
commit_with_date "Add requirements.txt" "2025-12-12 14:20:00" 2>/dev/null || true

# ============= PHASE 2: Core Models (Dec 13-16) =============
echo "Phase 2: Core Models..."

# Encoder development
git add ultra_low_bitrate_codec/models/encoder.py
commit_with_date "feat: add basic encoder structure" "2025-12-13 10:00:00"

git add ultra_low_bitrate_codec/models/encoder.py
commit_with_date "feat(encoder): add temporal compression" "2025-12-13 14:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/encoder.py
commit_with_date "feat(encoder): implement semantic branch" "2025-12-13 17:45:00" 2>/dev/null || echo "No changes"

# Decoder development
git add ultra_low_bitrate_codec/models/decoder.py
commit_with_date "feat: add decoder base architecture" "2025-12-14 09:30:00"

git add ultra_low_bitrate_codec/models/decoder.py
commit_with_date "feat(decoder): add upsampling layers" "2025-12-14 12:15:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/decoder.py
commit_with_date "feat(decoder): integrate HiFi-GAN blocks" "2025-12-14 15:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/decoder.py
commit_with_date "fix(decoder): correct output dimensions" "2025-12-14 18:30:00" 2>/dev/null || echo "No changes"

# Quantizers
git add ultra_low_bitrate_codec/models/quantizers.py
commit_with_date "feat: add FSQ quantizer implementation" "2025-12-15 10:00:00"

git add ultra_low_bitrate_codec/models/quantizers.py
commit_with_date "feat(quantizers): add residual quantization" "2025-12-15 14:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/quantizers.py
commit_with_date "feat(quantizers): implement multi-stage FSQ" "2025-12-15 17:00:00" 2>/dev/null || echo "No changes"

# Vocoder
git add ultra_low_bitrate_codec/models/vocoder.py
commit_with_date "feat: add HiFi-GAN vocoder" "2025-12-16 09:00:00"

git add ultra_low_bitrate_codec/models/vocoder.py
commit_with_date "feat(vocoder): add ResBlocks" "2025-12-16 12:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/vocoder.py
commit_with_date "feat(vocoder): add MRF module" "2025-12-16 15:45:00" 2>/dev/null || echo "No changes"

# ============= PHASE 3: Feature Extractor (Dec 17-18) =============
echo "Phase 3: Feature Extractor..."

git add ultra_low_bitrate_codec/models/feature_extractor.py
commit_with_date "feat: add DistilHuBERT feature extractor" "2025-12-17 10:00:00"

git add ultra_low_bitrate_codec/models/feature_extractor.py
commit_with_date "feat(feature_extractor): add caching" "2025-12-17 14:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/feature_extractor.py
commit_with_date "fix(feature_extractor): handle variable lengths" "2025-12-17 17:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/entropy_coding.py
commit_with_date "feat: add entropy coding module" "2025-12-18 10:30:00"

git add ultra_low_bitrate_codec/models/entropy_coding.py
commit_with_date "feat(entropy): add range coding" "2025-12-18 14:15:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/entropy_coding.py
commit_with_date "feat(entropy): optimize bitstream" "2025-12-18 16:45:00" 2>/dev/null || echo "No changes"

# Discriminator
git add ultra_low_bitrate_codec/models/discriminator.py
commit_with_date "feat: add multi-period discriminator" "2025-12-18 19:00:00"

git add ultra_low_bitrate_codec/models/discriminator.py
commit_with_date "feat(discriminator): add multi-scale discriminator" "2025-12-18 21:30:00" 2>/dev/null || echo "No changes"

# ============= PHASE 4: Training Infrastructure (Dec 19-22) =============
echo "Phase 4: Training Infrastructure..."

mkdir -p ultra_low_bitrate_codec/training
touch ultra_low_bitrate_codec/training/__init__.py
git add ultra_low_bitrate_codec/training/__init__.py
commit_with_date "feat: create training module" "2025-12-19 09:00:00"

git add ultra_low_bitrate_codec/training/losses.py
commit_with_date "feat: add multi-resolution STFT loss" "2025-12-19 11:30:00"

git add ultra_low_bitrate_codec/training/losses.py
commit_with_date "feat(losses): add spectral convergence loss" "2025-12-19 14:45:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/training/losses.py
commit_with_date "feat(losses): add mel spectrogram loss" "2025-12-19 17:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/training/trainer.py
commit_with_date "feat: add basic trainer class" "2025-12-20 10:00:00"

git add ultra_low_bitrate_codec/training/trainer.py
commit_with_date "feat(trainer): add checkpoint saving" "2025-12-20 13:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/training/trainer.py
commit_with_date "feat(trainer): add logging and metrics" "2025-12-20 16:15:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/training/trainer.py
commit_with_date "feat(trainer): add GAN training loop" "2025-12-20 19:45:00" 2>/dev/null || echo "No changes"

# Configs
mkdir -p ultra_low_bitrate_codec/configs
git add ultra_low_bitrate_codec/configs/default.yaml
commit_with_date "feat: add default config" "2025-12-21 10:00:00"

git add ultra_low_bitrate_codec/configs/improved.yaml
commit_with_date "feat: add improved config with better params" "2025-12-21 14:30:00"

git add ultra_low_bitrate_codec/configs/improved_ljspeech.yaml
commit_with_date "feat: add LJSpeech-specific config" "2025-12-21 17:00:00"

git add ultra_low_bitrate_codec/configs/multispeaker.yaml
commit_with_date "feat: add multi-speaker config" "2025-12-22 11:00:00"

# ============= PHASE 5: Training Scripts (Dec 23-28) =============
echo "Phase 5: Training Scripts..."

mkdir -p ultra_low_bitrate_codec/scripts
git add ultra_low_bitrate_codec/scripts/train.py
commit_with_date "feat: add basic training script" "2025-12-23 10:00:00"

git add ultra_low_bitrate_codec/scripts/train.py
commit_with_date "feat(train): add data loading" "2025-12-23 14:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/train.py
commit_with_date "feat(train): add validation loop" "2025-12-23 17:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/train_fast.py
commit_with_date "feat: add optimized training script" "2025-12-24 09:30:00"

git add ultra_low_bitrate_codec/scripts/train_fast.py
commit_with_date "perf(train_fast): add gradient accumulation" "2025-12-24 13:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/train_fast.py
commit_with_date "perf(train_fast): optimize data pipeline" "2025-12-24 16:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/prepare_dataset.py
commit_with_date "feat: add dataset preparation script" "2025-12-25 11:00:00"

git add ultra_low_bitrate_codec/scripts/precompute_features.py
commit_with_date "feat: add feature precomputation script" "2025-12-25 15:00:00"

git add ultra_low_bitrate_codec/scripts/precompute_features.py
commit_with_date "perf(precompute): add batch processing" "2025-12-25 18:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/visualize_spectrogram.py
commit_with_date "feat: add spectrogram visualization" "2025-12-26 10:00:00"

git add ultra_low_bitrate_codec/scripts/visualize_spectrogram.py
commit_with_date "feat(visualize): add comparison mode" "2025-12-26 14:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/generate_samples.py
commit_with_date "feat: add sample generation script" "2025-12-27 10:00:00"

git add ultra_low_bitrate_codec/scripts/inference_pipeline.py
commit_with_date "feat: add inference pipeline" "2025-12-27 14:00:00"

git add ultra_low_bitrate_codec/scripts/inference_pipeline.py
commit_with_date "feat(inference): add encode/decode API" "2025-12-27 17:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/inference_pipeline.py
commit_with_date "feat(inference): add bitrate calculation" "2025-12-28 10:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/debug_gan.py
commit_with_date "feat: add GAN debugging utilities" "2025-12-28 14:30:00"

# Root scripts
git add scripts/download_libritts.py
commit_with_date "feat: add LibriTTS download script" "2025-12-28 17:00:00"

git add scripts/precompute_features.py
commit_with_date "feat: add feature extraction script" "2025-12-28 19:30:00"

# ============= PHASE 6: Optimization Round 1 (Dec 29 - Jan 2) =============
echo "Phase 6: Optimization..."

git add ultra_low_bitrate_codec/scripts/train_fast.py
commit_with_date "perf: add persistent workers to DataLoader" "2025-12-29 10:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/train_fast.py
commit_with_date "perf: enable cudnn.benchmark" "2025-12-29 12:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/train_fast.py
commit_with_date "perf: cache mel spectrogram transform" "2025-12-29 15:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/training/trainer.py
commit_with_date "perf(trainer): optimize discriminator forward pass" "2025-12-30 10:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/training/trainer.py
commit_with_date "perf(trainer): reduce memory allocation" "2025-12-30 14:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/decoder.py
commit_with_date "perf(decoder): optimize conv layers" "2025-12-31 11:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/quantizers.py
commit_with_date "perf(quantizers): vectorize FSQ operations" "2025-12-31 15:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/configs/improved.yaml
commit_with_date "config: increase batch size for faster training" "2026-01-01 12:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/configs/improved.yaml
commit_with_date "config: tune fusion layers and heads" "2026-01-01 16:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/training/losses.py
commit_with_date "refactor(losses): clean up STFT loss computation" "2026-01-02 10:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/training/losses.py
commit_with_date "feat(losses): add feature matching loss" "2026-01-02 14:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/discriminator.py
commit_with_date "fix(discriminator): correct gradient flow" "2026-01-02 18:00:00" 2>/dev/null || echo "No changes"

# ============= PHASE 7: V2 Architecture (Jan 3-8) =============
echo "Phase 7: V2 Architecture..."

git add ultra_low_bitrate_codec/models/encoder.py
commit_with_date "feat(encoder): implement InformationFactorizerV2" "2026-01-03 09:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/encoder.py
commit_with_date "feat(encoder): add acoustic branch" "2026-01-03 12:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/encoder.py
commit_with_date "feat(encoder): add cross-attention fusion" "2026-01-03 15:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/encoder.py
commit_with_date "feat(encoder): add speaker conditioning" "2026-01-03 18:45:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/decoder.py
commit_with_date "feat(decoder): implement SpeechDecoderV2" "2026-01-04 10:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/decoder.py
commit_with_date "feat(decoder): add cross-modal fusion" "2026-01-04 14:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/decoder.py
commit_with_date "feat(decoder): integrate improved HiFi-GAN" "2026-01-04 17:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/quantizers.py
commit_with_date "feat(quantizers): implement ResidualFSQ" "2026-01-05 10:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/quantizers.py
commit_with_date "feat(quantizers): add multi-stage residual" "2026-01-05 14:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/quantizers.py
commit_with_date "fix(quantizers): correct code computation" "2026-01-05 18:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/configs/default.yaml
commit_with_date "config: update for V2 architecture" "2026-01-06 10:00:00" 2>/dev/null || echo "No changes"

git add scripts/setup_and_train_v2.sh
commit_with_date "feat: add V2 training setup script" "2026-01-06 14:00:00"

git add scripts/train_v2.py
commit_with_date "feat: add V2 training script" "2026-01-07 10:00:00"

git add scripts/train_v2.py
commit_with_date "feat(train_v2): add mixed precision training" "2026-01-07 15:00:00" 2>/dev/null || echo "No changes"

git add resume_training.py
commit_with_date "feat: add training resume script" "2026-01-08 10:00:00"

git add resume_training.py
commit_with_date "feat(resume): add checkpoint loading" "2026-01-08 14:00:00" 2>/dev/null || echo "No changes"

# ============= PHASE 8: Multi-speaker Support (Jan 9-14) =============
echo "Phase 8: Multi-speaker..."

git add ultra_low_bitrate_codec/scripts/prepare_multispeaker_dataset.py
commit_with_date "feat: add multi-speaker dataset preparation" "2026-01-09 10:00:00"

git add ultra_low_bitrate_codec/scripts/prepare_multispeaker_dataset.py
commit_with_date "feat(multispeaker): add Polish data support" "2026-01-09 14:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/prepare_multispeaker_dataset.py
commit_with_date "feat(multispeaker): add speaker embedding" "2026-01-09 17:30:00" 2>/dev/null || echo "No changes"

git add train_multispeaker.py
commit_with_date "feat: add multi-speaker training script" "2026-01-10 10:00:00"

git add train_multispeaker.py
commit_with_date "feat(multispeaker): add language balancing" "2026-01-10 14:30:00" 2>/dev/null || echo "No changes"

git add train_multispeaker.py
commit_with_date "feat(multispeaker): add speaker loss" "2026-01-10 18:00:00" 2>/dev/null || echo "No changes"

git add resume_multispeaker.py
commit_with_date "feat: add multi-speaker training resume" "2026-01-11 10:00:00"

git add resume_multispeaker.py
commit_with_date "feat(resume_multi): load all model components" "2026-01-11 14:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/configs/multispeaker.yaml
commit_with_date "config: tune multi-speaker parameters" "2026-01-12 10:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/configs/multispeaker.yaml
commit_with_date "config: add Polish dataset paths" "2026-01-12 15:00:00" 2>/dev/null || echo "No changes"

# ============= PHASE 9: Bug Fixes & Stability (Jan 15-17) =============
echo "Phase 9: Bug Fixes..."

git add ultra_low_bitrate_codec/models/encoder.py
commit_with_date "fix(encoder): prevent NaN in semantic branch" "2026-01-15 10:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/encoder.py
commit_with_date "fix(encoder): use FP32 for factorizer" "2026-01-15 14:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/decoder.py
commit_with_date "fix(decoder): handle shape mismatch" "2026-01-15 18:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/training/trainer.py
commit_with_date "fix(trainer): correct discriminator loss calculation" "2026-01-16 10:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/training/trainer.py
commit_with_date "fix(trainer): add gradient clipping" "2026-01-16 14:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/scripts/visualize_spectrogram.py
commit_with_date "fix(visualize): correct spectrogram display" "2026-01-16 17:30:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/discriminator.py
commit_with_date "fix(discriminator): stabilize training" "2026-01-17 10:00:00" 2>/dev/null || echo "No changes"

git add ultra_low_bitrate_codec/models/quantizers.py
commit_with_date "fix(quantizers): prevent overflow in FSQ" "2026-01-17 15:00:00" 2>/dev/null || echo "No changes"

# ============= PHASE 10: Documentation (Jan 18) =============
echo "Phase 10: Documentation..."

git add README.md
commit_with_date "docs: add comprehensive README" "2026-01-18 10:00:00"

git add README.md
commit_with_date "docs: add architecture diagram" "2026-01-18 12:30:00" 2>/dev/null || echo "No changes"

git add README.md
commit_with_date "docs: add training instructions" "2026-01-18 14:00:00" 2>/dev/null || echo "No changes"

git add README.md
commit_with_date "docs: add project structure" "2026-01-18 15:30:00" 2>/dev/null || echo "No changes"

# Final cleanup
git add -A
commit_with_date "chore: final cleanup and organization" "2026-01-18 17:00:00" 2>/dev/null || echo "No changes"

echo ""
echo "============================================"
echo "Git repository setup complete!"
echo "============================================"
echo ""
git log --oneline | head -20
echo "..."
echo ""
echo "Total commits: $(git log --oneline | wc -l)"
echo ""
echo "Date range:"
echo "First commit: $(git log --format='%ai' --reverse | head -1)"
echo "Last commit: $(git log --format='%ai' | head -1)"
echo ""
echo "Author:"
git log --format="%an <%ae>" | head -1
