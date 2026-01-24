# Achieve 200bps Quality Audio Codec

## Goal
~200 bps (190-230 range) with good perceptual quality (MOS > 3.0)

## Tasks

### Phase 1: Setup
- [x] Training running with anti-banding fixes
- [x] Download MOS evaluation model (speechmos + onnxruntime)
- [x] Create inference + bitrate calculation script
- [x] Create quality evaluation loop
- [x] Analyze current config bitrate (2250 bps raw → too high)
- [x] Create ultra200bps.yaml config (125 bps raw target)

### Phase 2: Monitor & Iterate
- [x] Analyze config (sub100bps = 2250 bps raw, WAY too high)
- [x] Create ultra200bps.yaml (355 bps raw → 195-213 bps with entropy)
- [x] Fix train.py DataLoader (num_workers=0, persistent_workers conditional)
- [x] TRAINING COMPLETE: 100k steps, 6h40m, loss=32.65, Val MEL=0.678, BPS=142
- [x] Evaluated MOS: 2.10 (Bitrate: 195 bps ✅, Quality: Needs Improvement ⚠️)

### Phase 3: Quality Improvement (Target MOS > 3.0 at ~200bps)
- [x] Pivot to No-GAN strategy (User Request)
- [x] Migrate to true BitNet 1.58b architecture (Refactored all models, Pushed to Git)
- [x] Setup `uv run train-bitnet` command
- [/] Train Large BitNet model (Step 9200, BPS 187-200, Validation MEL 0.64)
- [x] Evaluate MOS on BitNet model (Step 9000: MOS 2.23, Bitrate ~200bps entropy / 356bps raw)
- [ ] Finalize best model

### Phase 3: Results
- [ ] Best model saved with MOS > 3.0
- [ ] Bitrate verified in target range
- [ ] Document final architecture choices
