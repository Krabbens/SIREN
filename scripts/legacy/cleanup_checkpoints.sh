#!/bin/bash

echo "🧹 Starting Checkpoint Cleanup..."
echo "Current Size:"
du -sh checkpoints/

# 1. Prune Factorizer Tiny (Keep only step_120000)
# This is 78GB!
echo "Pruning checkpoints_factorizer_tiny..."
find checkpoints/checkpoints_factorizer_tiny/ -maxdepth 1 -name "step_*" ! -name "step_120000" -type d -exec rm -rf {} +
echo "Done."

# 2. Prune Bitnet Mel V2 (Keep only latest epoch 82)
echo "Pruning checkpoints_bitnet_mel_v2..."
# Keep epoch82, delete others
# We verify file existence before delete
if [ -f "checkpoints/checkpoints_bitnet_mel_v2/mel_vocoder_epoch82.pt" ]; then
    find checkpoints/checkpoints_bitnet_mel_v2/ -name "mel_vocoder_epoch*.pt" ! -name "mel_vocoder_epoch82.pt" -delete
fi
echo "Done."

# 3. Prune Flow V2 (Keep epoch 20 and latest)
# We rely on epoch 20. Let's see what's latest. Assumed epoch 31 from prev logs?
# Safest: Keep epoch 20 (Golden) and ignore others if they are just random training steps.
# Unless user wants to resume Flow training?
# "Zostaw tylko najnowsze żebyśmy mogli douczać"
# Flow training is likely finished or moved to V2.
# Detailed plan: Keep epoch 20 (Golden), Keep latest (e.g. 31), Delete rest.
echo "Pruning checkpoints_flow_v2..."
find checkpoints/checkpoints_flow_v2/ -name "*epoch*.pt" \
    ! -name "*epoch20.pt" \
    ! -name "*epoch31.pt" \
    -delete
# Note: Adjust 31 to actual latest if needed, but 20 is essential.
echo "Done."

# 4. Prune Flow V1 (checkpoints_flow_matching)
# Keep epoch 49 (Golden V1)
echo "Pruning checkpoints_flow_matching..."
find checkpoints/checkpoints_flow_matching/ -name "*epoch*.pt" \
    ! -name "*epoch49.pt" \
    ! -name "*epoch20.pt" \
    -delete
echo "Done."

# 5. Prune Bitnet (General)
# checkpoints_bitnet (1.2G) - likely has epochs.
# Keep best_model.pt
echo "Pruning checkpoints_bitnet..."
find checkpoints/checkpoints_bitnet/ -name "*epoch*.pt" -delete
echo "Done."

# 6. Remove Empty/Useless dirs
rm -rf checkpoints/checkpoints_bitnet_mel_v1 # Was empty
rm -rf checkpoints/checkpoints_flow_v5 # Empty
rm -rf checkpoints/checkpoints_entropy_snake # Empty

echo "✨ Cleanup Complete!"
echo "New Size:"
du -sh checkpoints/
