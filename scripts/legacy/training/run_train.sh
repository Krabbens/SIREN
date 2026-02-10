#!/bin/bash
echo "Waiting for data/flow_dataset_24k to populate (checking if > 1000 files)..."
while true; do
    count=$(ls data/flow_dataset_24k/*.pt 2>/dev/null | wc -l)
    if [ "$count" -gt 1000 ]; then
        echo "Found $count files. Starting training..."
        break
    fi
    echo "Files found: $count. Waiting..."
    sleep 30
done

uv run python train_flow_matching.py \
    --checkpoint checkpoints/checkpoints_stable/step_87000/decoder.pt \
    --data_dir data/flow_dataset_24k \
    --output_dir checkpoints/checkpoints_flow_new \
    --config ultra_low_bitrate_codec/configs/ultra200bps_large.yaml \
    --lr 4e-4 \
    --epochs 100 "$@"
