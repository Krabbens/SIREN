#!/bin/bash
# scripts/run_ddp_lan.sh
# Launch DDP distributed training over direct LAN connection

set -e

LOCAL_PYTHON="/home/sperm/siren/SIREN/.venv/bin/python3"
REMOTE_PYTHON="/home/kubs/siren_dist/.venv/bin/python3"

# Network config - Direct LAN IPs
MASTER_ADDR="192.168.0.100"
MASTER_PORT=29500

# Kill old processes
echo "=== Killing old processes ==="
pkill -f torch || true
ssh -p 2222 -o StrictHostKeyChecking=no kubs@192.168.0.102 "pkill -f torch" || true

sleep 2

echo "=== Syncing code to remote ==="
rsync -avz -e "ssh -p 2222 -o StrictHostKeyChecking=no" scripts src kubs@192.168.0.102:~/siren_dist/

echo "=== Launching DDP ==="

# Launch local master first (in background)
cd /tmp/siren_run
PYTHONPATH=$(pwd)/src \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NCCL_SOCKET_IFNAME=eth0 \
NCCL_DEBUG=INFO \
$LOCAL_PYTHON -m torch.distributed.run \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_microhubert.py \
    --audio_dir data/audio \
    --features_dir data/features_hubert_layer9 \
    --output_dir checkpoints/microhubert \
    --epochs 100 \
    --resume checkpoints/microhubert/microhubert_ep45.pt \
    --lr 3e-4 &

LOCAL_PID=$!
echo "Launched Local Master (PID $LOCAL_PID)"

sleep 5

# Launch remote worker
ssh -p 2222 -o StrictHostKeyChecking=no kubs@192.168.0.102 "cd ~/siren_dist && \
    PYTHONPATH=\$(pwd)/src \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$MASTER_PORT \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_DEBUG=INFO \
    $REMOTE_PYTHON -m torch.distributed.run \
        --nproc_per_node=1 \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        scripts/train_microhubert.py \
        --audio_dir data/audio \
        --features_dir data/features_hubert_layer9 \
        --output_dir checkpoints/microhubert \
        --epochs 100 \
        --resume checkpoints/microhubert/microhubert_ep45.pt \
        --lr 3e-4"

# Wait for local
wait $LOCAL_PID

echo "=== Training Complete ==="
