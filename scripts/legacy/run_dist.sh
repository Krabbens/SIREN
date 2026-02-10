#!/bin/bash
# Launch distributed training using PyTorch DDP (torchrun)
# Must be run from /tmp/siren_run (or where paths match)

# Ensure valid path
if [ "$(pwd)" != "/tmp/siren_run" ]; then
    echo "Please run this script from /tmp/siren_run"
    echo "cd /tmp/siren_run"
    exit 1
fi

# Configuration
MASTER_ADDR="192.168.0.XYZ" # Replace with actual local IP - wait, I need to know my IP
# Actually, torchrun handles this if we run on each node or use a launcher.
# But for simple SSH setup, we can run on master node and explicit workers.
# The simplest way for ad-hoc is running this script on EACH node or using a launcher.
# But I promised an auto-launcher.

# Let's get local IP
# Configuration
MASTER_ADDR="localhost" # Both nodes see Master as localhost (Local is real, Remote is tunneled)
MASTER_PORT=29500
NNODES=2

# Setup Reverse SSH Tunnel (Remote 29500 -> Local 29500)
# Remote Worker connects to localhost:29500 -> forwarded to Local Master
echo "Setting up Reverse SSH Tunnel..."
ssh -p 2222 -N -f -R $MASTER_PORT:localhost:$MASTER_PORT kubs@192.168.0.102
TUNNEL_PID=$!
echo "Tunnel active (PID approx via pgrep)"

# Pythons
REMOTE_PYTHON="python3"
LOCAL_PYTHON="/home/sperm/siren/SIREN/.venv/bin/python3"

echo "Master IP: $MASTER_ADDR"
echo "Starting Distributed Training (Remote is Master)..."

# Launch Remote Worker (Rank 1) in background
# Connects to localhost:29500 (tunneled to us)
ssh -p 2222 kubs@192.168.0.102 "cd /tmp/siren_run; \
    export OMP_NUM_THREADS=4; \
    export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src; \
    $REMOTE_PYTHON -m torch.distributed.run \
    --nproc_per_node=1 \
    --nnodes=$NNODES \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_microhubert.py \
    --audio_dir data/audio \
    --features_dir data/features_hubert_layer9 \
    --output_dir checkpoints/microhubert \
    --epochs 100 \
    --resume checkpoints/microhubert/microhubert_ep45.pt \
    --lr 3e-4" &

REMOTE_PID=$!
echo "Launched remote WORKER (PID $REMOTE_PID)"

# Launch Local Master (Rank 0)
# Listens on localhost:29500
export OMP_NUM_THREADS=4
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
$LOCAL_PYTHON -m torch.distributed.run \
    --nproc_per_node=1 \
    --nnodes=$NNODES \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_microhubert.py \
    --audio_dir data/audio \
    --features_dir data/features_hubert_layer9 \
    --output_dir checkpoints/microhubert \
    --epochs 100 \
    --resume checkpoints/microhubert/microhubert_ep45.pt \
    --lr 3e-4

# Cleanup
kill $REMOTE_PID
pkill -f "ssh -p 2222 -N -f -R"
