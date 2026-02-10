#!/bin/bash
# scripts/setup_vpn_and_run.sh
# Establishes SSH VPN (L3 Tunnel) and launches PyTorch DDP
# Requires sudo passwords for both ends (hardcoded here for automation)

LOCAL_PASS="lipton123"
REMOTE_PASS="zaqzaq"

echo "=== Killing old processes ==="
pkill -f "ssh -w"
pkill -f torch

echo "=== Creating VPN Tunnel ==="
# Launch SSH tunnel in background
# -w 5:5 creates tun5 on local and tun5 on remote
# We use sudo locally to create tun device
# Remote user kubs must accept sudo without tty (sudo -S)

# Tricky part: local sudo needs to wrap ssh, but ssh creates tun. 
# actually root is needed to create the tun dev interfacing.
# Better: User runs this script with sudo? Or we use sudo inside.

# Let's try running ssh with sudo locally.
echo $LOCAL_PASS | sudo -S -p "" modprobe tun
echo $LOCAL_PASS | sudo -S -p "" ssh -p 2222 -i /home/sperm/.ssh/id_rsa -o StrictHostKeyChecking=no -f -w 0:0 kubs@192.168.0.102 "echo 'Tunnel connection kept open'; sleep 3600" &
SSH_PID=$!
echo "SSH Tunnel Process launched (BG)"
sleep 5
echo "=== Interfaces ==="
ip link show
echo "=================="

echo "=== Configuring Interfaces ==="

# Local Interface (tun0) -> 10.0.0.1
echo $LOCAL_PASS | sudo -S -p "" ip addr add 10.0.0.1/24 dev tun0
echo $LOCAL_PASS | sudo -S -p "" ip link set tun0 up

# Remote Interface (tun0) -> 10.0.0.2
ssh -p 2222 -o StrictHostKeyChecking=no kubs@192.168.0.102 "echo '$REMOTE_PASS' | sudo -S ip addr add 10.0.0.2/24 dev tun0; echo '$REMOTE_PASS' | sudo -S ip link set tun0 up"

echo "=== Verifying Connectivity ==="
ping -c 2 10.0.0.2
if [ $? -ne 0 ]; then
    echo "VPN Ping failed. Check permissions."
    exit 1
fi

echo "=== Launching DDP Training ==="
# LOCAL (Master) -> 10.0.0.1
# REMOTE (Worker) -> 10.0.0.2

# Configuration
MASTER_ADDR="10.0.0.1"
MASTER_PORT=29500
NNODES=2
REMOTE_PYTHON="python3"
LOCAL_PYTHON="/home/sperm/siren/SIREN/.venv/bin/python3"

# 1. Launch Remote Worker
# It connects to MASTER_ADDR (10.0.0.1) which is reachable via tun0
ssh -p 2222 -o StrictHostKeyChecking=no kubs@192.168.0.102 "cd /tmp/siren_run; \
    export OMP_NUM_THREADS=4; \
    export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src; \
    export GLOO_SOCKET_IFNAME=tun0; \
    export TP_SOCKET_IFNAME=tun0; \
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
echo "Launched Remote Worker (PID $REMOTE_PID)"

# 2. Launch Local Master
# Binds to 10.0.0.1 (VPN IP)
export OMP_NUM_THREADS=4
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export GLOO_SOCKET_IFNAME=tun0
export TP_SOCKET_IFNAME=tun0

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
pkill -f "ssh -w"
