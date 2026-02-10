#!/bin/bash
# scripts/setup_socat_vpn_and_run.sh
# Uses socat with SUDO_ASKPASS to avoid stdin corruption

LOCAL_PASS="lipton123"
REMOTE_PASS="zaqzaq"

echo "=== Killing old processes ==="
pkill socat
pkill -f torch

echo "=== Creating Password Helpers ==="
# Local
echo "#!/bin/sh" > $(pwd)/local_pass.sh
echo "echo '$LOCAL_PASS'" >> $(pwd)/local_pass.sh
chmod +x $(pwd)/local_pass.sh

# Remote
ssh -p 2222 -i /home/sperm/.ssh/id_rsa -o StrictHostKeyChecking=no kubs@192.168.0.102 "echo '#!/bin/sh' > ~/pass.sh; echo 'echo \"$REMOTE_PASS\"' >> ~/pass.sh; chmod +x ~/pass.sh"

echo "=== Creating Socat VPN ==="
# Remote command
REMOTE_CMD="export SUDO_ASKPASS=~/pass.sh; sudo -A socat TUN:10.0.0.2/24,tun-name=tun0,iff-no-pi,up -"

# Local command
export SUDO_ASKPASS=$(pwd)/local_pass.sh
sudo -A socat \
    TUN:10.0.0.1/24,tun-name=tun0,iff-no-pi,up \
    EXEC:"ssh -p 2222 -i /home/sperm/.ssh/id_rsa -o StrictHostKeyChecking=no kubs@192.168.0.102 \"$REMOTE_CMD\"",nofork &

VPN_PID=$!
echo "VPN PID: $VPN_PID"
sleep 5

echo "=== Connectivity Check ==="
echo "Local Interfaces:"
ip link show tun0
ping -c 2 10.0.0.2

if [ $? -ne 0 ]; then
    echo "VPN failed."
    rm $(pwd)/local_pass.sh
    exit 1
fi

echo "=== Launching DDP ==="

# Launch DDP using VPN IPs
MASTER_ADDR="10.0.0.1"
MASTER_PORT=29500
NNODES=2
REMOTE_PYTHON="python3"
LOCAL_PYTHON="/home/sperm/siren/SIREN/.venv/bin/python3"

# 1. Launch Remote Worker
ssh -p 2222 -i /home/sperm/.ssh/id_rsa -o StrictHostKeyChecking=no kubs@192.168.0.102 "cd /tmp/siren_run; \
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
kill $VPN_PID
rm $(pwd)/local_pass.sh
ssh -p 2222 -i /home/sperm/.ssh/id_rsa -o StrictHostKeyChecking=no kubs@192.168.0.102 "rm ~/pass.sh"
