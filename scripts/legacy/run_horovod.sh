#!/bin/bash
# scripts/run_horovod.sh
# Launch Horovod distributed training across local and remote machines

LOCAL_PYTHON="/home/sperm/siren/SIREN/.venv/bin/python3"
REMOTE_PYTHON="/home/kubs/siren_dist/.venv/bin/python3"

# Create hostfile
echo "localhost:1" > /tmp/horovod_hostfile
echo "192.168.0.102:1" >> /tmp/horovod_hostfile

# Run horovodrun
# -np 2: 2 processes total
# -H hostfile: using hostfile
# --gloo: use Gloo backend (pure TCP, easier firewall)
# -p 2222: SSH port for remote

/home/sperm/siren/SIREN/.venv/bin/horovodrun -np 2 \\
    --gloo \
    --network-interface eth0 \
    --ssh-identity-file /home/sperm/.ssh/id_rsa \
    --ssh-port 2222 \
    -H localhost:1,192.168.0.102:1 \
    $LOCAL_PYTHON scripts/train_microhubert_horovod.py \
    --audio_dir data/audio \
    --features_dir data/features_hubert_layer9 \
    --output_dir checkpoints/microhubert \
    --epochs 100 \
    --resume checkpoints/microhubert/microhubert_ep45.pt \
    --lr 3e-4
