#!/bin/bash
# Forward ports 60540-60560 from Windows LAN to WSL
# Run this on the LOCAL machine (sperm) inside WSL

WSL_IP="172.17.140.118"

echo "Starting port forwarders..."

for port in $(seq 60540 60560); do
    # Kill any existing socat on this port
    pkill -f "socat.*:$port" 2>/dev/null
    
    # Start socat to forward from 0.0.0.0:port to WSL_IP:port
    socat TCP-LISTEN:$port,fork,reuseaddr,bind=0.0.0.0 TCP:$WSL_IP:$port &
    echo "Forwarding port $port"
done

# Also forward NCCL dynamic ports range
for port in $(seq 44100 44200); do
    pkill -f "socat.*:$port" 2>/dev/null
    socat TCP-LISTEN:$port,fork,reuseaddr,bind=0.0.0.0 TCP:$WSL_IP:$port &
done

echo "Port forwarders started. Press Ctrl+C to stop."
wait
