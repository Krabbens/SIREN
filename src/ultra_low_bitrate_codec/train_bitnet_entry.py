#!/usr/bin/env python3
"""
Entry point for BitNet training (Ultra 200bps Large).
Usage: uv run train-bitnet
"""
import sys
import os
from ultra_low_bitrate_codec.train import main as train_main

def main():
    # Set default arguments for the official BitNet Large training
    # Users can override if they know what they are doing, but we provide defaults
    
    base_args = [
        "train-bitnet",
        "--config", "ultra_low_bitrate_codec/configs/ultra200bps_large.yaml",
        "--checkpoint_dir", "checkpoints/checkpoints_ultra200bps_large"
    ]
    
    # If user provided args, just pass them (or append/override?)
    # Simpler: just replace sys.argv if it's just the script name
    if len(sys.argv) == 1:
        sys.argv = base_args
    else:
        # If user passed flags, maybe we should respect them? 
        # But this command implies a specific Preset.
        # Let's fallback to base_args + user args (ignoring script name)
        sys.argv = base_args + sys.argv[1:]
        
    print(f"🚀 Launching BitNet Training: {' '.join(sys.argv)}")
    train_main()

if __name__ == "__main__":
    main()
