#!/usr/bin/env python3
import sys
import os

# Ensure the package is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_low_bitrate_codec.train import main

if __name__ == "__main__":
    main()
