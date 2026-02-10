import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

try:
    import triton
    print(f"Triton version: {triton.__version__}")
except ImportError:
    print("Triton NOT installed")

try:
    from ultra_low_bitrate_codec.models.bitlinear import BitLinear, HAS_TRITON
    print(f"HAS_TRITON in bitlinear.py: {HAS_TRITON}")
    
    if HAS_TRITON:
        print("Attempting to run BitLinear with Triton...")
        layer = BitLinear(256, 256).cuda()
        x = torch.randn(16, 100, 256).cuda()
        y = layer(x)
        print("Success! Output shape:", y.shape)
    else:
        print("HAS_TRITON is False. Check imports.")
        
except Exception as e:
    print(f"Error: {e}")
