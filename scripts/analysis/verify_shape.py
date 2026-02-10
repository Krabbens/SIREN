import torch
import glob
import os
import random

files = glob.glob("data/flow_dataset/*.pt")
if not files:
    print("No files found in data/flow_dataset!")
    exit(1)

# Check a random file
f = random.choice(files)
d = torch.load(f)
print(f"Checking {f}...")
print("Keys:", d.keys())
print("Mel Shape:", d['mel'].shape)
print("Cond Shape:", d['cond'].shape)
print("Mel Min/Max:", d['mel'].min(), d['mel'].max()) # Check for Log values
