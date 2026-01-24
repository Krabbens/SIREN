import matplotlib.pyplot as plt
import re
import sys
import numpy as np

def parse_log_and_plot(log_path, output_path):
    steps = []
    losses = []
    stft_losses = []
    mel_losses = []
    ent_losses = []
    bps_values = []
    
    val_steps = []
    val_mels = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_segment_start_step = 0
    
    for line in lines:
        # Parse training steps
        # Step 7800: {'loss': '83.19', 'stft': '3.69', 'mel': '0.71', 'ent': '20.52', 'bps': '207'}
        match = re.search(r"Step (\d+): \{'loss': '([\d\.]+)', 'stft': '([\d\.]+)', 'mel': '([\d\.]+)', 'ent': '([\d\.]+)', 'bps': '([\d\.]+)'\}", line)
        if match:
            step = int(match.group(1))
            
            # Handle restart/reset logic roughly (if step counts reset, which they might have in previous attempts)
            # In this specific log, it looks like it restarts but the step count might be preserved or reset.
            # Looking at the file content, it has "=== SIREN v2 from step 0 ===" multiple times.
            # We want to plot the LATEST training run effectively, or all of them concatenated if they are continuous in reality.
            # However, seeing "Step 100" multiple times suggests restarts.
            # Let's simple plot ALL points but color them by segment or just scatter plot them.
            
            steps.append(step)
            losses.append(float(match.group(2)))
            stft_losses.append(float(match.group(3)))
            mel_losses.append(float(match.group(4)))
            ent_losses.append(float(match.group(5)))
            bps_values.append(float(match.group(6)))
            continue

        # Parse validation
        #   Validation MEL: 0.682
        val_match = re.search(r"Validation MEL: ([\d\.]+)", line)
        if val_match:
            # Associate with the most recent step
            if steps:
                val_steps.append(steps[-1])
                val_mels.append(float(val_match.group(1)))

    # Focus on the most recent continuous run? 
    # The log has multiple runs. Let's just plot everything linearly as they appear in the file
    # for x-axis, we can use an index or try to detect the "latest" run.
    # The user is interested in the current progress. The last run starts around line 124 "=== SIREN v2 from step 9000 ==="
    # Actually line 125 shows "Step 9200", so it resumed.
    
    # Let's plot the last 5000 steps or something, or better yet, plot everything using a simple index to avoid "Step 100" from run 1 overlapping with "Step 100" from run 2.
    # Actually, the step numbers in the log seem to be increasing in the last block (9200 -> 14400).
    
    # Let's find the start of the last run to make it clean
    last_run_start_index = 0
    for i, line in enumerate(lines):
        if "=== SIREN v2" in line:
            last_run_start_index = i
            
    # Re-parse only from the last meaningful start or just parse everything and if step < prev_step, it's a new run.
    # But wait, looking at the log provided in context:
    # 2: === SIREN v2 from step 0 ===
    # 13: === SIREN v2 from step 0 ===
    # 124: === SIREN v2 from step 9000 ===
    # So the last chunk is from line 124 onwards.
    
    # We will filter to keep only data from the latest run (or the run that led to it if it was a resume).
    # Actually, the run starting at line 13 goes from Step 100 to Step 9000.
    # Then line 124 resumes at Step 9200.
    # We should stitch these two together (Run 2 and Run 3). Run 1 (lines 2-12) seems short/aborted.
    
    # Let's just collect all data. If we see a large jump backwards in step (e.g. 9000 -> 100), we treat it as a new attempt.
    # But here we have 9000 -> 9200 (resume), so that's good.
    # The first run (lines 3-11) goes to 800. The second (14-122) goes to 9100. Then (125-end) goes 9200+.
    # We should probably exclude the first failed run (steps 100-800) if it overlaps.
    
    # Filter: If we have duplicate steps, keep the later ones.
    data_dict = {} # step -> values
    for s, l, st, m, e, b in zip(steps, losses, stft_losses, mel_losses, ent_losses, bps_values):
        data_dict[s] = (l, st, m, e, b)
        
    sorted_steps = sorted(data_dict.keys())
    # Reconstruct lists
    steps = sorted_steps
    losses = [data_dict[s][0] for s in steps]
    stft_losses = [data_dict[s][1] for s in steps]
    mel_losses = [data_dict[s][2] for s in steps]
    ent_losses = [data_dict[s][3] for s in steps]
    bps_values = [data_dict[s][4] for s in steps]
    
    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Losses
    ax1.plot(steps, losses, label='Total Loss', color='black', linewidth=1)
    ax1.plot(steps, [l*10 for l in mel_losses], label='Mel Loss (x10)', color='red', alpha=0.7)
    ax1.plot(steps, [l*10 for l in stft_losses], label='STFT Loss (x10)', color='blue', alpha=0.7)
    
    # Overlay Validation Mel on Plot 1
    # We need to map val_steps to the filtered steps if strictly needed, but scatter plot is fine.
    # Filter val_mels to only those in our sorted_steps range
    clean_val_steps = []
    clean_val_mels = []
    for s, v in zip(val_steps, val_mels):
        if s in data_dict:
            clean_val_steps.append(s)
            clean_val_mels.append(v)
            
    ax1.scatter(clean_val_steps, [v*10 for v in clean_val_mels], color='magenta', zorder=5, label='Val Mel (x10)')
    
    ax1.set_title(f'Training Metrics (BitNet Large) - Latest Step: {steps[-1]}')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Bitrate and Entropy
    ax2.plot(steps, bps_values, label='Bitrate (bps)', color='green')
    ax2.set_ylabel('Bits Per Second')
    ax2.set_xlabel('Step')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, ent_losses, label='Entropy Loss', color='orange', linestyle='--')
    ax2_twin.set_ylabel('Entropy Loss')
    ax2_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    # Print latest stats
    print(f"Latest Step: {steps[-1]}")
    print(f"Total Loss: {losses[-1]}")
    print(f"Mel Loss: {mel_losses[-1]}")
    print(f"STFT Loss: {stft_losses[-1]}")
    print(f"Entropy Loss: {ent_losses[-1]}")
    print(f"BPS: {bps_values[-1]}")
    if clean_val_mels:
        print(f"Latest Val Mel: {clean_val_mels[-1]} (Step {clean_val_steps[-1]})")

if __name__ == "__main__":
    parse_log_and_plot(sys.argv[1], sys.argv[2])
