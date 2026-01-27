import os
import sys
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_logs(log_dir):
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print("No event files found.")
        return

    print("Found event files:", event_files)
    for ef in event_files:
        print(f"\nProcessing {os.path.basename(ef)}...")
        try:
            ea = EventAccumulator(ef)
            ea.Reload()
            tags = ea.Tags()['scalars']
            print("Scalar tags:", tags)
            
            for tag in tags:
                events = ea.Scalars(tag)
                if not events: continue
                # Print first, middle, last
                print(f"  {tag}: Start={events[0].value:.4f}, End={events[-1].value:.4f} (Steps: {len(events)})")
                
        except Exception as e:
            print(f"Error reading {ef}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_tb.py <log_dir>")
    else:
        extract_logs(sys.argv[1])
