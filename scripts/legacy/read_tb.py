import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "checkpoints/microencoder_e2e/tensorboard"
event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
latest_event_file = max(event_files, key=os.path.getctime)

print(f"Reading {latest_event_file}...")
ea = EventAccumulator(latest_event_file)
ea.Reload()

tags = ea.Tags()['scalars']
print(f"Found tags: {tags}")

for tag in ['Loss/flow', 'Loss/mel', 'Loss/adv', 'Loss/disc', 'Loss/total']:
    if tag in tags:
        events = ea.Scalars(tag)
        last_event = events[-1]
        print(f"{tag}: {last_event.value:.4f} (step {last_event.step})")
