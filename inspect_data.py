import json
from scripts.move_utils import uci_to_policy_targets
import config

print("Inspecting first 100 samples for invalid moves...")

invalid_count = 0
with open(config.DATA_PUZZLES_PATH, 'r') as f:
    for i, line in enumerate(f):
        if i >= 100:
            break

        data = json.loads(line)
        targets = uci_to_policy_targets(data.get('policy_target', ''))

        if targets['from'] < 0 or targets['to'] < 0:
            invalid_count += 1
            print(f"Line {i}: Invalid move '{data.get('policy_target')}' -> {targets}")

print(f"\nFound {invalid_count} invalid moves in first 100 samples")
