import torch
import pandas as pd
import json
from torch_geometric.data import Dataset
from scripts.graph_utils import fen_to_graph_data
import os

class ChessGraphDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset to load chess positions from .jsonl files,
    convert FEN strings to graphs, and attach target labels.
    """
    def __init__(self, jsonl_paths: list, transform=None, pre_transform=None):
        """
        Args:
            jsonl_paths (list): A list of paths to the .jsonl data files.
        """
        super().__init__(None, transform, pre_transform)
        self.data_list = self._load_data(jsonl_paths)

    def _load_data(self, jsonl_paths):
        """Loads and concatenates data from multiple .jsonl files."""
        data_frames = []
        for path in jsonl_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_json(path, lines=True)
                    data_frames.append(df)
                except ValueError:
                    print(f"Warning: Could not parse {path}. It might be empty or malformed.")
            else:
                print(f"Warning: Data file not found at {path}. Skipping.")

        if not data_frames:
            # Return an empty dataframe if no data could be loaded
            return pd.DataFrame()

        # Concatenate all dataframes into one
        concatenated_df = pd.concat(data_frames, ignore_index=True)
        # Convert dataframe to a list of dictionaries for faster access
        return concatenated_df.to_dict('records')

    def len(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_list)

    def get(self, idx):
        """
        Gets a single data sample, converts its FEN to a graph, and attaches labels.
        """
        # 1. Get the data record (FEN, value, policy, etc.)
        data_record = self.data_list[idx]

        # 2. Convert the FEN string to a graph data object
        fen = data_record['fen']
        graph_data = fen_to_graph_data(fen)

        # 3. Attach the target labels to the graph object
        # The main target for Graph Neural Networks is often `y`
        graph_data.y = torch.tensor([data_record.get('value', 0.0)], dtype=torch.float32)

        # Attach other labels as attributes
        graph_data.tactic_flag = torch.tensor([data_record.get('tactic_flag', 0.0)], dtype=torch.float32)
        graph_data.strategic_flag = torch.tensor([data_record.get('strategic_flag', 0.0)], dtype=torch.float32)

        # For policy, we will handle the UCI string to index mapping in the training loop
        # as it can be complex and depends on the final model output layer.
        graph_data.policy_target = data_record.get('policy_target', '')

        return graph_data

if __name__ == '__main__':
    print("--- Testing ChessGraphDataset ---")

    # Create dummy .jsonl files for testing
    dummy_data_puzzles = [
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4", "tactic_flag": 1.0, "strategic_flag": 0.0},
        {"fen": "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "value": 0.2, "policy_target": "f1b5", "tactic_flag": 1.0, "strategic_flag": 0.0}
    ]
    dummy_data_strategic = [
        {"fen": "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11", "value": -0.5, "policy_target": "a7a5", "tactic_flag": 0.0, "strategic_flag": 1.0}
    ]

    puzzle_path = "dummy_puzzles.jsonl"
    strategic_path = "dummy_strategic.jsonl"

    with open(puzzle_path, 'w') as f:
        for item in dummy_data_puzzles:
            f.write(json.dumps(item) + '\\n')

    with open(strategic_path, 'w') as f:
        for item in dummy_data_strategic:
            f.write(json.dumps(item) + '\\n')

    # --- Test Dataset Initialization ---
    dataset = ChessGraphDataset(jsonl_paths=[puzzle_path, strategic_path, "non_existent_file.jsonl"])

    print(f"Dataset created successfully.")
    print(f"Total number of samples: {len(dataset)}")
    assert len(dataset) == 3

    # --- Test `get` method ---
    print("\\n--- Testing get(idx=2) ---")
    third_sample = dataset.get(2)
    print(f"Graph object: {third_sample}")

    # Verify attached labels
    assert 'y' in third_sample
    assert 'tactic_flag' in third_sample
    assert 'strategic_flag' in third_sample
    assert 'policy_target' in third_sample

    print(f"Value (y): {third_sample.y.item():.2f}")
    print(f"Tactic Flag: {third_sample.tactic_flag.item()}")
    print(f"Strategic Flag: {third_sample.strategic_flag.item()}")
    print(f"Policy Target (UCI): {third_sample.policy_target}")

    assert third_sample.y.item() == -0.5
    assert third_sample.strategic_flag.item() == 1.0
    assert third_sample.policy_target == "a7a5"

    print("\\nAll tests passed!")

    # --- Cleanup ---
    os.remove(puzzle_path)
    os.remove(strategic_path)
    print("\\nCleaned up dummy files.")
