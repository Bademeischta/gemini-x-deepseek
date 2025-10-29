import torch
import json
import os
from torch_geometric.data import Dataset
from scripts.graph_utils import fen_to_graph_data
from scripts.move_utils import uci_to_index

class ChessGraphDataset(Dataset):
    """
    A memory-efficient PyTorch Geometric Dataset for chess positions.
    It reads .jsonl files line by line by indexing the file offsets,
    avoiding loading the entire dataset into memory.
    """
    def __init__(self, jsonl_paths: list, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.file_handles = [open(path, 'r') for path in jsonl_paths if os.path.exists(path)]
        self.line_offsets = self._index_files()

    def _index_files(self):
        """Creates an index of (file_handle_index, offset) for each line."""
        offsets = []
        for i, f in enumerate(self.file_handles):
            f.seek(0)
            offset = f.tell()
            for line in f:
                offsets.append((i, offset))
                offset = f.tell()
        return offsets

    def len(self):
        return len(self.line_offsets)

    def get(self, idx):
        """
        Gets a single data sample by seeking to its offset, converting its FEN
        to a graph, and attaching pre-processed labels.
        """
        file_idx, offset = self.line_offsets[idx]
        f = self.file_handles[file_idx]
        f.seek(offset)
        line = f.readline()
        data_record = json.loads(line)

        # Convert FEN to graph
        graph_data = fen_to_graph_data(data_record['fen'])

        # Attach pre-processed target labels
        graph_data.y = torch.tensor([data_record.get('value', 0.0)], dtype=torch.float32)
        graph_data.policy_target = torch.tensor(uci_to_index(data_record.get('policy_target', '')), dtype=torch.long)
        graph_data.tactic_flag = torch.tensor([data_record.get('tactic_flag', 0.0)], dtype=torch.float32)
        graph_data.strategic_flag = torch.tensor([data_record.get('strategic_flag', 0.0)], dtype=torch.float32)

        return graph_data

    def __del__(self):
        """Ensures file handles are closed when the object is destroyed."""
        for f in self.file_handles:
            f.close()

if __name__ == '__main__':
    print("--- Testing Memory-Efficient ChessGraphDataset ---")

    # Create dummy .jsonl files
    dummy_data = [
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4", "tactic_flag": 1.0},
        {"fen": "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11", "value": -0.5, "policy_target": "a7a5", "strategic_flag": 1.0}
    ]
    test_path = "test_data.jsonl"
    with open(test_path, 'w') as f:
        for item in dummy_data:
            f.write(json.dumps(item) + '\n') # Correct newline character

    # Test dataset
    dataset = ChessGraphDataset(jsonl_paths=[test_path, "non_existent_file.jsonl"])
    print(f"Dataset created successfully. Length: {len(dataset)}")
    assert len(dataset) == 2

    # Test `get` method for the second sample
    sample = dataset.get(1)
    print(f"\nTesting sample 1: {sample}")

    assert sample.y.item() == -0.5
    assert sample.policy_target.item() == uci_to_index("a7a5")
    assert sample.strategic_flag.item() == 1.0
    # Check for tactic_flag which is missing in the record, should default to 0.0
    assert 'tactic_flag' in sample
    assert sample.tactic_flag.item() == 0.0
    print("\nLabel attachment and default values are correct.")

    print("\nAll tests passed!")

    # Cleanup
    del dataset # Explicitly delete to trigger __del__ for file closing
    os.remove(test_path)
    print("\nCleaned up dummy files.")
