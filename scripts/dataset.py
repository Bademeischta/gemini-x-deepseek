import torch
import json
import os
from torch_geometric.data import Dataset as PyGDataset
from torch.utils.data import Dataset as TorchDataset
from scripts.graph_utils import fen_to_graph_data
from scripts.move_utils import uci_to_policy_targets

class ChessGraphDataset(PyGDataset):
    """
    A memory-efficient PyTorch Geometric Dataset for chess positions.
    It reads .jsonl files line by line by indexing the file offsets,
    avoiding loading the entire dataset into memory.

    This class is implemented as a context manager to ensure proper file handle cleanup.
    """
    def __init__(self, jsonl_paths: list, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.file_paths = [p for p in jsonl_paths if os.path.exists(p)]
        self.file_handles = []
        self.line_offsets = []
        self._opened = False

    def __enter__(self):
        if not self._opened:
            self.file_handles = [open(p, 'r') for p in self.file_paths]
            self.line_offsets = self._index_files()
            self._opened = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False # Do not suppress exceptions

    def close(self):
        if self._opened:
            for f in self.file_handles:
                if not f.closed:
                    f.close()
            self.file_handles = []
            self._opened = False

    def _index_files(self):
        """Creates an index of (file_handle_index, offset) for each line."""
        if not self.file_handles:
            return []
        offsets = []
        for i, f in enumerate(self.file_handles):
            f.seek(0)
            # Use a while loop with tell() before readline() for accurate offsets
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append((i, offset))
        return offsets

    def len(self):
        if not self._opened:
            # Fallback for simple len() calls before context is entered
            self._enter_and_index()
        return len(self.line_offsets)

    def get(self, idx):
        """
        Gets a single data sample by seeking to its offset, converting its FEN
        to a graph, and attaching pre-processed labels.
        """
        if not self._opened:
            raise RuntimeError("Dataset must be opened using a 'with' statement before accessing items.")

        file_idx, offset = self.line_offsets[idx]
        f = self.file_handles[file_idx]
        f.seek(offset)
        line = f.readline()
        data_record = json.loads(line)

        # Convert FEN to graph
        graph_data = fen_to_graph_data(data_record['fen'])

        # Attach pre-processed target labels
        policy_targets = uci_to_policy_targets(data_record.get('policy_target', ''))
        graph_data.y = torch.tensor([data_record.get('value', 0.0)], dtype=torch.float32)
        graph_data.policy_target_from = torch.tensor(policy_targets['from'], dtype=torch.long)
        graph_data.policy_target_to = torch.tensor(policy_targets['to'], dtype=torch.long)
        graph_data.tactic_flag = torch.tensor([data_record.get('tactic_flag', 0.0)], dtype=torch.float32)
        graph_data.strategic_flag = torch.tensor([data_record.get('strategic_flag', 0.0)], dtype=torch.float32)

        return graph_data

    def _enter_and_index(self):
        """Internal helper to allow len() to work before 'with'."""
        if not self._opened:
            self.file_handles = [open(p, 'r') for p in self.file_paths]
            self.line_offsets = self._index_files()
            # Don't set _opened to True, so 'with' block can still manage it properly

    def __del__(self):
        """Ensures file handles are closed when the object is destroyed."""
        self.close()

# Wrapper for compatibility with torch.utils.data.random_split
class DatasetWrapper(TorchDataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Retrieve the actual index from our subset of indices
        original_idx = self.indices[idx]
        return self.base_dataset[original_idx]

    def __enter__(self): return self
    def __exit__(self, *args): pass

if __name__ == '__main__':
    print("--- Testing Memory-Efficient ChessGraphDataset ---")

    dummy_data = [{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4"},
                  {"fen": "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11", "value": -0.5, "policy_target": "a7a5"}]
    test_path = "test_data.jsonl"
    with open(test_path, 'w') as f:
        for item in dummy_data: f.write(json.dumps(item) + '\n')

    # Test len() before entering context
    dataset_standalone = ChessGraphDataset(jsonl_paths=[test_path])
    assert len(dataset_standalone) == 2
    dataset_standalone.close() # Manually close after len()
    print("len() works correctly before entering 'with' block.")

    with ChessGraphDataset(jsonl_paths=[test_path]) as dataset:
        print(f"Dataset opened. Length: {len(dataset)}")
        assert len(dataset) == 2
        sample = dataset.get(1)
        expected_targets = uci_to_policy_targets("a7a5")
        assert sample.policy_target_from.item() == expected_targets['from']
        assert sample.policy_target_to.item() == expected_targets['to']
        print("Sample retrieval and target generation are correct.")

    print("\nAll tests passed!")
    os.remove(test_path)
