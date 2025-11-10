"""
This module defines memory-efficient PyTorch Datasets for loading chess data.
"""
import torch
import json
import os
from torch_geometric.data import Dataset as PyGDataset, Data
from torch.utils.data import Dataset as TorchDataset
from typing import List, Tuple, Any, Optional

from scripts.graph_utils import fen_to_graph_data
from scripts.move_utils import uci_to_policy_targets

class ChessGraphDataset(PyGDataset):
    """
    A memory-efficient PyG Dataset for chess positions from .jsonl files.

    This dataset reads files line-by-line by indexing file offsets at
    initialization, avoiding loading the entire dataset into memory. It is
    implemented as a context manager to ensure proper handling of file resources.

    Attributes:
        file_paths (List[str]): A list of paths to the .jsonl data files.
        file_handles (List): A list of opened file handles (when context is active).
        line_offsets (List[Tuple[int, int]]): A list of (file_index, byte_offset)
            tuples for each line in the dataset.
    """
    def __init__(self, jsonl_paths: List[str], transform: Optional[Any] = None, pre_transform: Optional[Any] = None):
        """
        Initializes the ChessGraphDataset.

        Args:
            jsonl_paths: A list of string paths to the .jsonl files.
            transform: A function/transform that takes in a PyG Data object and
                returns a transformed version.
            pre_transform: A function/transform that takes in a PyG Data object
                and returns a transformed version.
        """
        super().__init__(None, transform, pre_transform)
        self.file_paths = [p for p in jsonl_paths if os.path.exists(p)]
        self.file_handles: List[Any] = []
        self.line_offsets: List[Tuple[int, int]] = []
        self._opened: bool = False

    def __enter__(self) -> 'ChessGraphDataset':
        """Opens file handles and indexes lines when entering a 'with' block."""
        if not self._opened:
            self.file_handles = [open(p, 'r') for p in self.file_paths]
            self.line_offsets = self._index_files()
            self._opened = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Closes file handles when exiting a 'with' block."""
        self.close()
        return False  # Do not suppress exceptions

    def close(self) -> None:
        """Closes all open file handles."""
        if self._opened:
            for f in self.file_handles:
                if not f.closed:
                    f.close()
            self.file_handles = []
            self._opened = False

    def _index_files(self) -> List[Tuple[int, int]]:
        """Creates an index of (file_handle_index, offset) for each line."""
        if not self.file_handles:
            return []
        offsets: List[Tuple[int, int]] = []
        for i, f in enumerate(self.file_handles):
            f.seek(0)
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append((i, offset))
        return offsets

    def len(self) -> int:
        """Returns the total number of samples in the dataset."""
        if not self._opened and not self.line_offsets:
            self._enter_and_index()
        return len(self.line_offsets)

    def get(self, idx: int) -> Data:
        """
        Gets a single data sample.

        This method seeks to the pre-indexed file offset, reads the line,
        converts the FEN to a graph, and attaches the target labels.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A PyG Data object representing the chess position.

        Raises:
            RuntimeError: If the dataset is accessed before its context is entered.
        """
        if not self._opened:
            raise RuntimeError("Dataset must be opened using a 'with' statement.")

        file_idx, offset = self.line_offsets[idx]
        f = self.file_handles[file_idx]
        f.seek(offset)
        line = f.readline()
        data_record = json.loads(line)

        graph_data = fen_to_graph_data(data_record['fen'])

        policy_targets = uci_to_policy_targets(data_record.get('policy_target', ''))
        graph_data.y = torch.tensor([data_record.get('value', 0.0)], dtype=torch.float32)

        # Validate targets - set ALL to -1 if any invalid
        from_sq = policy_targets['from']
        to_sq = policy_targets['to']
        promo = policy_targets.get('promo', -1)

        # Check if move targets are valid
        if from_sq < 0 or from_sq >= 64 or to_sq < 0 or to_sq >= 64:
            # Invalid move - set all to ignore_index
            from_sq, to_sq, promo = -1, -1, -1

        graph_data.policy_target_from = torch.tensor(from_sq, dtype=torch.long)
        graph_data.policy_target_to = torch.tensor(to_sq, dtype=torch.long)
        graph_data.policy_target_promo = torch.tensor(promo, dtype=torch.long)
        graph_data.tactic_flag = torch.tensor([data_record.get('tactic_flag', 0.0)], dtype=torch.float32)
        graph_data.strategic_flag = torch.tensor([data_record.get('strategic_flag', 0.0)], dtype=torch.float32)

        return graph_data

    def _enter_and_index(self) -> None:
        """Internal helper to allow len() to work before 'with'."""
        if not self._opened:
            temp_handles = [open(p, 'r') for p in self.file_paths]
            original_handles = self.file_handles
            self.file_handles = temp_handles

            self.line_offsets = self._index_files()

            for f in temp_handles:
                f.close()

            self.file_handles = original_handles

    def __del__(self) -> None:
        """Ensures file handles are closed when the object is destroyed."""
        self.close()

class DatasetWrapper(TorchDataset):
    """
    A wrapper to make PyG datasets compatible with `torch.utils.data.random_split`.
    """
    def __init__(self, base_dataset: ChessGraphDataset, indices: List[int]):
        """
        Initializes the DatasetWrapper.

        Args:
            base_dataset: The underlying `ChessGraphDataset`.
            indices: A list of indices representing the subset of the dataset.
        """
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        original_idx = self.indices[idx]
        return self.base_dataset[original_idx]

    def __enter__(self) -> 'DatasetWrapper': return self
    def __exit__(self, *args: Any) -> None: pass

if __name__ == '__main__':
    # This block is for self-contained validation of the script's logic.
    print("--- Testing Memory-Efficient ChessGraphDataset ---")

    dummy_data = [{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4"},
                  {"fen": "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11", "value": -0.5, "policy_target": "a7a5"},
                  {"fen": "8/k7/8/8/8/8/p7/K7 b - - 1 1", "value": -1.0, "policy_target": "a2a1q"}]
    test_path = "test_data.jsonl"
    with open(test_path, 'w') as f:
        for item in dummy_data: f.write(json.dumps(item) + '\n')

    dataset_standalone = ChessGraphDataset(jsonl_paths=[test_path])
    assert len(dataset_standalone) == 3
    dataset_standalone.close()
    print("len() works correctly before entering 'with' block.")

    with ChessGraphDataset(jsonl_paths=[test_path]) as dataset:
        print(f"Dataset opened. Length: {len(dataset)}")
        assert len(dataset) == 3

        sample1 = dataset.get(1)
        expected_targets1 = uci_to_policy_targets("a7a5")
        assert sample1.policy_target_from.item() == expected_targets1['from']

        sample2 = dataset.get(2)
        expected_targets2 = uci_to_policy_targets("a2a1q")
        assert sample2.policy_target_promo.item() == 3
        print("Sample retrieval and target generation are correct.")

    print("\nAll tests passed!")
    os.remove(test_path)
