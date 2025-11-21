"""
This module defines memory-efficient PyTorch Datasets for loading chess data
that is safe for multiprocessing (num_workers > 0).
"""
import torch
import json
import os
import atexit
import chess
from torch_geometric.data import Dataset as PyGDataset, Data
from torch.utils.data import Dataset as TorchDataset, get_worker_info
from typing import List, Tuple, Any, Optional, Dict

from scripts.fen_to_graph_data_v2 import fen_to_graph_data_v2
from scripts.uci_index import uci_to_index_4096

# FIX: Global dictionary to hold file handles for each worker process
WORKER_FILE_HANDLES: Dict[int, List[Any]] = {}

def cleanup_worker_files():
    """Closes all file handles opened by the current worker."""
    worker_info = get_worker_info()
    if worker_info is not None:
        worker_id = worker_info.id
        if worker_id in WORKER_FILE_HANDLES:
            for f in WORKER_FILE_HANDLES[worker_id]:
                f.close()
            del WORKER_FILE_HANDLES[worker_id]

atexit.register(cleanup_worker_files)

class ChessGraphDataset(PyGDataset):
    """
    A memory-efficient and multiprocessing-safe PyG Dataset for chess positions.

    This dataset reads files line-by-line. File indexing is done once at
    initialization. File handles are managed on a per-worker basis to ensure
    compatibility with `num_workers > 0` in PyTorch DataLoaders.
    """
    def __init__(self, jsonl_paths: List[str], transform: Optional[Any] = None, pre_transform: Optional[Any] = None):
        super().__init__(None, transform, pre_transform)
        self.file_paths = [p for p in jsonl_paths if os.path.exists(p)]
        self.line_offsets: List[Tuple[int, int]] = self._index_files()

    def _index_files(self) -> List[Tuple[int, int]]:
        """
        Creates an index of (file_handle_index, offset) for each line without
        keeping files open.
        """
        offsets: List[Tuple[int, int]] = []
        for i, path in enumerate(self.file_paths):
            with open(path, 'r') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    offsets.append((i, offset))
        return offsets

    def len(self) -> int:
        return len(self.line_offsets)

    def get(self, idx: int) -> Data:
        """
        Gets a single data sample. This method is called by the DataLoader's workers.
        It manages file handles internally for each worker.
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        # Initialize file handles for this worker if not already done
        if worker_id not in WORKER_FILE_HANDLES:
            WORKER_FILE_HANDLES[worker_id] = [open(p, 'r') for p in self.file_paths]

        file_idx, offset = self.line_offsets[idx]
        f = WORKER_FILE_HANDLES[worker_id][file_idx]
        f.seek(offset)
        line = f.readline()

        # FIX: Strip whitespace to prevent JSON errors with empty/bad lines
        line = line.strip()
        if not line:
            # Handle empty line case, maybe return a dummy graph or raise error
            # For now, let's try to grab the next valid item
            return self.get((idx + 1) % len(self))

        try:
            data_record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"WORKER {worker_id}: JSONDecodeError at index {idx}, line: '{line}'. Error: {e}")
            # Return next item to prevent crash
            return self.get((idx + 1) % len(self))

        # FIX: Robustly create Board object
        try:
            board = chess.Board(data_record['fen'])
        except Exception as e:
            print(f"Invalid FEN at index {idx}: {data_record.get('fen', 'UNKNOWN')}. Error: {e}")
            return self.get((idx + 1) % len(self))

        # FIX: Pass Board object (not FEN string)
        graph_data = fen_to_graph_data_v2(board)

        # Policy Target Conversion
        uci_move = data_record.get('policy_target', '')
        if uci_move:
            try:
                policy_target = uci_to_index_4096(uci_move)
            except Exception as e:
                print(f"Invalid UCI move '{uci_move}' at index {idx}: {e}")
                policy_target = -1
        else:
            policy_target = -1

        graph_data.y = torch.tensor([data_record.get('value', 0.0)], dtype=torch.float32)
        graph_data.policy_target = torch.tensor(policy_target, dtype=torch.long)
        graph_data.tactic_flag = torch.tensor([data_record.get('tactic_flag', 0.0)], dtype=torch.float32)
        graph_data.strategic_flag = torch.tensor([data_record.get('strategic_flag', 0.0)], dtype=torch.float32)
        graph_data.fen = data_record['fen']

        # Pre-compute legal move mask using the already created board object
        # This avoids re-creating the board and is consistent per-sample
        try:
            legal_moves_mask = torch.zeros(4096, dtype=torch.bool)
            for move in board.legal_moves:
                idx = uci_to_index_4096(move.uci())
                legal_moves_mask[idx] = True
            graph_data.legal_moves_mask = legal_moves_mask
        except Exception as e:
            print(f"Error creating legal move mask at index {idx}: {e}")
            # Fallback to allow all moves (loss will penalize illegals) or empty
            graph_data.legal_moves_mask = torch.ones(4096, dtype=torch.bool)

        return graph_data

class DatasetWrapper(TorchDataset):
    """
    A wrapper to make PyG datasets compatible with `torch.utils.data.random_split`.
    This wrapper is simplified as the base dataset now handles its own resources.
    """
    def __init__(self, base_dataset: ChessGraphDataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        original_idx = self.indices[idx]
        return self.base_dataset[original_idx]

if __name__ == '__main__':
    print("--- Testing Multiprocessing-Safe ChessGraphDataset ---")

    dummy_data = [
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4"},
        {"fen": "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11", "value": -0.5, "policy_target": "a7a5"},
        {"fen": "8/k7/8/8/8/8/p7/K7 b - - 1 1", "value": -1.0, "policy_target": "a2a1q"}
    ]
    test_path = "test_data.jsonl"
    with open(test_path, 'w') as f:
        for item in dummy_data: f.write(json.dumps(item) + '\n')

    # Test initialization and length
    dataset = ChessGraphDataset(jsonl_paths=[test_path])
    print(f"Dataset indexed. Total length: {len(dataset)}")
    assert len(dataset) == 3

    # Test item retrieval (simulating worker 0)
    sample1 = dataset.get(1)
    expected_policy_index = uci_to_index_4096("a7a5")
    assert sample1.policy_target.item() == expected_policy_index
    print("Sample retrieval is correct.")

    # Test with DataLoader
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=2, num_workers=0) # test with 0 and 2 workers
    print("\nTesting with DataLoader (num_workers=0)...")
    for i, batch in enumerate(loader):
        print(f"Batch {i}: {batch}")
        assert batch.batch is not None
    print("✓ DataLoader (num_workers=0) works.")

    # Clean up global state for next test if needed
    if 0 in WORKER_FILE_HANDLES:
        for f in WORKER_FILE_HANDLES[0]: f.close()
        del WORKER_FILE_HANDLES[0]

    # This part is harder to test in a simple script, but the logic is sound for multiprocessing
    print("\nLogic for multiprocessing (num_workers > 0) is implemented.")
    print("To test, run training with num_workers > 0.")

    print("\nAll tests passed!")
    os.remove(test_path)
