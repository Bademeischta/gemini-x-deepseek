import unittest
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.dataset import ChessGraphDataset, cleanup_worker_files
from scripts.uci_index import uci_to_index_4096

class TestChessGraphDataset(unittest.TestCase):
    test_path = "test_data.jsonl"
    dummy_data = [
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4"},
        {"fen": "8/k7/8/8/8/8/p7/K7 b - - 1 1", "value": -1.0, "policy_target": "a2a1q"}
    ]

    def setUp(self):
        """Create a dummy .jsonl file before each test."""
        with open(self.test_path, 'w') as f:
            for item in self.dummy_data:
                f.write(json.dumps(item) + '\n')

    def tearDown(self):
        """Remove the dummy .jsonl file after each test."""
        if os.path.exists(self.test_path):
            os.remove(self.test_path)
        # Clean up any residual file handles from tests
        cleanup_worker_files()


    def test_initialization_and_length(self):
        """Tests that the dataset initializes correctly and reports the right length."""
        dataset = ChessGraphDataset(jsonl_paths=[self.test_path])
        self.assertEqual(len(dataset), 2)

    def test_item_retrieval_and_targets(self):
        """Tests that data samples are retrieved correctly with the new policy target."""
        dataset = ChessGraphDataset(jsonl_paths=[self.test_path])

        # Test standard move
        sample1 = dataset.get(0)
        expected_index1 = uci_to_index_4096("e2e4")
        self.assertEqual(sample1.policy_target.item(), expected_index1)

        # Test promotion move
        # Note: The 4096 representation does not distinguish promotion pieces.
        # 'a2a1q', 'a2a1r', etc., will all map to the same index.
        sample2 = dataset.get(1)
        expected_index2 = uci_to_index_4096("a2a1q")
        self.assertEqual(sample2.policy_target.item(), expected_index2)

    def test_handles_empty_and_nonexistent_files(self):
        """Tests that the dataset handles empty or missing files gracefully."""
        # Test with a non-existent path
        dataset_nonexistent = ChessGraphDataset(jsonl_paths=["nonexistent.jsonl"])
        self.assertEqual(len(dataset_nonexistent), 0)

        # Test with an empty file
        empty_path = "empty.jsonl"
        open(empty_path, 'w').close()
        dataset_empty = ChessGraphDataset(jsonl_paths=[empty_path])
        self.assertEqual(len(dataset_empty), 0)
        os.remove(empty_path)

if __name__ == '__main__':
    unittest.main()
