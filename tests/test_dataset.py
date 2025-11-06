import unittest
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.dataset import ChessGraphDataset
from scripts.move_utils import uci_to_policy_targets

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

    def test_context_manager_opens_and_closes_files(self):
        """Tests that the 'with' statement correctly manages file handles."""
        dataset = ChessGraphDataset(jsonl_paths=[self.test_path])
        self.assertFalse(dataset._opened, "Dataset should not be opened before 'with'.")
        self.assertEqual(len(dataset.file_handles), 0)

        with dataset:
            self.assertTrue(dataset._opened, "Dataset should be opened inside 'with'.")
            self.assertEqual(len(dataset.file_handles), 1)
            self.assertFalse(dataset.file_handles[0].closed)
            # Perform a basic operation
            self.assertEqual(len(dataset), 2)

        self.assertFalse(dataset._opened, "Dataset should be closed after 'with'.")
        self.assertEqual(len(dataset.file_handles), 0)

    def test_item_retrieval_and_targets(self):
        """Tests that data samples are retrieved correctly, including promotion targets."""
        with ChessGraphDataset(jsonl_paths=[self.test_path]) as dataset:
            # Test standard move
            sample1 = dataset.get(0)
            expected1 = uci_to_policy_targets("e2e4")
            self.assertEqual(sample1.policy_target_from.item(), expected1['from'])
            self.assertEqual(sample1.policy_target_to.item(), expected1['to'])
            self.assertEqual(sample1.policy_target_promo.item(), -1)

            # Test promotion move
            sample2 = dataset.get(1)
            expected2 = uci_to_policy_targets("a2a1q")
            self.assertEqual(sample2.policy_target_from.item(), expected2['from'])
            self.assertEqual(sample2.policy_target_to.item(), expected2['to'])
            self.assertEqual(sample2.policy_target_promo.item(), 3) # Queen

    def test_len_works_without_context(self):
        """Tests that len() can be called before the context is entered."""
        dataset = ChessGraphDataset(jsonl_paths=[self.test_path])
        # Calling len() should implicitly open, index, and close the files
        length = len(dataset)
        self.assertEqual(length, 2)
        # Ensure it cleaned up after itself
        self.assertFalse(dataset._opened)
        self.assertEqual(len(dataset.file_handles), 0)

    def test_handles_empty_and_nonexistent_files(self):
        """Tests that the dataset handles empty or missing files gracefully."""
        # Test with a non-existent path
        with ChessGraphDataset(jsonl_paths=["nonexistent.jsonl"]) as dataset:
            self.assertEqual(len(dataset), 0)

        # Test with an empty file
        empty_path = "empty.jsonl"
        open(empty_path, 'w').close()
        with ChessGraphDataset(jsonl_paths=[empty_path]) as dataset:
            self.assertEqual(len(dataset), 0)
        os.remove(empty_path)


if __name__ == '__main__':
    unittest.main()
