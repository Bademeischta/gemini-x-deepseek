import unittest
from unittest.mock import patch
import torch
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import train

class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        # Create a tiny dummy dataset for a quick training run
        self.test_path = "e2e_test_data.jsonl"
        dummy_data = [
            {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4", "tactic_flag": 0.0, "strategic_flag": 1.0},
            {"fen": "8/k7/8/8/8/8/p7/K7 b - - 1 1", "value": -1.0, "policy_target": "a2a1q", "tactic_flag": 1.0, "strategic_flag": 0.0}
        ]
        with open(self.test_path, 'w') as f:
            for item in dummy_data:
                f.write(json.dumps(item) + '\n')

    def tearDown(self):
        if os.path.exists(self.test_path):
            os.remove(self.test_path)
        if os.path.exists("e2e_test_model.pth"):
            os.remove("e2e_test_model.pth")

    @patch('config.DATA_PUZZLES_PATH', "e2e_test_data.jsonl")
    @patch('config.DATA_STRATEGIC_PATH', "") # Use one file for simplicity
    @patch('config.MODEL_SAVE_PATH', "models/e2e_test_model.pth")
    @patch('config.NUM_EPOCHS', 1)
    @patch('config.BATCH_SIZE', 2)
    def test_short_training_run(self):
        """
        Tests that the training process can run for one epoch without errors.
        This is an integration test that touches many parts of the codebase.
        """
        try:
            # The train function will run with the patched config values
            train()
            # Check that a model was actually saved
            self.assertTrue(os.path.exists("models/e2e_test_model.pth"))
        except Exception as e:
            self.fail(f"End-to-end training test failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
