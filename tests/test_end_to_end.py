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
        # Create a dummy dataset sufficient for a test run
        self.test_path = "e2e_test_data.jsonl"
        dummy_data = [
            {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4", "tactic_flag": 0.0, "strategic_flag": 1.0},
            {"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "value": 0.1, "policy_target": "g1f3", "tactic_flag": 0.0, "strategic_flag": 0.0},
            {"fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 3", "value": 0.1, "policy_target": "f1c4", "tactic_flag": 0.0, "strategic_flag": 0.0},
            {"fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 4", "value": 0.1, "policy_target": "f6e4", "tactic_flag": 1.0, "strategic_flag": 0.0},
            {"fen": "rnbqk2r/pppp1ppp/5n2/4p3/1bB1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 2 5", "value": 0.2, "policy_target": "e1g1", "tactic_flag": 0.0, "strategic_flag": 0.0},
            {"fen": "r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 6", "value": 0.1, "policy_target": "e8g8", "tactic_flag": 0.0, "strategic_flag": 0.0},
            {"fen": "8/k7/8/8/8/8/p7/K7 b - - 1 1", "value": -1.0, "policy_target": "a2a1q", "tactic_flag": 1.0, "strategic_flag": 0.0},
            {"fen": "r1bq1rk1/ppppbppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 1 7", "value": 0.1, "policy_target": "a2a4", "tactic_flag": 0.0, "strategic_flag": 0.0}
        ]
        with open(self.test_path, 'w') as f:
            for item in dummy_data:
                f.write(json.dumps(item) + '\n')

    def tearDown(self):
        if os.path.exists(self.test_path):
            os.remove(self.test_path)
        if os.path.exists("models/e2e_test_model.pth"):
            os.remove("models/e2e_test_model.pth")
        if os.path.exists("models/e2e_test_checkpoint.pth"):
            os.remove("models/e2e_test_checkpoint.pth")
        # Clean up the 'models' directory if it's empty
        if os.path.exists("models") and not os.listdir("models"):
            os.rmdir("models")


    @patch('config.DATA_PUZZLES_PATH', "e2e_test_data.jsonl")
    @patch('config.DATA_STRATEGIC_PATH', "") # Use one file for simplicity
    @patch('config.MODEL_SAVE_PATH', "models/e2e_test_model.pth")
    @patch('config.TRAINING_CHECKPOINT_PATH', "models/e2e_test_checkpoint.pth")
    @patch('config.NUM_EPOCHS', 1)
    @patch('config.BATCH_SIZE', 4) # Use a valid batch size >= MIN_BATCH_SIZE
    @patch('config.TRAIN_TEST_SPLIT', 0.5) # Use a split to ensure val_loss is calculated
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
