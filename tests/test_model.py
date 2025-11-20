# tests/test_model.py
import unittest
import torch
import os
import sys
import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.model import RCNModel
from scripts.fen_to_graph_data_v2 import fen_to_graph_data_v2
import config

class TestRCNModel(unittest.TestCase):

    def test_model_forward_pass(self):
        """
        Tests that the model's forward pass executes with the new data structure
        and returns the correct shapes for the joint policy head.
        """
        # Initialize the model with the new, correct dimensions
        model = RCNModel(
            in_channels=15,  # From fen_to_graph_data_v2
            out_channels=config.MODEL_OUT_CHANNELS,
            num_edge_features=2  # From fen_to_graph_data_v2
        )
        model.eval()

        # Create a sample graph data object using the new builder
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        graph_data = fen_to_graph_data_v2(board)

        # The model expects a batch, so we create one
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([graph_data])

        with torch.no_grad():
            value, policy_logits, tactic_flag, strategic_flag = model(batch)

        # Check output shapes for the new architecture
        self.assertEqual(value.shape, (1, 1))
        self.assertEqual(policy_logits.shape, (1, 4096))  # Joint policy head
        self.assertEqual(tactic_flag.shape, (1, 1))
        self.assertEqual(strategic_flag.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
