import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.model import RCNModel
from scripts.graph_utils import TOTAL_NODE_FEATURES, NUM_EDGE_FEATURES, fen_to_graph_data
import config

class TestRCNModel(unittest.TestCase):

    def test_model_forward_pass(self):
        """Tests that the model's forward pass executes without errors and returns correct shapes."""
        model = RCNModel(
            in_channels=TOTAL_NODE_FEATURES,
            out_channels=config.MODEL_OUT_CHANNELS,
            num_edge_features=NUM_EDGE_FEATURES
        )
        model.eval()

        # Create a sample graph data object
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        graph_data = fen_to_graph_data(fen)

        # The model expects a batch, so we create one
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([graph_data])

        with torch.no_grad():
            value, policy_logits, tactic_flag, strategic_flag = model(batch)

        # Check output shapes
        self.assertEqual(value.shape, (1, 1))
        self.assertEqual(policy_logits[0].shape, (1, 64)) # from_sq
        self.assertEqual(policy_logits[1].shape, (1, 64)) # to_sq
        self.assertEqual(policy_logits[2].shape, (1, 4))  # promo
        self.assertEqual(tactic_flag.shape, (1, 1))
        self.assertEqual(strategic_flag.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
