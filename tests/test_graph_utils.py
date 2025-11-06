import unittest
import torch
import chess
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.graph_utils import fen_to_graph_data, EDGE_TYPE_TO_INT, NUM_EDGE_FEATURES

class TestGraphUtils(unittest.TestCase):

    def test_pin_edge_creation(self):
        """Tests that a pin edge is correctly identified."""
        fen_pin = "4q3/8/8/8/4R3/8/8/4K3 w - - 0 1"
        graph = fen_to_graph_data(fen_pin)
        board = chess.Board(fen_pin)

        sq_map = {sq: i for i, (sq, p) in enumerate(sorted(board.piece_map().items()))}
        pinner_idx = sq_map[chess.E8]
        pinned_idx = sq_map[chess.E4]

        found = False
        for i, (s, t) in enumerate(graph.edge_index.t().tolist()):
            if s == pinner_idx and t == pinned_idx:
                expected_one_hot = torch.nn.functional.one_hot(torch.tensor(EDGE_TYPE_TO_INT["PIN"]), num_classes=NUM_EDGE_FEATURES).float()
                if torch.equal(graph.edge_attr[i], expected_one_hot):
                    found = True
                    break

        self.assertTrue(found, "One-hot encoded pin edge was not found.")

    def test_xray_edge_creation(self):
        """Tests that an X-Ray edge is correctly identified."""
        fen_xray = "8/8/8/4k3/4p3/4Q3/8/8 w - - 0 1"
        graph = fen_to_graph_data(fen_xray)
        board = chess.Board(fen_xray)

        sq_map = {sq: i for i, (sq, p) in enumerate(sorted(board.piece_map().items()))}
        attacker_idx = sq_map[chess.E3]
        target_idx = sq_map[chess.E5]

        found = False
        for i, (s, t) in enumerate(graph.edge_index.t().tolist()):
            if s == attacker_idx and t == target_idx:
                expected_one_hot = torch.nn.functional.one_hot(torch.tensor(EDGE_TYPE_TO_INT["XRAY"]), num_classes=NUM_EDGE_FEATURES).float()
                if torch.equal(graph.edge_attr[i], expected_one_hot):
                    found = True
                    break

        self.assertTrue(found, "One-hot encoded X-Ray edge was not found.")

    def test_global_features_in_nodes(self):
        """Tests that global features (castling, turn, etc.) are in every node."""
        fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        graph = fen_to_graph_data(fen)
        board = chess.Board(fen)

        self.assertEqual(graph.x.shape[0], board.occupied.bit_count())
        self.assertEqual(graph.x.shape[1], 11)

        sq_map = {sq: i for i, (sq, p) in enumerate(sorted(board.piece_map().items()))}
        king_node = graph.x[sq_map[chess.E1]]

        self.assertEqual(king_node[3].item(), 1.0) # Turn
        self.assertTrue(torch.all(king_node[4:8] == torch.tensor([1.0, 1.0, 1.0, 1.0]))) # Castling

if __name__ == '__main__':
    unittest.main()
