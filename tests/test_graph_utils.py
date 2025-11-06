import unittest
import chess
from scripts.graph_utils import fen_to_graph_data

# Assuming STARTING_FEN is available or defined here
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

class TestGraphUtils(unittest.TestCase):
    def test_fen_to_graph_startpos(self):
        """
        Tests if the graph conversion for the starting position is correct.
        - Should have 32 nodes (one for each piece).
        - Edge indices should be within the valid range.
        """
        graph = fen_to_graph_data(STARTING_FEN)
        self.assertEqual(graph.num_nodes, 32, "Should have 32 nodes for the starting position")

        # Check if edge indices are valid
        if graph.edge_index.numel() > 0:
            self.assertTrue(graph.edge_index.max() < 32, "Edge indices should be less than the number of nodes")
            self.assertTrue(graph.edge_index.min() >= 0, "Edge indices should be non-negative")

    def test_fen_to_graph_node_features(self):
        """
        Tests if the node features are correctly assigned.
        For a white pawn on e2 from the starting FEN.
        """
        board = chess.Board(STARTING_FEN)
        white_pawn_square = chess.E2

        # Find the node index corresponding to the piece on e2
        # This requires knowledge of how nodes are ordered in fen_to_graph_data
        # Assuming the order is consistent (e.g., by square index)
        piece_map = board.piece_map()
        sorted_squares = sorted(piece_map.keys())
        node_idx = sorted_squares.index(white_pawn_square)

        graph = fen_to_graph_data(STARTING_FEN)

        # Expected features for a white pawn on e2 (file 4, rank 1)
        # Assuming format [piece_type, file, rank]
        # piece_type for white pawn is 0
        expected_features = [0, 4, 1]

        # This test is fragile if the node ordering changes. A more robust
        # test would iterate through all nodes to find the one matching the square.
        # For now, this provides a basic sanity check.
        # self.assertEqual(graph.x[node_idx].tolist(), expected_features)

        # A better test: check if a node with these features exists
        found_node = any((node.tolist()[:3] == expected_features for node in graph.x))
        self.assertTrue(found_node, "Node for white pawn on e2 with correct features not found")


    def test_empty_board(self):
        """Tests the graph conversion for an empty board."""
        empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        graph = fen_to_graph_data(empty_fen)
        self.assertEqual(graph.num_nodes, 0, "Empty board should have 0 nodes")
        self.assertEqual(graph.edge_index.numel(), 0, "Empty board should have 0 edges")

if __name__ == '__main__':
    unittest.main()
