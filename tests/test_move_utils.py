import unittest
import chess
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.move_utils import uci_to_policy_targets, policy_targets_to_uci

class TestMoveUtils(unittest.TestCase):
    def test_uci_to_policy_targets(self):
        """Tests conversion from UCI string to policy target indices."""
        # Standard move
        self.assertEqual(uci_to_policy_targets("e2e4"), {'from': chess.E2, 'to': chess.E4, 'promo': -1})
        # Capture
        self.assertEqual(uci_to_policy_targets("g1f3"), {'from': chess.G1, 'to': chess.F3, 'promo': -1})
        # Promotion
        self.assertEqual(uci_to_policy_targets("a7a8q"), {'from': chess.A7, 'to': chess.A8, 'promo': 3}) # Queen
        # Invalid move
        self.assertEqual(uci_to_policy_targets("e2e9"), {'from': -1, 'to': -1, 'promo': -1})
        # Empty string
        self.assertEqual(uci_to_policy_targets(""), {'from': -1, 'to': -1, 'promo': -1})

    def test_policy_targets_to_uci_roundtrip(self):
        """Tests the roundtrip conversion for various moves."""
        board = chess.Board()

        # Standard move
        uci = "e2e4"
        targets = uci_to_policy_targets(uci)
        self.assertEqual(policy_targets_to_uci(targets['from'], targets['to'], board), uci)

        # Promotion move
        board_promo = chess.Board("rnbqkbnr/pPpppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        uci_promo = "b7a8q"
        targets_promo = uci_to_policy_targets(uci_promo)
        self.assertEqual(policy_targets_to_uci(targets_promo['from'], targets_promo['to'], board_promo), uci_promo)

if __name__ == '__main__':
    unittest.main()
