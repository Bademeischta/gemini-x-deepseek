import unittest
from scripts.move_utils import uci_to_index, index_to_uci

class TestMoveUtils(unittest.TestCase):
    def test_uci_to_index_basic_moves(self):
        """Tests if common UCI moves are converted to unique indices."""
        move1 = "e2e4"
        move2 = "e7e5"
        move3 = "g1f3"

        index1 = uci_to_index(move1)
        index2 = uci_to_index(move2)
        index3 = uci_to_index(move3)

        self.assertIsInstance(index1, int)
        self.assertIsInstance(index2, int)
        self.assertIsInstance(index3, int)

        self.assertNotEqual(index1, index2, "Different moves should have different indices")
        self.assertNotEqual(index1, index3, "Different moves should have different indices")
        self.assertNotEqual(index2, index3, "Different moves should have different indices")

    def test_uci_to_index_promotion(self):
        """Tests promotion moves."""
        promo_q = "e7e8q"
        promo_r = "e7e8r"

        index_q = uci_to_index(promo_q)
        index_r = uci_to_index(promo_r)

        self.assertNotEqual(index_q, index_r, "Different promotion moves should have different indices")

    def test_index_to_uci_roundtrip(self):
        """Tests if converting a UCI to an index and back yields the original UCI."""
        moves_to_test = ["a2a4", "b8c6", "f7f8q", "h1g1"]

        for move in moves_to_test:
            with self.subTest(move=move):
                index = uci_to_index(move)
                reconstructed_move = index_to_uci(index)
                self.assertEqual(move, reconstructed_move, "Roundtrip conversion should yield the original move")

    def test_invalid_uci_raises_error(self):
        """Tests if an invalid UCI string raises a KeyError or ValueError."""
        invalid_uci = "e2e9" # Invalid square
        with self.assertRaises((KeyError, ValueError)):
            uci_to_index(invalid_uci)

    def test_index_out_of_bounds_raises_error(self):
        """Tests if an out-of-bounds index raises an IndexError."""
        # Assuming the move space is smaller than 5000
        invalid_index = 5000
        with self.assertRaises(IndexError):
            index_to_uci(invalid_index)

if __name__ == '__main__':
    unittest.main()
