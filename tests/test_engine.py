import unittest
from unittest.mock import patch, MagicMock
import io
import sys
import os
import chess

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to import the main function from the engine script
from engine import main as engine_main, Searcher, calculate_search_time

class TestEngineUCI(unittest.TestCase):

    def run_engine_command(self, commands):
        """Helper to run engine commands and capture the output."""
        # Redirect stdout to capture engine's responses
        new_stdout = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = new_stdout

        # Simulate stdin by patching it
        with patch('sys.stdin', io.StringIO("\n".join(commands) + "\n")):
            try:
                engine_main()
            except (SystemExit, StopIteration):
                # The main loop might exit, which is fine for testing
                pass

        # Restore stdout
        sys.stdout = old_stdout
        return new_stdout.getvalue().strip().split('\n')

    @patch('engine.Searcher')
    def test_uci_handshake(self, MockSearcher):
        """Tests the initial 'uci' -> 'id' -> 'uciok' handshake."""
        # Arrange
        mock_searcher_instance = MockSearcher.return_value
        commands = ["uci", "quit"]

        # Act
        output = self.run_engine_command(commands)

        # Assert
        self.assertIn("id name RCN Engine", output)
        self.assertIn("id author Jules", output)
        self.assertIn("uciok", output)
        MockSearcher.assert_called_once() # Check that the searcher was initialized

    @patch('engine.Searcher')
    def test_isready_response(self, MockSearcher):
        """Tests that 'isready' command produces a 'readyok' response."""
        # Arrange
        mock_searcher_instance = MockSearcher.return_value
        commands = ["uci", "isready", "quit"]

        # Act
        output = self.run_engine_command(commands)

        # Assert
        self.assertIn("readyok", output)

    @patch('engine.Searcher')
    def test_go_command_triggers_search(self, MockSearcher):
        """Tests that a 'go' command calls the search method."""
        # Arrange
        mock_searcher_instance = MockSearcher.return_value
        # We need to mock the search method to avoid actually running a search
        mock_searcher_instance.search.return_value = chess.Move.from_uci("e2e4")
        commands = [
            "uci",
            "position startpos",
            "go depth 1",
            "quit"
        ]

        # Act
        output = self.run_engine_command(commands)

        # Assert
        mock_searcher_instance.search.assert_called_once()
        self.assertTrue(any("bestmove e2e4" in line for line in output))

    def test_calculate_search_time(self):
        """Tests the time calculation logic for various scenarios."""
        # Scenario 1: Sudden death, plenty of time
        # 60s left, 0s increment -> should use a fraction (e.g., 1/25)
        self.assertGreater(calculate_search_time(60000, 60000, 0, 0, None, chess.WHITE), 2000)
        self.assertLess(calculate_search_time(60000, 60000, 0, 0, None, chess.WHITE), 3000)

        # Scenario 2: Moves to go
        # 60s left, 20 moves to go -> should be close to 3s
        self.assertGreater(calculate_search_time(60000, 60000, 0, 0, 20, chess.WHITE), 2900)
        self.assertLess(calculate_search_time(60000, 60000, 0, 0, 20, chess.WHITE), 3100)

        # Scenario 3: Low time
        # 5s left, 0s increment -> should use a smaller fraction (e.g., 1/15)
        self.assertGreater(calculate_search_time(5000, 5000, 0, 0, None, chess.WHITE), 300)
        self.assertLess(calculate_search_time(5000, 5000, 0, 0, None, chess.WHITE), 400)


if __name__ == '__main__':
    unittest.main()
