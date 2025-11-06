import unittest
from unittest.mock import patch
import io
import sys
import os
import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine import uci_loop, Searcher, calculate_search_time

class TestEngineUCI(unittest.TestCase):

    def run_engine_commands(self, commands: list[str]) -> list[str]:
        """Helper to run engine commands and capture the output."""
        stdin = io.StringIO("\n".join(commands) + "\n")
        stdout = io.StringIO()

        with patch('engine.send_command', lambda cmd, out=stdout: out.write(cmd + '\n')):
             uci_loop(stdin, stdout)

        return stdout.getvalue().strip().split('\n')

    @patch('engine.Searcher')
    def test_uci_handshake(self, MockSearcher):
        """Tests the initial 'uci' -> 'id' -> 'uciok' handshake."""
        MockSearcher.return_value
        commands = ["uci", "quit"]

        output = self.run_engine_commands(commands)

        self.assertIn("id name RCN Engine", output)
        self.assertIn("id author Jules", output)
        self.assertIn("uciok", output)
        MockSearcher.assert_called_once()

    @patch('engine.Searcher')
    def test_isready_response(self, MockSearcher):
        """Tests that 'isready' command produces a 'readyok' response."""
        MockSearcher.return_value
        commands = ["uci", "isready", "quit"]

        output = self.run_engine_commands(commands)

        self.assertIn("readyok", output)

    @patch('engine.Searcher')
    def test_go_command_triggers_search(self, MockSearcher):
        """Tests that a 'go' command calls the search method."""
        mock_searcher_instance = MockSearcher.return_value
        mock_searcher_instance.search.return_value = chess.Move.from_uci("e2e4")
        commands = ["uci", "position startpos", "go depth 1", "quit"]

        # We need a custom send_command mock for this test to capture 'bestmove'
        stdin = io.StringIO("\n".join(commands) + "\n")
        stdout = io.StringIO()

        uci_loop(stdin, stdout)

        output = stdout.getvalue().strip().split('\n')

        mock_searcher_instance.search.assert_called_once()
        self.assertTrue(any("bestmove e2e4" in line for line in output))

    def test_calculate_search_time(self):
        """Tests the time calculation logic for various scenarios."""
        # Scenario 1: Sudden death, plenty of time
        self.assertGreater(calculate_search_time(60000, 60000, 0, 0, None, chess.WHITE), 2000)
        # ... (rest of the test)

if __name__ == '__main__':
    unittest.main()
