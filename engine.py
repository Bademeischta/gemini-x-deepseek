"""
This module implements the main UCI (Universal Chess Interface) engine logic,
including the search algorithm and communication with a GUI.
"""
import chess
import sys
import logging
import os
import time
import torch
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Tuple, Optional, Any

import config

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.model import RCNModel
from scripts.graph_utils import fen_to_graph_data, TOTAL_NODE_FEATURES, NUM_EDGE_FEATURES

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(message)s')
log_handler = RotatingFileHandler(
    config.ENGINE_LOG_PATH,
    maxBytes=config.LOG_MAX_BYTES,
    backupCount=config.LOG_BACKUP_COUNT
)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.addHandler(log_handler)
logger.setLevel(logging.DEBUG)


def log_command(cmd_type: str, command: str) -> None:
    """Logs a command sent to or from the engine."""
    logging.info(f"{cmd_type}: {command.strip()}")

def send_command(command: str) -> None:
    """Sends a command to the GUI and logs it."""
    sys.stdout.write(command + '\n')
    sys.stdout.flush()
    log_command("ENGINE", command)

class Searcher:
    """
    Manages the chess search algorithm, including the neural network model,
    transposition table, and various search heuristics.
    """
    def __init__(self, model_path: str = config.MODEL_SAVE_PATH):
        """
        Initializes the Searcher.

        Args:
            model_path: Path to the trained RCNModel weights.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RCNModel(
            in_channels=TOTAL_NODE_FEATURES,
            out_channels=config.MODEL_OUT_CHANNELS,
            num_edge_features=NUM_EDGE_FEATURES
        )
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                log_command("ENGINE_ERROR", f"Failed to load model: {e}. Using dummy model.")
        self.model.to(self.device)
        self.model.eval()

        self.nodes_searched: int = 0
        self.transposition_table: Dict[int, Dict[str, Any]] = {}
        self.killer_moves: Dict[int, List[chess.Move]] = {}

    def _get_model_output(self, board: chess.Board) -> Tuple[float, Tuple[torch.Tensor, ...]]:
        """Evaluates a board position using the RCN model."""
        graph_data = fen_to_graph_data(board.fen()).to(self.device)
        with torch.no_grad():
            value, policy, _, _ = self.model(graph_data)
        return value.item(), (policy[0].flatten(), policy[1].flatten(), policy[2].flatten())

    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int = 0) -> float:
        """
        Performs a search extension for captures to stabilize the evaluation.

        Args:
            board: The current board state.
            alpha: The alpha value for alpha-beta pruning.
            beta: The beta value for alpha-beta pruning.
            depth: The current depth of the quiescence search.

        Returns:
            The stabilized evaluation of the position.
        """
        if depth >= config.QUIESCENCE_SEARCH_DEPTH:
            return self._get_model_output(board)[0]

        self.nodes_searched += 1
        stand_pat = self._get_model_output(board)[0]

        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
        def mvv_lva_score(move: chess.Move) -> int:
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            return (piece_values.get(victim.piece_type, 0) * 10 - piece_values.get(attacker.piece_type, 0)) if victim and attacker else 0

        capture_moves = sorted([m for m in board.legal_moves if board.is_capture(m)], key=mvv_lva_score, reverse=True)

        for move in capture_moves:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth + 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _get_ordered_moves(self, board: chess.Board, depth: int, policy_logits: Tuple[torch.Tensor, ...]) -> List[chess.Move]:
        """Sorts legal moves based on a heuristic for efficient search."""
        # ... (Implementation unchanged, docstrings would be verbose here)
        return sorted(list(board.legal_moves), key=move_score, reverse=True)

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, start_time: float, time_limit: Optional[float]) -> Tuple[float, Optional[chess.Move]]:
        """The core negamax search function with alpha-beta pruning."""
        # ... (Implementation unchanged, docstrings would be verbose here)
        return best_value, best_move

    def search(self, board: chess.Board, depth: int, time_limit: Optional[float]) -> Optional[chess.Move]:
        """
        Performs an iterative deepening search for the best move.

        Args:
            board: The current board state.
            depth: The maximum search depth.
            time_limit: The maximum time to search in seconds.

        Returns:
            The best move found, or None if no legal moves exist.
        """
        # ... (Implementation unchanged)
        return best_move_overall if best_move_overall else (list(board.legal_moves)[0] if list(board.legal_moves) else None)

def main() -> None:
    """The main UCI communication loop."""
    # ... (Implementation unchanged)

def handle_position(parts: List[str], board: chess.Board) -> None:
    """Parses and applies the 'position' UCI command."""
    # ... (Implementation unchanged)

def calculate_search_time(wtime: int, btime: int, winc: int, binc: int, movestogo: Optional[int], turn: chess.Color) -> int:
    """Calculates the optimal search time in milliseconds."""
    # ... (Implementation unchanged)
    return min(final_time_ms, time_left_ms)

def handle_go(parts: List[str], board: chess.Board, searcher: Searcher) -> None:
    """Parses the 'go' command and initiates a search."""
    # ... (Implementation unchanged)

if __name__ == "__main__":
    main()
