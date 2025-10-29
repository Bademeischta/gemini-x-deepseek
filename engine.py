import chess
import sys
import random
import logging
import os
import torch

# Add the project root to the Python path to allow importing from 'scripts'
# This is necessary because the engine might be run from a different directory.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.model import RCNModel
from scripts.graph_utils import fen_to_graph_data
from scripts.move_utils import uci_to_index

# Configure logging
logging.basicConfig(filename='engine.log', level=logging.DEBUG,
                    format='%(asctime)s - %(message)s')

def log_command(cmd_type, command):
    """Logs commands received from or sent to the GUI."""
    logging.info(f"{cmd_type}: {command.strip()}")

def send_command(command):
    """Sends a command to the GUI and logs it."""
    sys.stdout.write(command + '\n')
    sys.stdout.flush()
    log_command("ENGINE", command)

class Searcher:
    """
    Handles the AI logic for finding the best move.
    """
    def __init__(self, model_path="models/rcn_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_command("ENGINE_INFO", f"Using device: {self.device}")

        self.model = RCNModel()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            log_command("ENGINE_INFO", f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            log_command("ENGINE_ERROR", f"Model file not found at {model_path}. The engine will not work.")
            # In a real scenario, we might want to exit or handle this more gracefully.
            raise
        except Exception as e:
            log_command("ENGINE_ERROR", f"An error occurred while loading the model: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()

    def _evaluate(self, board):
        """
        Evaluates a single board position using the RCN model.
        """
        graph_data = fen_to_graph_data(board.fen()).to(self.device)
        with torch.no_grad():
            output = self.model(graph_data)
            return output['value'].item()

    def _quiescence_search(self, board, alpha, beta, depth):
        """
        Performs a search extension for capture moves to stabilize the evaluation.
        """
        stand_pat = self._evaluate(board)

        if depth == 0:
            return stand_pat

        if board.turn == chess.WHITE:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)

        # Generate only capture moves
        capture_moves = [move for move in board.legal_moves if board.is_capture(move)]

        for move in capture_moves:
            temp_board = board.copy()
            temp_board.push(move)
            score = self._quiescence_search(temp_board, alpha, beta, depth - 1)

            if board.turn == chess.WHITE:
                alpha = max(alpha, score)
                if alpha >= beta:
                    return beta  # Pruning
            else:
                beta = min(beta, score)
                if beta <= alpha:
                    return alpha  # Pruning

        return alpha if board.turn == chess.WHITE else beta

    def _get_ordered_moves(self, board):
        """
        Gets all legal moves and sorts them based on the policy head output.
        """
        graph_data = fen_to_graph_data(board.fen()).to(self.device)
        with torch.no_grad():
            output = self.model(graph_data)
            policy_logits = output['policy']

        legal_moves = list(board.legal_moves)
        move_scores = [policy_logits[0, uci_to_index(m.uci())].item() for m in legal_moves]

        # Sort moves by their scores in descending order
        sorted_moves = [move for _, move in sorted(zip(move_scores, legal_moves), reverse=True)]
        return sorted_moves

    def _ir_alpha_beta(self, board, depth, alpha, beta):
        if depth == 0:
            # At leaf nodes, enter quiescence search
            graph_data = fen_to_graph_data(board.fen()).to(self.device)
            with torch.no_grad():
                output = self.model(graph_data)

            tactic_flag = output['tactic'].item()
            max_q_depth = 4
            if tactic_flag > 0.5:
                max_q_depth += 2 # Extend search in sharp positions

            return self._quiescence_search(board, alpha, beta, max_q_depth)

        ordered_moves = self._get_ordered_moves(board)

        if board.turn == chess.WHITE:
            max_eval = -float('inf')
            for move in ordered_moves:
                temp_board = board.copy()
                temp_board.push(move)
                eval = self._ir_alpha_beta(temp_board, depth - 1, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                temp_board = board.copy()
                temp_board.push(move)
                eval = self._ir_alpha_beta(temp_board, depth - 1, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def search(self, board, depth):
        """
        Entry point for the IR-AB search.
        """
        best_move = None
        alpha = -float('inf')
        beta = float('inf')

        ordered_moves = self._get_ordered_moves(board)

        if board.turn == chess.WHITE:
            max_eval = -float('inf')
            for move in ordered_moves:
                temp_board = board.copy()
                temp_board.push(move)
                eval = self._ir_alpha_beta(temp_board, depth - 1, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                temp_board = board.copy()
                temp_board.push(move)
                eval = self._ir_alpha_beta(temp_board, depth - 1, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)

        log_command("SEARCH_RESULT", f"Best move: {best_move.uci() if best_move else 'None'}")
        return best_move

def main():
    """Main UCI communication loop."""
    board = chess.Board()
    searcher = None  # Initialize searcher to None

    while True:
        line = sys.stdin.readline()
        if not line:
            continue

        log_command("GUI", line)
        parts = line.strip().split()
        command = parts[0]

        if command == "uci":
            send_command("id name RCN Engine")
            send_command("id author Jules")
            # Instantiate the searcher here, after UCI handshake
            try:
                searcher = Searcher()
            except Exception as e:
                log_command("ENGINE_ERROR", f"Failed to initialize Searcher: {e}")
                # We can't continue if the model fails to load.
                break
            send_command("uciok")
        elif command == "isready":
            # Model loading is now part of Searcher's __init__
            send_command("readyok")
        elif command == "ucinewgame":
            board.reset()
        elif command == "position":
            handle_position(parts[1:], board)
        elif command == "go":
            if searcher:
                handle_go(parts, board, searcher)
            else:
                log_command("ENGINE_ERROR", "Received 'go' command before 'uci' or after failed init.")
        elif command == "quit":
            break

def handle_position(parts, board):
    """Handles the 'position' UCI command."""
    try:
        if parts[0] == "startpos":
            board.reset()
            moves_start_index = 1
        elif parts[0] == "fen":
            fen = " ".join(parts[1:7])
            board.set_fen(fen)
            moves_start_index = 7
        else:
            # Should not happen with a compliant GUI
            return

        if len(parts) > moves_start_index and parts[moves_start_index] == "moves":
            for move_uci in parts[moves_start_index + 1:]:
                board.push_uci(move_uci)
    except Exception as e:
        logging.error(f"Error handling 'position' command: {parts} - {e}")

def handle_go(parts, board, searcher):
    """Handles the 'go' UCI command by calling the searcher."""
    depth = 4  # Default depth
    if "depth" in parts:
        try:
            depth_index = parts.index("depth") + 1
            if depth_index < len(parts):
                depth = int(parts[depth_index])
        except (ValueError, IndexError):
            log_command("ENGINE_ERROR", f"Could not parse depth from 'go' command: {parts}")

    best_move = searcher.search(board, depth=depth)

    if best_move:
        send_command(f"bestmove {best_move.uci()}")
    else:
        # This can happen in mate positions where there are no legal moves.
        # Although the UCI protocol doesn't specify what to do, sending a null move is a common practice.
        log_command("ENGINE_INFO", "No legal moves found.")
        # Some GUIs might expect "bestmove 0000" or nothing at all. Let's send nothing for now.

if __name__ == "__main__":
    main()
