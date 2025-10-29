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

    def search(self, board, depth):
        """
        Finds the best move using a 1-ply search based on the model's value output.
        """
        best_move = None
        # Use -infinity for white's turn and +infinity for black's turn
        best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')

        for move in board.legal_moves:
            # Create a copy of the board and make the move
            temp_board = board.copy()
            temp_board.push(move)

            # Convert the resulting position to a graph
            graph_data = fen_to_graph_data(temp_board.fen())
            graph_data = graph_data.to(self.device)

            # Get the model's evaluation
            with torch.no_grad():
                output = self.model(graph_data)
                eval = output['value'].item()

            # Log the evaluation for debugging
            log_command("SEARCH_INFO", f"Move: {move.uci()}, Eval: {eval:.4f}")

            # Compare evaluations
            if board.turn == chess.WHITE:
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
            else: # Black's turn
                if eval < best_eval:
                    best_eval = eval
                    best_move = move

        log_command("SEARCH_RESULT", f"Best move: {best_move.uci() if best_move else 'None'}, Final Eval: {best_eval:.4f}")
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
                handle_go(board, searcher)
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

def handle_go(board, searcher):
    """Handles the 'go' UCI command by calling the searcher."""
    # For now, we use a fixed depth of 1, as requested.
    best_move = searcher.search(board, depth=1)

    if best_move:
        send_command(f"bestmove {best_move.uci()}")
    else:
        # This can happen in mate positions where there are no legal moves.
        # Although the UCI protocol doesn't specify what to do, sending a null move is a common practice.
        log_command("ENGINE_INFO", "No legal moves found.")
        # Some GUIs might expect "bestmove 0000" or nothing at all. Let's send nothing for now.

if __name__ == "__main__":
    main()
