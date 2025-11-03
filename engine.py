import chess
import sys
import random
import logging
import os
import time
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
    def __init__(self, model_path="models/rcn_model.pth", use_negamax=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_command("ENGINE_INFO", f"Using device: {self.device}")
        self.use_negamax = use_negamax

        self.model = RCNModel()
        if not os.path.exists(model_path):
            log_command("ENGINE_WARNING", f"Model not found at {model_path}")
            log_command("ENGINE_WARNING", "Creating dummy model for testing...")
            from scripts.create_dummy_model import create_dummy_model
            create_dummy_model()

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            log_command("ENGINE_INFO", f"Model loaded successfully from {model_path}")
        except Exception as e:
            log_command("ENGINE_ERROR", f"An error occurred while loading the model: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()

        # --- Search Enhancements ---
        self.pv_table = {} # To store Principal Variation
        self.nodes_searched = 0

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
        if depth == 0:
            return self._evaluate(board)

        stand_pat = self._evaluate(board)

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

    def _get_ordered_moves(self, board, pv_move=None):
        """
        Gets all legal moves and sorts them using enhanced move ordering heuristics.
        1. PV Move from previous iteration.
        2. MVV-LVA for captures.
        3. Policy network predictions as a fallback/bonus.
        """
        graph_data = fen_to_graph_data(board.fen()).to(self.device)
        with torch.no_grad():
            output = self.model(graph_data)
            policy_logits = output['policy']

        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

        def move_score(move):
            score = 0
            # 1. PV Move has the highest priority
            if pv_move and move == pv_move:
                return 10000

            # 2. Captures with MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            if board.is_capture(move):
                victim_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                # In case of en-passant, the victim piece is not at the 'to_square'
                if victim_piece is None and board.is_en_passant(move):
                    victim_piece = chess.Piece(chess.PAWN, not board.turn)

                if victim_piece and attacker_piece:
                    victim_value = piece_values.get(victim_piece.piece_type, 0)
                    attacker_value = piece_values.get(attacker_piece.piece_type, 0)
                    score += 1000 + (victim_value * 10 - attacker_value)

            # 3. Add policy head score as a bonus
            try:
                score += policy_logits[0, uci_to_index(move.uci())].item()
            except IndexError:
                pass

            return score

        legal_moves = list(board.legal_moves)
        return sorted(legal_moves, key=move_score, reverse=True)

    def _ir_alpha_beta(self, board, depth, alpha, beta, pv_line, start_time, time_limit):
        # Time check inside the recursive search
        if time_limit and (time.time() - start_time) > time_limit:
            raise TimeoutError

        self.nodes_searched += 1

        if depth == 0:
            # Quiescence search doesn't return a move, only an evaluation.
            return self._quiescence_search(board, alpha, beta, 4), None

        if board.is_game_over(claim_draw=True):
            if board.is_checkmate():
                # The move that led to this state was the winning one for the opponent.
                # From this position, there is no best move.
                return -float('inf'), None
            return 0, None # Draw

        # Get the PV move for the current position from the main pv_line
        current_pv_move = pv_line[0] if pv_line else None
        ordered_moves = self._get_ordered_moves(board, pv_move=current_pv_move)

        best_move_found = None

        if board.turn == chess.WHITE:
            max_eval = -float('inf')
            for move in ordered_moves:
                temp_board = board.copy()
                temp_board.push(move)

                child_pv = pv_line[1:] if current_pv_move == move and pv_line else []
                eval, _ = self._ir_alpha_beta(temp_board, depth - 1, alpha, beta, child_pv, start_time, time_limit)

                if eval > max_eval:
                    max_eval = eval
                    best_move_found = move

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break # Beta cutoff
            return max_eval, best_move_found
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                temp_board = board.copy()
                temp_board.push(move)

                child_pv = pv_line[1:] if current_pv_move == move and pv_line else []
                eval, _ = self._ir_alpha_beta(temp_board, depth - 1, alpha, beta, child_pv, start_time, time_limit)

                if eval < min_eval:
                    min_eval = eval
                    best_move_found = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break # Alpha cutoff
            return min_eval, best_move_found

    def _search_negamax(self, board, depth, time_limit):
        """
        Placeholder for a future true Negamax search implementation.
        """
        log_command("ENGINE_ERROR", "Negamax search is not yet implemented.")
        # Fallback to the first legal move
        if list(board.legal_moves):
            return list(board.legal_moves)[0]
        return None

    def _search_minmax(self, board, depth, time_limit):
        """
        The original Minimax-style search implementation.
        """
        start_time = time.time()
        self.nodes_searched = 0
        best_move = None
        principal_variation = []

        for current_depth in range(1, depth + 1):
            try:
                alpha = -float('inf')
                beta = float('inf')
                ordered_moves = self._get_ordered_moves(board, pv_move=principal_variation[0] if principal_variation else None)
                current_best_move = None

                # This is the root move evaluation logic for Minimax.
                current_pv = []
                for move in ordered_moves:
                    temp_board = board.copy()
                    temp_board.push(move)

                    # The PV for the child node is what follows the current move in the PV from the last iteration.
                    child_pv = principal_variation[1:] if principal_variation and move == principal_variation[0] else []

                    # Call the recursive search, which now returns (eval, best_move_from_child)
                    eval, best_move_from_child = self._ir_alpha_beta(temp_board, current_depth - 1, alpha, beta, child_pv, start_time, time_limit)

                    # Update best move based on the side to move
                    if board.turn == chess.WHITE:
                        if eval > alpha:
                            alpha = eval
                            current_best_move = move
                            # Reconstruct the PV: current move + PV from the child node
                            current_pv = [current_best_move] + ([best_move_from_child] if best_move_from_child else [])
                    else: # Black's turn
                        if eval < beta:
                            beta = eval
                            current_best_move = move
                            # Reconstruct the PV
                            current_pv = [current_best_move] + ([best_move_from_child] if best_move_from_child else [])

                if current_best_move:
                    best_move = current_best_move
                    principal_variation = current_pv # Update the main PV for the next iteration

                elapsed_time = time.time() - start_time
                # The score perspective depends on whose move it is at the root.
                # For UCI, 'cp' score is from the current player's perspective.
                score = alpha if board.turn == chess.WHITE else beta
                score_cp = int(score * 100)
                pv_str = best_move.uci() if best_move else "none"
                send_command(f"info depth {current_depth} score cp {score_cp} nodes {self.nodes_searched} time {int(elapsed_time * 1000)} pv {pv_str}")

            except TimeoutError:
                log_command("ENGINE_INFO", f"Time limit reached at depth {current_depth}. Stopping search.")
                break

        log_command("SEARCH_RESULT", f"Best move: {best_move.uci() if best_move else 'None'} after {self.nodes_searched} nodes")
        return best_move

    def search(self, board, depth, time_limit):
        """
        Entry point for the search. Dispatches to the selected search algorithm.
        """
        if self.use_negamax:
            best_move = self._search_negamax(board, depth, time_limit)
        else:
            best_move = self._search_minmax(board, depth, time_limit)

        # Fallback if no legal moves are available at the start
        if not best_move and list(board.legal_moves):
            return list(board.legal_moves)[0]

        return best_move

def main():
    """Main UCI communication loop."""
    board = chess.Board()
    searcher = None
    is_initialized = False

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
            # Instantiate the searcher in a controlled way.
            # The 'isready' command will confirm when it's done.
            if not searcher:
                try:
                    # Heavy model loading happens here
                    searcher = Searcher()
                    is_initialized = True
                except Exception as e:
                    log_command("ENGINE_ERROR", f"Failed to initialize Searcher: {e}")
                    # In case of error, we cannot proceed.
                    break
            send_command("uciok")

        elif command == "isready":
            # Only send 'readyok' if the searcher (and model) are fully loaded.
            if is_initialized and searcher:
                send_command("readyok")
            else:
                log_command("ENGINE_WARNING", "Received 'isready' command but engine is not initialized yet.")

        elif command == "ucinewgame":
            board.reset()

        elif command == "position":
            handle_position(parts[1:], board)

        elif command == "go":
            # The check for 'is_initialized' is implicitly handled by 'searcher' being non-None.
            if searcher:
                handle_go(parts, board, searcher)
            else:
                log_command("ENGINE_ERROR", "Received 'go' command before engine was ready.")

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

def calculate_search_time(wtime, btime, winc, binc, movestogo, side_to_move) -> int:
    """
    Calculates the optimal search time based on UCI parameters, working
    internally with integers (milliseconds) to avoid floating-point errors.
    Returns the allocated time in milliseconds.
    """
    time_left_ms = wtime if side_to_move == chess.WHITE else btime
    increment_ms = winc if side_to_move == chess.WHITE else binc

    allocated_time_ms = 0

    if movestogo and movestogo > 0:
        # Classical time control: divide remaining time by moves to go.
        allocated_time_ms = (time_left_ms // movestogo) + (increment_ms * 8 // 10)
    else:
        # Sudden death: use a fraction of the remaining time.
        # Use a larger fraction for shorter time controls.
        divider = 25 if time_left_ms > 20000 else 15
        allocated_time_ms = (time_left_ms // divider) + (increment_ms * 9 // 10)

    # Safety buffer: never use more than 80% of the remaining time.
    max_time_ms = time_left_ms * 8 // 10

    # Ensure a minimum thinking time, e.g., 50ms, but not more than available.
    min_time_ms = 50

    # Clamp the allocated time between min and max, ensuring we don't exceed max_time.
    # Also, ensure we don't use more time than we have.
    final_time_ms = max(min_time_ms, min(allocated_time_ms, max_time_ms))

    # Final check to ensure we don't allocate more time than we have left
    return min(final_time_ms, time_left_ms)


def handle_go(parts, board, searcher):
    """Handles the 'go' UCI command by calling the searcher."""
    # Default values
    depth = 100  # Set a high depth limit for timed searches
    time_limit_ms = None # Time limit is now in milliseconds

    # --- Parse UCI 'go' parameters ---
    params = {}
    i = 1
    while i < len(parts):
        if parts[i] in ["wtime", "btime", "winc", "binc", "movestogo", "depth"]:
            try:
                params[parts[i]] = int(parts[i+1])
                i += 2
            except (ValueError, IndexError):
                log_command("ENGINE_ERROR", f"Could not parse value for '{parts[i]}'")
                i += 1
        else:
            i += 1

    # --- Time Management ---
    if "wtime" in params and "btime" in params:
        time_limit_ms = calculate_search_time(
            params.get("wtime", 0),
            params.get("btime", 0),
            params.get("winc", 0),
            params.get("binc", 0),
            params.get("movestogo"), # Can be None
            board.turn
        )
        log_command("ENGINE_INFO", f"Time control active. Search time: {time_limit_ms / 1000:.3f}s")
    elif "depth" in params:
        depth = params["depth"]
        time_limit_ms = None # No time limit for fixed depth search
        log_command("ENGINE_INFO", f"Fixed depth search: {depth}")
    else:
        # Default to depth 4 if no params are given
        depth = 4
        time_limit_ms = None
        log_command("ENGINE_INFO", f"No params. Using default depth: {depth}")

    # The search function expects the time limit in seconds.
    time_limit_sec = time_limit_ms / 1000.0 if time_limit_ms is not None else None
    best_move = searcher.search(board, depth=depth, time_limit=time_limit_sec)

    if best_move:
        send_command(f"bestmove {best_move.uci()}")
    else:
        # UCI-Standard für "kein legaler Zug" (Matt/Patt)
        send_command("bestmove 0000")
        log_command("ENGINE_INFO", "No legal moves (mate/stalemate)")

if __name__ == "__main__":
    main()
