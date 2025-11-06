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
from typing import List, Dict, Tuple, Optional, Any, IO

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

def send_command(command: str, stdout: IO[str] = sys.stdout) -> None:
    """Sends a command to the GUI and logs it."""
    stdout.write(command + '\n')
    stdout.flush()
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
        self.move_cache: Dict[int, List[chess.Move]] = {}


    def _get_model_output(self, board: chess.Board) -> Tuple[float, Tuple[torch.Tensor, ...]]:
        """Evaluates a board position using the RCN model."""
        graph_data = fen_to_graph_data(board.fen()).to(self.device)
        with torch.no_grad():
            value, policy, _, _ = self.model(graph_data)
        return value.item(), (policy[0].flatten(), policy[1].flatten(), policy[2].flatten())

    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int = 0) -> float:
        """
        Performs a search extension for captures to stabilize the evaluation.
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
        z_hash = chess.zobrist_hash(board)
        if z_hash in self.move_cache:
            return self.move_cache[z_hash]

        pv_move = self.transposition_table.get(z_hash, {}).get('move')

        from_logits, to_logits, promo_logits = policy_logits
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

        def move_score(move: chess.Move) -> float:
            if move == pv_move: return float('inf')
            score = 0.0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker: score += 1e6 + (piece_values.get(victim.piece_type, 0) * 10 - piece_values.get(attacker.piece_type, 0))
            elif depth in self.killer_moves and move in self.killer_moves[depth]:
                score += 1e5
            else:
                from_sq_prob = torch.softmax(from_logits, dim=0)[move.from_square].item()
                to_sq_prob = torch.softmax(to_logits, dim=0)[move.to_square].item()
                promo_prob = 1.0
                if move.promotion:
                    promo_idx = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN].index(move.promotion)
                    promo_prob = torch.softmax(promo_logits, dim=0)[promo_idx].item()
                score += from_sq_prob * to_sq_prob * promo_prob
            return score

        ordered_moves = sorted(list(board.legal_moves), key=move_score, reverse=True)
        self.move_cache[z_hash] = ordered_moves
        return ordered_moves

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, start_time: float, time_limit: Optional[float]) -> Tuple[float, Optional[chess.Move]]:
        """The core negamax search function."""
        alpha_orig = alpha
        z_hash = chess.zobrist_hash(board)

        if z_hash in self.transposition_table and self.transposition_table[z_hash]['depth'] >= depth:
            entry = self.transposition_table[z_hash]
            if entry['flag'] == 'EXACT': return entry['value'], entry['move']
            elif entry['flag'] == 'LOWERBOUND': alpha = max(alpha, entry['value'])
            elif entry['flag'] == 'UPPERBOUND': beta = min(beta, entry['value'])
            if alpha >= beta: return entry['value'], entry['move']

        if depth == 0 or board.is_game_over(claim_draw=True):
            if board.is_checkmate(): return -30000, None
            if board.is_game_over(claim_draw=True): return 0, None
            return self._quiescence_search(board, alpha, beta), None

        if time_limit and (time.time() - start_time) > time_limit:
            raise TimeoutError

        self.nodes_searched += 1
        _, policy_logits = self._get_model_output(board)
        ordered_moves = self._get_ordered_moves(board, depth, policy_logits)

        best_value = -float('inf')
        best_move = ordered_moves[0] if ordered_moves else None

        for move in ordered_moves:
            board.push(move)
            value, _ = self._negamax(board, depth - 1, -beta, -alpha, start_time, time_limit)
            value = -value
            board.pop()
            if value > best_value:
                best_value, best_move = value, move
            alpha = max(alpha, best_value)
            if alpha >= beta:
                if not board.is_capture(move):
                    if depth not in self.killer_moves: self.killer_moves[depth] = []
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, move)
                        self.killer_moves[depth] = self.killer_moves[depth][:2]
                break

        flag = 'UPPERBOUND' if best_value <= alpha_orig else ('LOWERBOUND' if best_value >= beta else 'EXACT')
        self.transposition_table[z_hash] = {'depth': depth, 'value': best_value, 'move': best_move, 'flag': flag}
        return best_value, best_move

    def search(self, board: chess.Board, depth: int, time_limit: Optional[float]) -> Optional[chess.Move]:
        """Performs iterative deepening search."""
        self.transposition_table.clear()
        self.killer_moves.clear()
        self.move_cache.clear()
        self.nodes_searched = 0
        start_time = time.time()
        best_move_overall = None

        for d in range(1, depth + 1):
            try:
                score, best_move = self._negamax(board, d, -float('inf'), float('inf'), start_time, time_limit)
                if best_move: best_move_overall = best_move

                pv = [best_move] if best_move else []
                temp_board = board.copy()
                if best_move:
                    temp_board.push(best_move)
                    while chess.zobrist_hash(temp_board) in self.transposition_table:
                        entry = self.transposition_table.get(chess.zobrist_hash(temp_board))
                        if not entry or not entry.get('move'): break
                        move = entry['move']
                        pv.append(move)
                        temp_board.push(move)
                        if len(pv) >= d: break

                elapsed = int((time.time() - start_time) * 1000)
                cp_score = int(score * 100) if board.turn == chess.WHITE else int(-score * 100)
                send_command(f"info depth {d} score cp {cp_score} nodes {self.nodes_searched} time {elapsed} pv {' '.join([m.uci() for m in pv if m])}")
            except TimeoutError:
                break
            except Exception as e:
                log_command("ENGINE_ERROR", f"Error in search at depth {d}: {e}")
                break

        return best_move_overall if best_move_overall else (list(board.legal_moves)[0] if board.legal_moves else None)

def uci_loop(stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
    """The main UCI communication loop."""
    board = chess.Board()
    searcher: Optional[Searcher] = None
    is_initialized = False

    while True:
        line = stdin.readline()
        if not line:
            break

        log_command("GUI", line)
        parts = line.strip().split()
        cmd = parts[0] if parts else ""

        if cmd == "uci":
            send_command("id name RCN Engine", stdout)
            send_command("id author Jules", stdout)
            try:
                if searcher is None:
                    searcher = Searcher()
                is_initialized = True
            except Exception as e:
                log_command("ENGINE_ERROR", f"Failed to initialize Searcher: {e}")
                break
            send_command("uciok", stdout)
        elif cmd == "isready":
            if is_initialized:
                send_command("readyok", stdout)
        elif cmd == "ucinewgame":
            board.reset()
            if searcher:
                searcher.transposition_table.clear()
                searcher.killer_moves.clear()
        elif cmd == "position":
            handle_position(parts[1:], board)
        elif cmd == "go":
            if searcher:
                handle_go(parts, board, searcher, stdout)
        elif cmd == "quit":
            break

def main() -> None:
    """Main entry point that runs the UCI loop with standard I/O."""
    uci_loop(sys.stdin, sys.stdout)

def handle_position(parts: List[str], board: chess.Board) -> None:
    """Parses and applies the 'position' UCI command."""
    try:
        if parts[0] == "startpos":
            board.reset()
            moves_idx = 1
        elif parts[0] == "fen":
            fen = " ".join(parts[1:7])
            board.set_fen(fen)
            moves_idx = 7
        else: return
        if len(parts) > moves_idx and parts[moves_idx] == "moves":
            for move in parts[moves_idx+1:]: board.push_uci(move)
    except Exception as e:
        logging.error(f"Error in handle_position: {e}")

def calculate_search_time(wtime: int, btime: int, winc: int, binc: int, movestogo: Optional[int], turn: chess.Color) -> int:
    """Calculates the optimal search time in milliseconds."""
    time_left_ms = wtime if turn == chess.WHITE else btime
    increment_ms = winc if turn == chess.WHITE else binc

    if movestogo and movestogo > 0:
        allocated_time_ms = (time_left_ms // movestogo) + (increment_ms * 8 // 10)
    else:
        divider = 25 if time_left_ms > 20000 else 15
        allocated_time_ms = (time_left_ms // divider) + (increment_ms * 9 // 10)

    max_time_ms = time_left_ms * 8 // 10
    min_time_ms = 50

    final_time_ms = max(min_time_ms, min(allocated_time_ms, max_time_ms))
    return min(final_time_ms, time_left_ms)

def handle_go(parts: List[str], board: chess.Board, searcher: Searcher, stdout: IO[str] = sys.stdout) -> None:
    """Parses the 'go' command and initiates a search."""
    params = {parts[i]: int(parts[i+1]) for i in range(len(parts)-1) if parts[i] in ["wtime", "btime", "winc", "binc", "movestogo", "depth"]}
    time_limit = None
    if "wtime" in params:
        time_limit_ms = calculate_search_time(
            params.get("wtime", 0), params.get("btime", 0),
            params.get("winc", 0), params.get("binc", 0),
            params.get("movestogo"), board.turn
        )
        time_limit = time_limit_ms / 1000.0

    depth = params.get("depth", config.SEARCH_DEPTH)

    # Temporarily redirect stdout for the search's send_command calls
    original_stdout = sys.stdout
    sys.stdout = stdout
    best_move = searcher.search(board, depth, time_limit)
    sys.stdout = original_stdout # Restore stdout

    send_command(f"bestmove {best_move.uci() if best_move else '0000'}", stdout)

if __name__ == "__main__":
    main()
