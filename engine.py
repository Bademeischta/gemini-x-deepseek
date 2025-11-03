import chess
import sys
import random
import logging
import os
import time
import torch
import config
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.model import RCNModel
from scripts.graph_utils import fen_to_graph_data
from scripts.move_utils import uci_to_index

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


def log_command(cmd_type, command):
    logging.info(f"{cmd_type}: {command.strip()}")

def send_command(command):
    sys.stdout.write(command + '\n')
    sys.stdout.flush()
    log_command("ENGINE", command)

class Searcher:
    def __init__(self, model_path=config.MODEL_SAVE_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RCNModel()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.nodes_searched = 0
        self.transposition_table = {}
        self.killer_moves = {}

    def _evaluate(self, board):
        graph_data = fen_to_graph_data(board.fen()).to(self.device)
        with torch.no_grad():
            value = self.model(graph_data)['value'].item()
        return value if board.turn == chess.WHITE else -value

    def _quiescence_search(self, board, alpha, beta, depth):
        if depth == 0:
            return self._evaluate(board)
        stand_pat = self._evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        for move in [m for m in board.legal_moves if board.is_capture(m)]:
            temp_board = board.copy()
            temp_board.push(move)
            score = -self._quiescence_search(temp_board, -beta, -alpha, depth - 1)
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _get_ordered_moves(self, board, depth, pv_move=None):
        def move_score(move):
            if pv_move and move == pv_move: return float('inf')
            score = 0.0
            if depth in self.killer_moves and move in self.killer_moves[depth]: score += 2.0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
                    score += 1.0 + ((piece_values.get(victim.piece_type, 0) * 10 - piece_values.get(attacker.piece_type, 0)) / 100.0)
            return score
        return sorted(list(board.legal_moves), key=move_score, reverse=True)

    def _negamax(self, board, depth, alpha, beta, color, start_time, time_limit):
        z_hash = chess.zobrist_hash(board)
        if z_hash in self.transposition_table and self.transposition_table[z_hash]['depth'] >= depth:
            entry = self.transposition_table[z_hash]
            if entry['flag'] == 'EXACT': return entry['value'], entry['move']
            if entry['flag'] == 'LOWERBOUND': alpha = max(alpha, entry['value'])
            elif entry['flag'] == 'UPPERBOUND': beta = min(beta, entry['value'])
            if alpha >= beta: return entry['value'], entry['move']

        if depth == 0 or (time_limit and (time.time() - start_time) > time_limit) or board.is_game_over(claim_draw=True):
            self.nodes_searched += 1
            if board.is_checkmate(): return -float('inf'), None
            if board.is_game_over(claim_draw=True): return 0, None
            return self._quiescence_search(board, alpha, beta, config.QUIESCENCE_SEARCH_DEPTH) * color, None

        best_value, best_move = -float('inf'), None
        for move in self._get_ordered_moves(board, depth):
            board.push(move)
            value, _ = self._negamax(board, depth - 1, -beta, -alpha, -color, start_time, time_limit)
            value = -value
            board.pop()
            if value > best_value:
                best_value, best_move = value, move
            alpha = max(alpha, value)
            if alpha >= beta:
                if depth not in self.killer_moves: self.killer_moves[depth] = []
                if move not in self.killer_moves[depth]: self.killer_moves[depth].insert(0, move)
                self.killer_moves[depth] = self.killer_moves[depth][:2]
                break

        flag = 'EXACT'
        if best_value <= alpha: flag = 'UPPERBOUND'
        elif best_value >= beta: flag = 'LOWERBOUND'
        self.transposition_table[z_hash] = {'depth': depth, 'value': best_value, 'move': best_move, 'flag': flag}
        return best_value, best_move

    def search(self, board, depth, time_limit):
        self.transposition_table, self.killer_moves, self.nodes_searched = {}, {}, 0
        start_time = time.time()
        best_move, pv = None, []
        color = 1 if board.turn == chess.WHITE else -1
        for d in range(1, depth + 1):
            try:
                best_value, best_move = self._negamax(board, d, -float('inf'), float('inf'), color, start_time, time_limit)
                pv = [best_move]
                temp_board = board.copy()
                temp_board.push(best_move)
                while chess.zobrist_hash(temp_board) in self.transposition_table:
                    entry = self.transposition_table[chess.zobrist_hash(temp_board)]
                    if not entry['move']: break
                    pv.append(entry['move'])
                    temp_board.push(entry['move'])

                send_command(f"info depth {d} score cp {int(best_value * 100 * color)} nodes {self.nodes_searched} time {int((time.time() - start_time) * 1000)} pv {' '.join([m.uci() for m in pv])}")
            except TimeoutError:
                break
        return best_move if best_move else (list(board.legal_moves)[0] if list(board.legal_moves) else None)

def main():
    board, searcher, is_initialized = chess.Board(), None, False
    while True:
        line = sys.stdin.readline()
        if not line: continue
        log_command("GUI", line)
        parts = line.strip().split()
        cmd = parts[0]

        if cmd == "uci":
            send_command("id name RCN Engine")
            send_command("id author Jules")
            try:
                searcher = Searcher()
                is_initialized = True
            except Exception as e:
                log_command("ENGINE_ERROR", f"Failed to initialize Searcher: {e}")
                break
            send_command("uciok")
        elif cmd == "isready":
            if is_initialized:
                send_command("readyok")
        elif cmd == "ucinewgame":
            board.reset()
            if searcher:
                searcher.transposition_table.clear()
                searcher.killer_moves.clear()
        elif cmd == "position":
            handle_position(parts[1:], board)
        elif cmd == "go":
            if searcher: handle_go(parts, board, searcher)
        elif cmd == "quit":
            break

def handle_position(parts, board):
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

def calculate_search_time(wtime, btime, winc, binc, movestogo, turn) -> int:
    """
    Calculates the optimal search time in milliseconds, using integer arithmetic.
    """
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

def handle_go(parts, board, searcher):
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
    best_move = searcher.search(board, depth, time_limit)
    send_command(f"bestmove {best_move.uci() if best_move else '0000'}")

if __name__ == "__main__":
    main()
