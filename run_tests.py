
import subprocess
import sys
import os
import chess
import time
from typing import List, Optional

# --- Configuration ---
RCN_ENGINE_CMD = [sys.executable, "main.py", "run-engine"]
STOCKFISH_PATH = "./tools/stockfish"
NUM_GAMES = 20
THINK_TIME_SECONDS = 10.0

class Engine:
    """A simple UCI engine wrapper."""
    def __init__(self, command: List[str], name: str):
        self.name = name
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        self._send_command("uci")
        self._wait_for("uciok")

    def _send_command(self, command: str):
        if self.process.stdin:
            print(f"> {self.name}: {command}")
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

    def _wait_for(self, expected: str) -> List[str]:
        lines = []
        while self.process.stdout:
            line = self.process.stdout.readline().strip()
            if line:
                print(f"< {self.name}: {line}")
                lines.append(line)
                if expected in line:
                    break
        return lines

    def set_position(self, fen: str):
        self._send_command(f"position fen {fen}")

    def go(self, think_time: float) -> Optional[str]:
        self._send_command(f"go movetime {int(think_time * 1000)}")
        lines = self._wait_for("bestmove")
        for line in reversed(lines):
            if line.startswith("bestmove"):
                return line.split(" ")[1]
        return None

    def quit(self):
        self._send_command("quit")
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            print(f"Engine {self.name} did not quit gracefully and was killed.")

def play_game(white_engine: Engine, black_engine: Engine) -> Optional[str]:
    """Plays a single game between two engines."""
    board = chess.Board()
    print(f"\n--- New Game: {white_engine.name} (White) vs {black_engine.name} (Black) ---")

    while not board.is_game_over(claim_draw=True):
        engine_to_move = white_engine if board.turn == chess.WHITE else black_engine

        engine_to_move.set_position(board.fen())
        move_uci = engine_to_move.go(THINK_TIME_SECONDS)

        if not move_uci:
            print(f"Error: Engine {engine_to_move.name} failed to provide a move.")
            return None

        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                print(f"Error: Engine {engine_to_move.name} made an illegal move: {move_uci} in FEN {board.fen()}")
                return None
        except ValueError:
            print(f"Error: Engine {engine_to_move.name} returned an invalid move format: {move_uci}")
            return None

    result = board.result(claim_draw=True)
    print(f"Game finished. Result: {result}")
    return result

def main():
    """Main tournament logic."""
    print("--- Starting RCN Engine Validation Tournament ---")

    if not os.path.exists(STOCKFISH_PATH):
        print(f"Error: Stockfish not found at '{STOCKFISH_PATH}'", file=sys.stderr)
        sys.exit(1)

    rcn_engine = Engine(RCN_ENGINE_CMD, "RCN-Engine")
    stockfish_engine = Engine([STOCKFISH_PATH], "Stockfish")

    rcn_score = 0.0
    total_games = 0

    try:
        for i in range(NUM_GAMES // 2):
            # Game as White
            total_games += 1
            result_white = play_game(rcn_engine, stockfish_engine)
            if result_white == "1-0":
                rcn_score += 1.0
            elif result_white == "1/2-1/2":
                rcn_score += 0.5

            # Game as Black
            total_games += 1
            result_black = play_game(stockfish_engine, rcn_engine)
            if result_black == "0-1":
                rcn_score += 1.0
            elif result_black == "1/2-1/2":
                rcn_score += 0.5
    finally:
        rcn_engine.quit()
        stockfish_engine.quit()

    print("\n--- Tournament Finished ---")
    print(f"Final Score for RCN-Engine: {rcn_score} / {total_games}")

    target_score = total_games / 2.0
    if rcn_score > target_score:
        print("\n--- VALIDATION SUCCESSFUL ---")
        print("RCN Engine scored more than 50% against the benchmark.")
        sys.exit(0)
    else:
        print("\n--- VALIDATION FAILED ---")
        print("RCN Engine did not score more than 50% against the benchmark.")
        sys.exit(1)

if __name__ == "__main__":
    main()
