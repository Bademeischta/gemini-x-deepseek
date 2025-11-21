"""
This module implements the main UCI (Universal Chess Interface) engine logic,
including the search algorithm and communication with a GUI.
"""
import chess
import chess.polyglot
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
from scripts.mcts import BatchMCTS

# --- Logging Setup ---
# Logger is configured in setup_logging() to avoid issues in test environments.
logger = logging.getLogger()

def setup_logging():
    """Configures the logger to use a rotating file handler."""
    # Avoid adding handlers multiple times if this function is called more than once.
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return

    if config.ENGINE_LOG_PATH:
        os.makedirs(os.path.dirname(config.ENGINE_LOG_PATH), exist_ok=True)
        log_formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_handler = RotatingFileHandler(
            config.ENGINE_LOG_PATH,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT
        )
        log_handler.setFormatter(log_formatter)
        logger.addHandler(log_handler)
        logger.setLevel(logging.DEBUG)

def log_command(cmd_type: str, command: str) -> None:
    """Logs a command sent to or from the engine."""
    logger.info(f"{cmd_type}: {command.strip()}")

def send_command(command: str, stdout: IO[str] = sys.stdout) -> None:
    """Sends a command to the GUI and logs it."""
    stdout.write(command + '\n')
    stdout.flush()
    log_command("ENGINE", command)

def uci_loop(stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
    """The main UCI communication loop."""
    board = chess.Board()
    searcher: Optional[BatchMCTS] = None
    is_initialized = False

    # Initialize model and searcher outside the loop
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RCNModel(
            in_channels=15,
            out_channels=config.MODEL_OUT_CHANNELS,
            num_edge_features=2
        )

        if os.path.exists(config.MODEL_SAVE_PATH):
            try:
                state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
                model.load_state_dict(state_dict)
                print(f"✓ Model loaded from {config.MODEL_SAVE_PATH}", file=sys.stderr)
            except Exception as e:
                print(f"⚠ WARNING: Failed to load model weights: {e}", file=sys.stderr)
                print(f"⚠ Using randomly initialized model!", file=sys.stderr)
        else:
            print(f"⚠ WARNING: No model found at {config.MODEL_SAVE_PATH}", file=sys.stderr)
            print(f"⚠ Using randomly initialized model!", file=sys.stderr)

        model.to(device)
        model.eval()
        searcher = BatchMCTS(model, device=device)
        is_initialized = True

    except Exception as e:
        log_command("ENGINE_ERROR", f"CRITICAL: Failed to initialize Model/Searcher: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return  # Cannot proceed without model

    while True:
        line = stdin.readline()
        if not line:
            break

        log_command("GUI", line)
        parts = line.strip().split()
        cmd = parts[0] if parts else ""

        if cmd == "uci":
            send_command("id name RCN Engine v2 (MCTS)", stdout)
            send_command("id author Jules", stdout)
            send_command("uciok", stdout)
        elif cmd == "isready":
            if is_initialized:
                send_command("readyok", stdout)
        elif cmd == "ucinewgame":
            board.reset()
        elif cmd == "position":
            handle_position(parts[1:], board)
        elif cmd == "go":
            if searcher:
                handle_go(parts, board, searcher, stdout)
        elif cmd == "quit":
            break

def main() -> None:
    """Main entry point that runs the UCI loop with standard I/O."""
    setup_logging()
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

def handle_go(parts: List[str], board: chess.Board, searcher: BatchMCTS, stdout: IO[str] = sys.stdout) -> None:
    """Parses the 'go' command and initiates an MCTS search."""
    # A simple approach: use a fixed number of simulations.
    # A more advanced engine would dynamically adjust this based on time controls.
    num_simulations = 1024  # Default simulations

    params = {parts[i]: int(parts[i+1]) for i in range(len(parts)-1) if parts[i] in ["nodes", "movetime"]}

    if "nodes" in params:
        num_simulations = params["nodes"]
    elif "movetime" in params:
        # Crude conversion: assume ~50 nodes/sec on CPU to estimate simulations
        # This is a placeholder and should be benchmarked.
        time_limit_ms = params["movetime"]
        nodes_per_second = 50
        num_simulations = int((time_limit_ms / 1000.0) * nodes_per_second)

    send_command(f"info string Starting MCTS search with {num_simulations} simulations.", stdout)

    best_move = searcher.search(board, num_simulations=num_simulations)

    send_command(f"bestmove {best_move.uci() if best_move else '0000'}", stdout)

if __name__ == "__main__":
    main()
