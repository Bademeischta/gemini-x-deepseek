import chess
import sys
import random
import logging

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

def main():
    """Main UCI communication loop."""
    board = chess.Board()

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
            send_command("uciok")
        elif command == "isready":
            send_command("readyok")
        elif command == "ucinewgame":
            board.reset()
        elif command == "position":
            handle_position(parts[1:], board)
        elif command == "go":
            handle_go(board)
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

def handle_go(board):
    """Handles the 'go' UCI command with dummy logic."""
    if board.legal_moves:
        move = random.choice(list(board.legal_moves))
        send_command(f"bestmove {move.uci()}")

if __name__ == "__main__":
    main()
