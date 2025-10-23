import requests
import zstandard
import chess.pgn
import json
import math
import os
import time
import re

# --- Constants ---
PGN_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst"
DOWNLOAD_PATH = "data/lichess_db_standard_rated_2023-01.pgn.zst"
OUTPUT_FILE = "data/kkk_subset_strategic.jsonl"
ERROR_LOG_FILE = "data/error_log_strategic.txt"

# --- Filters ---
MIN_ELO = 2400
MIN_HALF_MOVES = 20
MIN_EVAL = -0.75
MAX_EVAL = 0.75
POSITION_LIMIT = 1_000_000

# --- Functions ---

def download_file_with_retry(url, dest_path, retries=3, delay=10):
    """Downloads a file with retries and streaming."""
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1} of {retries}...")
            with requests.get(url, stream=True, headers={'User-Agent': 'RCN-Data-Pipeline/1.0'}) as r:
                r.raise_for_status()
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download successful.")
            return
        except requests.exceptions.RequestException as e:
            print(f"Error during download: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("All download attempts failed.")
                # Exit the script if the download fails
                exit(1)

def normalize_eval(evaluation):
    """Normalizes a chess evaluation score using a sigmoid-like function."""
    # This is value = eval / sqrt(1 + eval^2)
    return evaluation / math.sqrt(1 + evaluation**2)

def process_pgn_stream():
    """Processes the PGN stream to extract strategic positions."""
    positions_found = 0
    games_processed = 0

    # Resume logic: Check if output file exists and count lines
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        with open(OUTPUT_FILE, 'r') as f:
            positions_found = sum(1 for _ in f)
        print(f"Resuming. Found {positions_found} positions in existing output file.")

    with open(DOWNLOAD_PATH, 'rb') as f_in, \
         open(OUTPUT_FILE, 'a') as f_out, \
         open(ERROR_LOG_FILE, 'a') as f_err:

        dctx = zstandard.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f_in)

        # We need a text wrapper to handle the PGN decoding
        import io
        pgn = io.TextIOWrapper(stream_reader, encoding='utf-8')

        while positions_found < POSITION_LIMIT:
            try:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    print("End of PGN file reached.")
                    break

                games_processed += 1
                if games_processed % 1000 == 0:
                    print(f"Games processed: {games_processed}, Positions found: {positions_found}", flush=True)

                # --- Elo Filter ---
                try:
                    white_elo = int(game.headers["WhiteElo"])
                    black_elo = int(game.headers["BlackElo"])
                    if white_elo < MIN_ELO or black_elo < MIN_ELO:
                        continue # Skip game
                except (ValueError, KeyError):
                    # Skip game if Elo is not available or not a valid integer
                    continue

                board = game.board()
                half_move_count = 0
                node = game # Start at the root node

                # Iterate through nodes in the mainline to access comments correctly
                for move in game.mainline_moves():
                    node = node.variation(0)
                    board.push(move)
                    half_move_count += 1

                    # --- Half-move Filter ---
                    if half_move_count <= MIN_HALF_MOVES:
                        continue

                    # --- Evaluation Filter ---
                    comment = node.comment

                    if "[%eval" not in comment:
                        continue # Skip position if no eval comment

                    # Use regex to find the evaluation, handling positive/negative floats/integers
                    match = re.search(r"\[%eval (-?\d+\.?\d*)\]", comment)
                    if not match:
                        continue # Skip if eval is a mate (#) or format is unexpected

                    try:
                        evaluation = float(match.group(1))
                    except (ValueError, IndexError):
                        f_err.write(f"Could not parse eval from comment: {comment} in game {game.headers.get('Site', '?')}\n")
                        continue

                    if MIN_EVAL <= evaluation <= MAX_EVAL:
                        # All filters passed, extract data
                        fen = board.fen()
                        normalized_value = normalize_eval(evaluation)

                        target_vector = {
                            "value": round(normalized_value, 4),
                            "policy_target": move.uci(),
                            "tactic_flag": 0.0,
                            "strategic_flag": 1.0,
                            "fen": fen
                        }

                        # Write to JSONL file with correct newline character
                        f_out.write(json.dumps(target_vector) + "\n")
                        positions_found += 1

                        if positions_found >= POSITION_LIMIT:
                            break # Exit the inner loop (moves)

                if positions_found >= POSITION_LIMIT:
                    break # Exit the outer loop (games)

            except Exception as e:
                # Log errors related to reading/parsing a game
                f_err.write(f"Error processing game #{games_processed}: {e}\n")
                continue # Skip to the next game

    print(f"\nProcessing finished. Target of {POSITION_LIMIT} positions reached or end of file.")
    print(f"Total games processed: {games_processed}")
    print(f"Total positions found: {positions_found}")

def main():
    """Main function to run the data processing pipeline."""
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DOWNLOAD_PATH), exist_ok=True)

    # Step 1: Download the file (if it doesn't exist)
    if not os.path.exists(DOWNLOAD_PATH):
        print(f"Downloading {PGN_URL}...")
        download_file_with_retry(PGN_URL, DOWNLOAD_PATH)
        print("Download complete.")
    else:
        print(f"File {DOWNLOAD_PATH} already exists. Skipping download.")

    # Step 2: Process the PGN file
    print("Processing PGN stream...")
    process_pgn_stream()
    print("Processing finished.")

if __name__ == "__main__":
    main()
