import chess.pgn
import requests
import zstandard
import json
import io
import os
from tqdm import tqdm

# --- Configuration ---
DATA_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-10.pgn.zst"
OUTPUT_DIR = "data"
OUTPUT_FILENAME = "kkk_subset_strategic.jsonl"
ZST_FILENAME = "lichess_db_standard_rated_2024-10.pgn.zst"
# PGN_FILENAME removed as it's no longer used

TARGET_POSITIONS = 500000
MIN_ELO = 2400
MIN_MOVE_NUM = 15  # Full moves
MAX_MOVE_NUM = 40  # Full moves

def download_file(url, local_filename):
    """Downloads a file from a URL with a progress bar."""
    if os.path.exists(local_filename):
        print(f"File {local_filename} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {local_filename}...")
    # Per memory, add a User-Agent header
    with requests.get(url, stream=True, headers={'User-Agent': 'RCN-Data-Pipeline/1.0'}) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def process_games():
    """
    Processes the PGN stream to extract strategic positions from high-ELO games.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    zst_path = os.path.join(OUTPUT_DIR, ZST_FILENAME)
    error_log_file = os.path.join(OUTPUT_DIR, "process_elite_games_error.log")

    # Step 1: Download the file
    download_file(DATA_URL, zst_path)

    # Step 2: Process the PGN stream
    positions_found = 0
    games_processed = 0
    print(f"Processing stream from {zst_path}...")

    # The core change: stream processing
    try:
        with open(zst_path, 'rb') as f_in, \
             open(output_path, 'w') as f_out, \
             open(error_log_file, 'w') as f_err:

            dctx = zstandard.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f_in)
            pgn = io.TextIOWrapper(stream_reader, encoding='utf-8')

            with tqdm(total=TARGET_POSITIONS, desc="Extracting Positions") as pbar:
                while positions_found < TARGET_POSITIONS:
                    games_processed += 1
                    try:
                        game = chess.pgn.read_game(pgn)
                        if game is None:
                            print("\nEnde der PGN-Datei erreicht.")
                            break

                        # Keep the original filtering logic
                        white_elo = int(game.headers.get("WhiteElo", 0))
                        black_elo = int(game.headers.get("BlackElo", 0))

                        if white_elo < MIN_ELO or black_elo < MIN_ELO:
                            continue

                        board = game.board()
                        for i, move in enumerate(game.mainline_moves()):
                            move_num = (i // 2) + 1
                            board.push(move)

                            if MIN_MOVE_NUM <= move_num <= MAX_MOVE_NUM:
                                data_point = {
                                    "fen": board.fen(),
                                    "strategic_flag": 1.0,
                                    "tactic_flag": 0.0
                                }
                                f_out.write(json.dumps(data_point) + '\n')
                                positions_found += 1
                                pbar.update(1)

                            if positions_found >= TARGET_POSITIONS:
                                break

                        if positions_found >= TARGET_POSITIONS:
                            break

                    except Exception as e:
                        f_err.write(f"Fehler bei der Verarbeitung von Spiel #{games_processed}: {e}\n")
                        continue
    finally:
        # Clean up the large downloaded file
        try:
            if os.path.exists(zst_path):
                os.remove(zst_path)
                print(f"\nCleaned up temporary ZST file: {zst_path}")
        except OSError as e:
            print(f"\nError cleaning up file {zst_path}: {e}")

    print(f"\nProcessing complete. Saved {positions_found} positions to {output_path}.")
    if os.path.exists(error_log_file) and os.path.getsize(error_log_file) > 0:
        print(f"Some non-fatal errors occurred. Check log: {error_log_file}")


if __name__ == "__main__":
    process_games()
