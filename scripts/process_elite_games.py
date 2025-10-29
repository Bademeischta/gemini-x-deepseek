import chess.pgn
import requests
import zstandard
import json
import os
from tqdm import tqdm

# --- Configuration ---
DATA_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-10.pgn.zst"
OUTPUT_DIR = "data"
OUTPUT_FILENAME = "kkk_subset_strategic.jsonl"
ZST_FILENAME = "lichess_db_standard_rated_2024-10.pgn.zst"
PGN_FILENAME = "lichess_db_standard_rated_2024-10.pgn"

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
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def decompress_zst(zst_filename, pgn_filename):
    """Decompresses a .zst file."""
    if os.path.exists(pgn_filename):
        print(f"File {pgn_filename} already exists. Skipping decompression.")
        return
    print(f"Decompressing {zst_filename} to {pgn_filename}...")
    with open(zst_filename, 'rb') as ifh, open(pgn_filename, 'wb') as ofh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_writer(ofh) as writer, tqdm(
            total=os.path.getsize(zst_filename), unit='iB', unit_scale=True, desc="Decompressing"
        ) as pbar:
            while True:
                chunk = ifh.read(16384)
                if not chunk:
                    break
                writer.write(chunk)
                pbar.update(len(chunk))

def process_games():
    """
    Processes the PGN file to extract strategic positions from high-ELO games.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    zst_path = os.path.join(OUTPUT_DIR, ZST_FILENAME)
    pgn_path = os.path.join(OUTPUT_DIR, PGN_FILENAME)

    # Step 1: Download and Decompress
    download_file(DATA_URL, zst_path)
    decompress_zst(zst_path, pgn_path)

    # Step 2: Process the PGN file
    positions_count = 0
    print(f"Processing games from {pgn_path}...")
    with open(pgn_path) as pgn_file, open(output_path, 'w') as out_file:
        with tqdm(total=TARGET_POSITIONS, desc="Extracting Positions") as pbar:
            while positions_count < TARGET_POSITIONS:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # End of file

                # Filter by ELO
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                except ValueError:
                    continue # Skip if ELO is not a valid integer

                if white_elo < MIN_ELO or black_elo < MIN_ELO:
                    continue

                # Iterate through moves and extract positions
                board = game.board()
                for i, move in enumerate(game.mainline_moves()):
                    move_num = (i // 2) + 1
                    board.push(move)

                    if MIN_MOVE_NUM <= move_num <= MAX_MOVE_NUM:
                        # Create the data point
                        data_point = {
                            "fen": board.fen(),
                            "strategic_flag": 1.0,
                            "tactic_flag": 0.0
                        }
                        out_file.write(json.dumps(data_point) + '\n')
                        positions_count += 1
                        pbar.update(1)

                        if positions_count >= TARGET_POSITIONS:
                            break

    # Clean up large files
    try:
        os.remove(pgn_path)
        os.remove(zst_path)
        print("Cleaned up temporary PGN and ZST files.")
    except OSError as e:
        print(f"Error cleaning up files: {e}")


    print(f"\nProcessing complete. Saved {positions_count} positions to {output_path}.")


if __name__ == "__main__":
    # This self-contained block allows for direct execution and verification.
    # It demonstrates the script's functionality before integration.
    process_games()
