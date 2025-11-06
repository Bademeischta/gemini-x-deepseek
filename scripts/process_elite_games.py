import chess.pgn
import requests
import zstandard
import json
import io
import os
from tqdm import tqdm
import config  # Import the centralized config

# --- Configuration ---
DATA_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-10.pgn.zst"
# OUTPUT_DIR, OUTPUT_FILENAME are now sourced from config
ZST_FILENAME = "lichess_db_standard_rated_2024-10.pgn.zst"

TARGET_POSITIONS = 500000
MIN_ELO = 2400
MIN_MOVE_NUM = 15  # Full moves
MAX_MOVE_NUM = 40  # Full moves
SAVE_INTERVAL = 10000 # Save progress every 10,000 games

def download_file(url, local_filename):
    """Downloads a file from a URL with a progress bar."""
    if os.path.exists(local_filename):
        print(f"File {local_filename} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {local_filename}...")
    with requests.get(url, stream=True, headers={'User-Agent': 'RCN-Data-Pipeline/1.0'}) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def load_processing_state():
    """Loads the processing state from a JSON file."""
    if os.path.exists(config.PROCESSING_STATE_PATH):
        with open(config.PROCESSING_STATE_PATH, 'r') as f:
            state = json.load(f)
            # Convert list back to set for efficient lookups
            state['unique_fens'] = set(state.get('unique_fens', []))
            return state
    return {"byte_offset": 0, "positions_found": 0, "unique_fens": set()}

def save_processing_state(offset, positions, fens_set):
    """Saves the processing state to a JSON file."""
    state = {
        "byte_offset": offset,
        "positions_found": positions,
        # Convert set to list for JSON serialization
        "unique_fens": list(fens_set)
    }
    with open(config.PROCESSING_STATE_PATH, 'w') as f:
        json.dump(state, f)
    # print(f"Checkpoint saved: Offset={offset}, Positions={positions}") # Optional: for debugging

def process_games():
    """
    Processes the PGN stream to extract strategic positions from high-ELO games,
    with logic to resume from a saved state.
    """
    # --- 0. Setup ---
    # Create directories based on the actual file paths, which might be patched in tests.
    if config.DATA_STRATEGIC_PATH:
        os.makedirs(os.path.dirname(config.DATA_STRATEGIC_PATH), exist_ok=True)
    if config.PROCESSING_STATE_PATH:
        os.makedirs(os.path.dirname(config.PROCESSING_STATE_PATH), exist_ok=True)

    zst_path = os.path.join(os.path.dirname(config.DATA_STRATEGIC_PATH or '.'), ZST_FILENAME)
    error_log_file = os.path.join(config.DATA_DIR, "process_elite_games_error.log")

    # Step 1: Download the file
    download_file(DATA_URL, zst_path)

    # Step 2: Load state and set up for processing
    state = load_processing_state()
    start_offset = state['byte_offset']
    positions_found = state['positions_found']
    unique_fens = state['unique_fens']
    games_processed = 0

    print(f"Resuming processing from byte offset {start_offset}. {positions_found} positions already found.")
    print(f"Processing stream from {zst_path}...")

    # The core change: stream processing with resume capability
    f_in = None # Define here to be accessible in finally
    try:
        with open(zst_path, 'rb') as f_in, \
             open(config.DATA_STRATEGIC_PATH, 'a') as f_out, \
             open(error_log_file, 'a') as f_err:

            f_in.seek(start_offset)
            dctx = zstandard.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f_in)
            pgn = io.TextIOWrapper(stream_reader, encoding='utf-8')

            with tqdm(total=TARGET_POSITIONS, initial=positions_found, desc="Extracting Positions") as pbar:
                while positions_found < TARGET_POSITIONS:
                    games_processed += 1
                    try:
                        game = chess.pgn.read_game(pgn)
                        if game is None:
                            print("\nEnd of PGN stream reached.")
                            break

                        white_elo = int(game.headers.get("WhiteElo", 0))
                        black_elo = int(game.headers.get("BlackElo", 0))

                        if white_elo < MIN_ELO or black_elo < MIN_ELO:
                            continue

                        board = game.board()
                        for i, move in enumerate(game.mainline_moves()):
                            move_num = (i // 2) + 1
                            board.push(move)

                            if MIN_MOVE_NUM <= move_num <= MAX_MOVE_NUM:
                                fen = board.fen()
                                if fen not in unique_fens:
                                    unique_fens.add(fen)
                                    data_point = {
                                        "fen": fen, "strategic_flag": 1.0, "tactic_flag": 0.0
                                    }
                                    f_out.write(json.dumps(data_point) + '\n')
                                    positions_found += 1
                                    pbar.update(1)

                            if positions_found >= TARGET_POSITIONS:
                                break

                        if games_processed % SAVE_INTERVAL == 0:
                            current_offset = f_in.tell()
                            save_processing_state(current_offset, positions_found, unique_fens)

                        if positions_found >= TARGET_POSITIONS:
                            break

                    except Exception as e:
                        f_err.write(f"Error processing game #{games_processed}: {e}\n")
                        continue

    except (KeyboardInterrupt, Exception) as e:
        print(f"\nInterruption detected ({type(e).__name__}). Saving final state...")
    finally:
        # Save final state on exit, whether successful or interrupted
        if f_in and not f_in.closed:
            final_offset = f_in.tell()
            save_processing_state(final_offset, positions_found, unique_fens)
            print(f"Final state saved. Offset: {final_offset}, Positions: {positions_found}")
        else: # Handle case where f_in failed to open
             save_processing_state(start_offset, positions_found, unique_fens)
             print(f"Could not get final offset. Resaved state at offset {start_offset}.")


    print(f"\nProcessing complete or paused. Saved {positions_found} total positions to {config.DATA_STRATEGIC_PATH}.")
    if os.path.exists(error_log_file) and os.path.getsize(error_log_file) > 0:
        print(f"Some non-fatal errors occurred. Check log: {error_log_file}")


if __name__ == "__main__":
    process_games()
