
import csv
import json
import chess
import bz2
import requests
import os
import shutil

def download_and_decompress(url, compressed_path, decompressed_path):
    """Downloads and decompresses the puzzle database."""
    if not os.path.exists(decompressed_path):
        print(f"Downloading {url}...", flush=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(compressed_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Decompressing {compressed_path}...", flush=True)
        with bz2.open(compressed_path, 'rb') as bz2f, open(decompressed_path, 'wb') as f:
            shutil.copyfileobj(bz2f, f)
        print("Decompression complete.", flush=True)

input_file = 'data/puzzles.csv'
output_file = 'data/kkk_subset_puzzles.jsonl'
error_log_file = 'error_log.txt'
processed_count = 0
column_names = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags"]
data_url = "https://database.lichess.org/lichess_db_puzzle.csv.bz2"
compressed_file = "puzzles.csv.bz2"


if not os.path.exists("data"):
    os.makedirs("data")

compressed_file_path = os.path.join("data", compressed_file)

download_and_decompress(data_url, compressed_file_path, input_file)

try:
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(error_log_file, 'w', encoding='utf-8') as errfile:

        reader = csv.DictReader(infile, fieldnames=column_names)

        print(f"Starting processing...")

        for i, row in enumerate(reader):
            if (i + 1) % 100000 == 0:
                print(f"Rows processed: {i+1}, Valid positions written: {processed_count}", flush=True)

            try:
                # Rating Filter
                rating = int(row['Rating'])
                if not 1800 <= rating <= 2400:
                    continue

                # FEN Validation
                try:
                    fen = row['FEN']
                    board = chess.Board(fen)
                except ValueError as e:
                    raise ValueError(f"Invalid FEN: {e}")

                moves = row['Moves'].split(' ')
                if not moves or not moves[0]:
                    raise ValueError("Empty 'Moves' column")

                # Move Validation
                policy_target = moves[0]
                try:
                    move = board.parse_uci(policy_target)
                except ValueError as e:
                    raise ValueError(f"Invalid move format: {policy_target}")

                if move not in board.legal_moves:
                    raise ValueError(f"Illegal move '{policy_target}' for FEN '{fen}'")

                turn = fen.split(' ')[1]
                value = 1.0 if turn == 'w' else -1.0

                target_data = {
                    "fen": fen,
                    "value": value,
                    "policy_target": policy_target,
                    "tactic_flag": 1.0,
                    "strategic_flag": 0.0
                }

                outfile.write(json.dumps(target_data) + '\n')
                processed_count += 1

                if processed_count >= 1000000:
                    print("Target of 1 million valid positions reached. Stopping.", flush=True)
                    break

            except Exception as e:
                error_message = f"Error in line {i+1} (CSV line {i+2}): {e}\nDATA: {row}\n"
                print(error_message)
                errfile.write(error_message)
                continue

        print(f"Processing complete. {processed_count} positions written.")

except FileNotFoundError:
    print(f"ERROR: Input file '{input_file}' not found.")
except Exception as e:
    print(f"A serious, unexpected error occurred: {e}")
