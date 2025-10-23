
import csv
import json
import chess
import bz2
import requests
import os

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
        with bz2.open(compressed_path, "rb") as bz2f, open(decompressed_path, "wb") as f:
            f.write(bz2f.read())
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
            try:
                fen = row['FEN']
                moves = row['Moves'].split(' ')

                if not moves:
                    raise ValueError("Empty 'Moves' column")

                policy_target = moves[0]

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
