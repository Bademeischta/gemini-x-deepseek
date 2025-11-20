# scripts/clean_dataset.py
"""
This script implements a robust, idempotent data cleaning pipeline for the chess dataset.
It performs the following steps as outlined in the project documentation:
1.  Validates each line of a JSONL dataset for structural and chess-specific correctness.
2.  Applies a set of deterministic, safe repair rules for common errors.
3.  Removes lines with unfixable errors.
4.  Re-computes derived fields like policy index and legal move masks.
5.  Outputs a cleaned dataset, along with detailed audit logs and a checksum.
"""

import json
import chess
import hashlib
import os
import sys
from tqdm import tqdm
import csv

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.uci_index import uci_to_index_4096

def create_legal_move_mask(board: chess.Board) -> list:
    """Creates a boolean mask for legal moves (4096 indices) as a list."""
    mask = [False] * 4096
    for move in board.legal_moves:
        idx = uci_to_index_4096(move.uci())
        mask[idx] = True
    return mask

def clean_dataset(input_path: str):
    """
    Reads a JSONL dataset, cleans it according to predefined rules, and writes
    the output to new files.
    """
    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Define output paths
    cleaned_path = os.path.join(base_dir, f"{base_name}_cleaned.jsonl")
    removed_path = os.path.join(base_dir, f"{base_name}_removed.jsonl")
    log_path = os.path.join(base_dir, f"{base_name}_cleaning_log.csv")
    checksum_path = f"{cleaned_path}.sha256"

    stats = {
        "total_lines": 0,
        "kept": 0,
        "repaired": 0,
        "removed": 0,
        "errors": {}
    }

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(cleaned_path, 'w', encoding='utf-8') as f_cleaned, \
         open(removed_path, 'w', encoding='utf-8') as f_removed, \
         open(log_path, 'w', encoding='utf-8', newline='') as f_log:

        log_writer = csv.writer(f_log)
        log_writer.writerow(['line_number', 'status', 'reason', 'original_line', 'repaired_line'])

        for i, line in tqdm(enumerate(f_in), desc="Cleaning dataset"):
            stats["total_lines"] += 1
            original_line = line.strip()
            repaired_record = None

            # 1. JSON Parse Check
            try:
                record = json.loads(original_line)
            except json.JSONDecodeError:
                stats["removed"] += 1
                stats["errors"]["json_parse_error"] = stats["errors"].get("json_parse_error", 0) + 1
                log_writer.writerow([i + 1, 'removed', 'json_parse_error', original_line, ''])
                f_removed.write(json.dumps({"line": i + 1, "reason": "json_parse_error", "data": original_line}) + '\n')
                continue

            # 2. Required Fields Check
            if 'fen' not in record or 'policy_target' not in record:
                stats["removed"] += 1
                stats["errors"]["missing_fields"] = stats["errors"].get("missing_fields", 0) + 1
                log_writer.writerow([i + 1, 'removed', 'missing_fields', original_line, ''])
                f_removed.write(json.dumps({"line": i + 1, "reason": "missing_fields", "data": record}) + '\n')
                continue

            is_repaired = False

            # 3. FEN & Board Check
            try:
                board = chess.Board(record['fen'])
            except ValueError:
                stats["removed"] += 1
                stats["errors"]["invalid_fen"] = stats["errors"].get("invalid_fen", 0) + 1
                log_writer.writerow([i + 1, 'removed', 'invalid_fen', original_line, ''])
                f_removed.write(json.dumps({"line": i + 1, "reason": "invalid_fen", "data": record}) + '\\n')
                continue

            # 4. Policy Target Canonicalization
            policy_target = record['policy_target'].strip().replace('-', '').replace(' ', '')
            if policy_target != record['policy_target']:
                is_repaired = True
                repaired_record = record.copy()
                repaired_record['policy_target'] = policy_target

            # 5. UCI & Legality Check
            try:
                move = chess.Move.from_uci(policy_target)
                if move not in board.legal_moves:
                    # Try promotion normalization
                    if len(policy_target) == 4:
                        piece = board.piece_at(chess.Move.from_uci(policy_target).from_square)
                        if piece and piece.piece_type == chess.PAWN:
                            if (piece.color == chess.WHITE and chess.square_rank(move.from_square) == 6) or \
                               (piece.color == chess.BLACK and chess.square_rank(move.from_square) == 1):
                                test_move = chess.Move.from_uci(policy_target + 'q')
                                if test_move in board.legal_moves:
                                    policy_target += 'q'
                                    is_repaired = True
                                    if repaired_record is None: repaired_record = record.copy()
                                    repaired_record['policy_target'] = policy_target
                                    move = test_move

                if move not in board.legal_moves:
                    raise ValueError("Illegal move")

            except (ValueError, IndexError):
                stats["removed"] += 1
                stats["errors"]["illegal_or_invalid_move"] = stats["errors"].get("illegal_or_invalid_move", 0) + 1
                log_writer.writerow([i + 1, 'removed', 'illegal_or_invalid_move', original_line, ''])
                f_removed.write(json.dumps({"line": i + 1, "reason": "illegal_or_invalid_move", "data": record}) + '\n')
                continue

            # If we've reached here, the sample is valid.
            final_record = repaired_record if is_repaired else record

            # 6. Value Check
            if 'value' in final_record:
                try:
                    v = float(final_record['value'])
                    if not -100.0 <= v <= 100.0:
                        raise ValueError("Value out of plausible range")
                except (ValueError, TypeError):
                    stats["removed"] += 1
                    stats["errors"]["invalid_value"] = stats["errors"].get("invalid_value", 0) + 1
                    log_writer.writerow([i + 1, 'removed', 'invalid_value', original_line, ''])
                    f_removed.write(json.dumps({"line": i + 1, "reason": "invalid_value", "data": record}) + '\n')
                    continue

            # G. Recompute derived fields
            final_record['policy_index'] = uci_to_index_4096(final_record['policy_target'])
            final_record['legal_moves_mask'] = create_legal_move_mask(board)

            f_cleaned.write(json.dumps(final_record) + '\n')

            if is_repaired:
                stats["repaired"] += 1
                log_writer.writerow([i + 1, 'repaired', 'canonicalization', original_line, json.dumps(final_record)])
            else:
                stats["kept"] += 1
                log_writer.writerow([i + 1, 'kept', '', original_line, ''])

    # Finalize stats
    stats['removed'] = stats['total_lines'] - stats['kept'] - stats['repaired']

    # Calculate SHA256 checksum
    hasher = hashlib.sha256()
    with open(cleaned_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    checksum = hasher.hexdigest()
    with open(checksum_path, 'w') as f:
        f.write(checksum)

    # Print summary report
    print("\\n--- Dataset Cleaning Summary ---")
    print(f"Total lines processed: {stats['total_lines']}")
    print(f"  - Kept (unchanged): {stats['kept']}")
    print(f"  - Repaired: {stats['repaired']}")
    print(f"  - Removed: {stats['removed']}")
    percent_removed = (stats['removed'] / stats['total_lines'] * 100) if stats['total_lines'] > 0 else 0
    print(f"Percentage removed: {percent_removed:.2f}%")
    print(f"\\nTop error reasons for removal:")
    for reason, count in sorted(stats['errors'].items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"  - {reason}: {count}")
    print(f"\\nCleaned dataset written to: {cleaned_path}")
    print(f"SHA256 checksum: {checksum} (saved to {checksum_path})")
    print(f"Removed samples log: {removed_path}")
    print(f"Detailed cleaning log: {log_path}\\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/clean_dataset.py <path_to_dataset.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    clean_dataset(input_path)

if __name__ == '__main__':
    main()
