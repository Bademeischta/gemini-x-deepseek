# validate_dataset.py
# Simple dataset auditor that checks for invalid FENs and illegal policy moves.
# Input: JSONL with fields `fen`, `policy_target` (UCI), `value` (float, optional)
# Output: prints summary and optionally writes corrupted-line indices to a CSV.

import json, sys
import chess

def audit_dataset(jsonl_path, max_errors_to_show=20):
    corrupted = []
    total = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total += 1
            line = line.strip()
            if not line:
                corrupted.append((i, "empty_line"))
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                corrupted.append((i, f"json_parse_error: {e}"))
                continue
            # Required fields check
            if 'fen' not in obj or 'policy_target' not in obj:
                corrupted.append((i, "missing_fields"))
                continue
            fen = obj['fen']
            move_str = obj['policy_target']
            try:
                board = chess.Board(fen)
            except Exception as e:
                corrupted.append((i, f"invalid_fen: {e}"))
                continue
            try:
                mv = chess.Move.from_uci(move_str)
            except Exception as e:
                corrupted.append((i, f"invalid_uci: {e}"))
                continue
            if mv not in board.legal_moves:
                corrupted.append((i, f"illegal_move: {move_str}"))
                continue
            # Value plausibility (optional)
            if 'value' in obj:
                try:
                    v = float(obj['value'])
                    if not -100.0 <= v <= 100.0:
                        corrupted.append((i, f"value_out_of_range: {v}"))
                except:
                    corrupted.append((i, "value_not_float"))
            # Stop early if too many errors
            if len(corrupted) >= max_errors_to_show:
                break
    print(f"Total lines checked: {total}, corrupted found: {len(corrupted)}")
    for idx, reason in corrupted[:max_errors_to_show]:
        print(f"Line {idx}: {reason}")
    return corrupted

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py dataset.jsonl")
        sys.exit(1)
    audit_dataset(sys.argv[1])
