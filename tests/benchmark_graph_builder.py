# benchmark_graph_builder.py
# Micro-benchmark script for fen_to_graph_data_v2. Feeds a set of FENs and reports per-FEN times.
# Usage:
#   python benchmark_graph_builder.py sample_fens.txt
# Note: This script expects fen_to_graph_data_v2 to be importable in the Python path.

import time, sys, statistics
from pathlib import Path
import chess
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.fen_to_graph_data_v2 import fen_to_graph_data_v2

def benchmark(fen_list, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for fen in fen_list:
            board = chess.Board(fen.strip())
            _ = fen_to_graph_data_v2(board)
        t1 = time.perf_counter()
        times.append((t1 - t0) / len(fen_list))
    print("Per-FEN average times (seconds):")
    for i, t in enumerate(times):
        print(f" Run {i}: {t:.6f}s per FEN")
    print("Summary: mean {:.6f}s, median {:.6f}s, min {:.6f}s".format(statistics.mean(times), statistics.median(times), min(times)))
    return times

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python benchmark_graph_builder.py fens.txt")
        sys.exit(1)
    fens = [l.strip() for l in open(sys.argv[1], 'r', encoding='utf-8') if l.strip()]
    benchmark(fens)
