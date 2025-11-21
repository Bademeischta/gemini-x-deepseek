"""
DEPRECATED: This file is kept for backwards compatibility only.
All new code should use fen_to_graph_data_v2.py

This file now simply re-exports the new implementation.
"""
import warnings
from scripts.fen_to_graph_data_v2 import fen_to_graph_data_v2, NODE_FEATURES, EDGE_FEATURES

# Legacy constants (for old code that still imports them)
PIECE_TO_INT = {}  # Empty - not used in v2
EDGE_TYPE_TO_INT = {"ATTACKS": 0, "DEFENDS": 1}  # Simplified for v2
NUM_EDGE_FEATURES = EDGE_FEATURES
TOTAL_NODE_FEATURES = NODE_FEATURES  # Updated for v2

def fen_to_graph_data(fen: str):
    """
    DEPRECATED: Use fen_to_graph_data_v2 directly.

    This function is kept for backwards compatibility but will be removed in v3.
    """
    warnings.warn(
        "fen_to_graph_data is deprecated. Use fen_to_graph_data_v2 instead.",
        DeprecationWarning,
        stacklevel=2
    )
    import chess
    board = chess.Board(fen)
    return fen_to_graph_data_v2(board)
