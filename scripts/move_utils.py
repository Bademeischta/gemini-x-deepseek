"""
Utilities for converting between UCI move strings and policy head targets.

This module handles the logic for the 'From-To-Promotion' policy head
representation used by the RCN model.
"""
import chess
from typing import Dict

MAX_FROM_SQUARES: int = 64
"""The size of the 'from' square policy head output."""

MAX_TO_SQUARES: int = 64
"""The size of the 'to' square policy head output."""

MAX_PROMOTION_PIECES: int = 4
"""The size of the promotion piece policy head output (N, B, R, Q)."""

def uci_to_policy_targets(uci_move_str: str) -> Dict[str, int]:
    """Converts a UCI move string into a dictionary of target indices.

    Args:
        uci_move_str: The move in UCI format (e.g., "e2e4", "a7a8q").

    Returns:
        A dictionary with 'from', 'to', and 'promo' keys.
        'from' and 'to' are integer indices (0-63).
        'promo' is an integer index (0-3) for N, B, R, Q, or -1 if not a promotion.
        Returns -1 for all keys on invalid input.
    """
    if not uci_move_str:
        return {'from': -1, 'to': -1, 'promo': -1}

    try:
        move = chess.Move.from_uci(uci_move_str)
        promo_idx: int = -1
        if move.promotion:
            promo_map: Dict[chess.PieceType, int] = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}
            promo_idx = promo_map.get(move.promotion, -1)

        return {
            'from': move.from_square,
            'to': move.to_square,
            'promo': promo_idx
        }
    except (chess.InvalidMoveError, chess.IllegalMoveError):
        return {'from': -1, 'to': -1, 'promo': -1}

def policy_targets_to_uci(from_square: int, to_square: int, board: chess.Board) -> str:
    """Converts from/to square indices back to a legal UCI move string.

    This function requires board context to determine if a move is a promotion
    and to validate its legality. It defaults to queen promotion if ambiguous.

    Args:
        from_square: The starting square index (0-63).
        to_square: The ending square index (0-63).
        board: The current chess.Board object for context.

    Returns:
        The legal move in UCI format, or '0000' if no legal move can be
        constructed from the given squares.
    """
    move = chess.Move(from_square, to_square)

    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN:
        if chess.square_rank(to_square) in [0, 7]:
            move.promotion = chess.QUEEN

    if move in board.legal_moves:
        return move.uci()

    # Try again with queen promotion in case it wasn't specified
    move.promotion = chess.QUEEN
    if move in board.legal_moves:
        return move.uci()

    return "0000"

if __name__ == '__main__':
    # This block is for self-contained validation of the script's logic.
    print("--- Testing Move Utilities ---")

    # Test promotion
    targets = uci_to_policy_targets("a7a8q")
    print(f"'a7a8q' -> {targets}")
    assert targets == {'from': chess.A7, 'to': chess.A8, 'promo': 3}

    # Test standard move
    targets_std = uci_to_policy_targets("e2e4")
    print(f"'e2e4' -> {targets_std}")
    assert targets_std == {'from': chess.E2, 'to': chess.E4, 'promo': -1}

    # Test roundtrip
    board_promo = chess.Board("rnbqkbnr/pPpppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    reconstructed_promo = policy_targets_to_uci(chess.B7, chess.A8, board_promo)
    print(f"Roundtrip for b7a8q -> {reconstructed_promo}")
    assert reconstructed_promo == "b7a8q"

    print("\nAll tests passed!")
