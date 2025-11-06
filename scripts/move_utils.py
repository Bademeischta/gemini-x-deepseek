import chess

# --- New Policy Head Configuration ---
MAX_FROM_SQUARES = 64
MAX_TO_SQUARES = 64
MAX_PROMOTION_PIECES = 4  # N, B, R, Q

def uci_to_policy_targets(uci_move_str: str) -> dict:
    """
    Converts a UCI move string into a dictionary of target indices for the policy heads.
    """
    if not uci_move_str:
        return {'from': -1, 'to': -1, 'promo': -1}

    try:
        move = chess.Move.from_uci(uci_move_str)
        promo_idx = -1
        if move.promotion:
            promo_map = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}
            promo_idx = promo_map.get(move.promotion, -1)

        return {
            'from': move.from_square,
            'to': move.to_square,
            'promo': promo_idx
        }
    except (chess.InvalidMoveError, chess.IllegalMoveError):
        return {'from': -1, 'to': -1, 'promo': -1}

def policy_targets_to_uci(from_square: int, to_square: int, board: chess.Board) -> str:
    """
    Converts from and to square indices back to a legal UCI move string.
    This is more complex as it needs the board context to determine promotions.
    """
    move = chess.Move(from_square, to_square)

    # Check if the move is a promotion and handle it
    if board.piece_at(from_square) and board.piece_at(from_square).piece_type == chess.PAWN:
        if chess.square_rank(to_square) == 0 or chess.square_rank(to_square) == 7:
            move.promotion = chess.QUEEN

    if move in board.legal_moves:
        return move.uci()

    move.promotion = chess.QUEEN
    if move in board.legal_moves:
        return move.uci()

    return "0000"

if __name__ == '__main__':
    print("--- Testing Move Utilities ---")

    # Test promotion
    targets = uci_to_policy_targets("a7a8q")
    print(f"'a7a8q' -> {targets}")
    assert targets == {'from': chess.A7, 'to': chess.A8, 'promo': 3}

    # Test standard move
    targets_std = uci_to_policy_targets("e2e4")
    print(f"'e2e4' -> {targets_std}")
    assert targets_std == {'from': chess.E2, 'to': chess.E4, 'promo': -1}

    print("\nAll tests passed!")
