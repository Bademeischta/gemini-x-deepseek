import chess

# --- New Policy Head Configuration ---
# The model will have two policy heads:
# 1. 'from_square': A distribution over the 64 possible starting squares.
# 2. 'to_square': A distribution over the 64 possible destination squares.
POLICY_HEAD_FROM_SIZE = 64
POLICY_HEAD_TO_SIZE = 64

# TODO: The current "From-To" representation does not explicitly handle promotions.
# For a move like 'e7e8q', the target will be (from=e7, to=e8). The network
# must implicitly learn to associate this move with a promotion based on the pawn's position.
# A future improvement could be to add a separate small head to predict the promotion piece.

def uci_to_policy_targets(uci_move_str: str) -> dict:
    """
    Converts a UCI move string into a dictionary of target indices for the policy heads.

    Args:
        uci_move_str: The move in UCI format (e.g., "e2e4", "a7a8q").

    Returns:
        A dictionary with 'from' and 'to' keys, containing the integer indices
        (0-63) for the respective squares. Returns {'from': -1, 'to': -1} for invalid
        or empty strings.
    """
    if not uci_move_str:
        return {'from': -1, 'to': -1} # Invalid target for loss function to ignore

    try:
        move = chess.Move.from_uci(uci_move_str)
        return {
            'from': move.from_square,
            'to': move.to_square
        }
    except chess.InvalidMoveError:
        # This can happen if the UCI string is malformed (e.g., "e2e9")
        return {'from': -1, 'to': -1}

def policy_targets_to_uci(from_square: int, to_square: int, board: chess.Board) -> str:
    """
    Converts from and to square indices back to a legal UCI move string.
    This is more complex as it needs the board context to determine promotions.
    """
    move = chess.Move(from_square, to_square)

    # Check if the move is a promotion and handle it
    if board.piece_at(from_square) and board.piece_at(from_square).piece_type == chess.PAWN:
        if chess.square_rank(to_square) == 0 or chess.square_rank(to_square) == 7:
            # Assume queen promotion for simplicity in this context
            move.promotion = chess.QUEEN

    # Ensure the move is legal before returning its UCI representation
    if move in board.legal_moves:
        return move.uci()

    # Fallback for pawn moves that might be promotions but weren't specified as such
    # (e.g., if the move was just 'e7e8' instead of 'e7e8q')
    move.promotion = chess.QUEEN
    if move in board.legal_moves:
        return move.uci()

    # If the move is still not legal, we cannot determine the correct UCI string
    # (this can happen with ambiguous moves or illegal predictions)
    return "0000" # UCI null move

if __name__ == '__main__':
    print("--- Testing New Move Utilities (From-To Representation) ---")

    # Test standard move
    targets1 = uci_to_policy_targets("e2e4")
    print(f"'e2e4' -> {targets1}")
    assert targets1 == {'from': chess.E2, 'to': chess.E4}

    # Test capture
    targets2 = uci_to_policy_targets("g1f3")
    print(f"'g1f3' -> {targets2}")
    assert targets2 == {'from': chess.G1, 'to': chess.F3}

    # Test promotion
    targets3 = uci_to_policy_targets("a7a8q")
    print(f"'a7a8q' -> {targets3}")
    assert targets3 == {'from': chess.A7, 'to': chess.A8}

    # Test invalid move
    targets4 = uci_to_policy_targets("e2e9")
    print(f"Invalid move 'e2e9' -> {targets4}")
    assert targets4 == {'from': -1, 'to': -1}

    # Test empty string
    targets5 = uci_to_policy_targets("")
    print(f"Empty string '' -> {targets5}")
    assert targets5 == {'from': -1, 'to': -1}

    # Test roundtrip (requires a board context)
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    uci_move = "e2e4"
    targets = uci_to_policy_targets(uci_move)
    reconstructed_uci = policy_targets_to_uci(targets['from'], targets['to'], board)
    print(f"\nRoundtrip test for 'e2e4': {targets} -> '{reconstructed_uci}'")
    assert uci_move == reconstructed_uci

    board_promo = chess.Board("rnbqkbnr/pPpppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    uci_promo = "b7a8q"
    targets_promo = uci_to_policy_targets(uci_promo)
    reconstructed_promo = policy_targets_to_uci(targets_promo['from'], targets_promo['to'], board_promo)
    print(f"Roundtrip test for promotion '{uci_promo}': {targets_promo} -> '{reconstructed_promo}'")
    assert uci_promo == reconstructed_promo

    print("\nAll tests passed!")
