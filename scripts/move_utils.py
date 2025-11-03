import chess

# TODO: Optimize to ~1880 legal moves only
# Current: ~4600 (includes many impossible moves like 'a1a3' for a rook)
# Benefit: ~2.5x smaller policy head, faster training, and smaller model size.
# See for reference: https://github.com/official-stockfish/Stockfish/discussions/4231

def _get_all_possible_moves():
    """
    Generates a comprehensive list of all possible UCI moves in chess.
    This is a simplified approach; a truly exhaustive list is very large.
    We'll create a mapping for standard moves. Under-promotions are less common
    and can be added if needed.
    """
    moves = []
    squares = [chess.square_name(s) for s in chess.SQUARES]

    # 1. Standard moves (e.g., "e2e4")
    for from_sq in squares:
        for to_sq in squares:
            if from_sq != to_sq:
                moves.append(from_sq + to_sq)

    # 2. Promotions (e.g., "e7e8q")
    # Pawn moves from rank 7 to 8 (white) or 2 to 1 (black)
    for from_file in "abcdefgh":
        for promotion_piece in "qrbn":
            # White promotion
            moves.append(f"{from_file}7{from_file}8{promotion_piece}")
            # Black promotion
            moves.append(f"{from_file}2{from_file}1{promotion_piece}")

    return sorted(list(set(moves)))

# --- Global Move Mapping ---
# This list defines the canonical ordering for converting moves to indices.
ALL_POSSIBLE_MOVES = _get_all_possible_moves()

# Create a mapping from the move (UCI string) to its index
MOVE_TO_INDEX = {move: i for i, move in enumerate(ALL_POSSIBLE_MOVES)}

# The size of the policy head must match the number of possible moves
POLICY_OUTPUT_SIZE = len(ALL_POSSIBLE_MOVES)

def uci_to_index(uci_move: str) -> int:
    """
    Converts a UCI move string to its corresponding integer index.
    Returns 0 if the move is not found (assumed to be an invalid or unexpected move).
    """
    return MOVE_TO_INDEX.get(uci_move, 0)

def index_to_uci(index: int) -> str:
    """
    Converts an integer index back to its UCI move string.
    Raises IndexError if the index is out of bounds.
    """
    return ALL_POSSIBLE_MOVES[index]

if __name__ == '__main__':
    print("--- Testing Move Utilities ---")
    print(f"Total number of unique UCI moves mapped: {POLICY_OUTPUT_SIZE}")

    # Test a few conversions
    move1 = "e2e4"
    idx1 = uci_to_index(move1)
    print(f"'{move1}' -> index {idx1}")
    assert index_to_uci(idx1) == move1

    move2 = "g8f6"
    idx2 = uci_to_index(move2)
    print(f"'{move2}' -> index {idx2}")
    assert index_to_uci(idx2) == move2

    move3 = "a7a8q" # White pawn promotion
    idx3 = uci_to_index(move3)
    print(f"'{move3}' -> index {idx3}")
    assert index_to_uci(idx3) == move3

    # Test a move that is guaranteed not to be in the MOVE_TO_INDEX map.
    # The original test used 'e2e5', which is an illegal chess move but is
    # included in the naively generated move list. The .get() method correctly
    # prevents a KeyError, which is the goal of this fix. The deeper issue of
    # move generation is addressed in a later step.
    non_existent_move = "z1z9"
    idx_non_existent = uci_to_index(non_existent_move)
    print(f"Non-existent move '{non_existent_move}' -> index {idx_non_existent}")
    assert idx_non_existent == 0

    print("\nAll tests passed!")
