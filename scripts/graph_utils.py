"""
Utilities for converting chess positions in FEN format to graph data structures
for use with PyTorch Geometric.
"""
import torch
import chess
from torch_geometric.data import Data
from typing import Dict, Tuple

PIECE_TO_INT: Dict[Tuple[chess.PieceType, chess.Color], int] = {
    (p_type, color): i for i, (p_type, color) in enumerate(
        (p, c) for c in (chess.WHITE, chess.BLACK) for p in
        (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
    )
}
"""Mapping from (piece_type, color) tuples to integer indices."""

EDGE_TYPE_TO_INT: Dict[str, int] = {"ATTACKS": 0, "DEFENDS": 1, "PIN": 2, "XRAY": 3}
"""Mapping from edge type names to integer indices."""

NUM_EDGE_FEATURES: int = len(EDGE_TYPE_TO_INT)
"""The total number of possible edge types."""

TOTAL_NODE_FEATURES: int = 11
"""The total number of features in a node tensor: piece, file, rank, turn, 4x castling, ep, 50-move, is_real."""

def fen_to_graph_data(fen: str) -> Data:
    """Converts a FEN string into a PyTorch Geometric Data object.

    This function creates a graph representation of a chess position where each
    piece is a node. Node features include piece identity, position, and global
    game state (turn, castling rights, etc.). Edges represent relationships
    like attacks, defends, pins, and x-rays.

    Args:
        fen: The chess position in Forsyth-Edwards Notation (FEN).

    Returns:
        A PyG Data object representing the position as a graph.
    """
    board = chess.Board(fen)
    turn: float = 1.0 if board.turn == chess.WHITE else -1.0

    castling_rights: list[float] = [
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
    ]
    half_move_clock: float = board.halfmove_clock / 100.0
    ep_square: float = (board.ep_square / 63.0) if board.ep_square else 0.0

    nodes: list[list[float]] = []
    sq_to_idx: Dict[chess.Square, int] = {}
    for i, (sq, piece) in enumerate(sorted(board.piece_map().items())):
        real_node_features: list[float] = [
            float(PIECE_TO_INT[(piece.piece_type, piece.color)]),
            float(chess.square_file(sq)),
            float(chess.square_rank(sq)),
            turn
        ] + castling_rights + [ep_square, half_move_clock, 1.0]
        nodes.append(real_node_features)
        sq_to_idx[sq] = i

    edges: list[list[int]] = []
    edge_attrs: list[int] = []
    for source_sq, source_idx in sq_to_idx.items():
        # Attacks/Defends
        for target_sq in board.attacks(source_sq):
            if target_sq in sq_to_idx:
                edges.append([source_idx, sq_to_idx[target_sq]])
                edge_attrs.append(EDGE_TYPE_TO_INT["ATTACKS" if board.color_at(target_sq) != board.color_at(source_sq) else "DEFENDS"])
        # Pins
        if board.is_pinned(board.color_at(source_sq), source_sq):
            for pinner_sq in board.pin(board.color_at(source_sq), source_sq):
                 if pinner_sq in sq_to_idx:
                    edges.append([sq_to_idx[pinner_sq], source_idx])
                    edge_attrs.append(EDGE_TYPE_TO_INT["PIN"])

        # X-Ray attacks logic
        piece = board.piece_at(source_sq)
        if piece and piece.piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
            for blocker_sq in board.attacks(source_sq):
                if board.piece_at(blocker_sq) and board.color_at(blocker_sq) != piece.color:
                    file_dir: int = chess.square_file(blocker_sq) - chess.square_file(source_sq)
                    rank_dir: int = chess.square_rank(blocker_sq) - chess.square_rank(source_sq)
                    step_file: int = 0 if file_dir == 0 else (1 if file_dir > 0 else -1)
                    step_rank: int = 0 if rank_dir == 0 else (1 if rank_dir > 0 else -1)

                    current_sq: chess.Square = blocker_sq
                    while True:
                        next_file: int = chess.square_file(current_sq) + step_file
                        next_rank: int = chess.square_rank(current_sq) + step_rank
                        if not (0 <= next_file <= 7 and 0 <= next_rank <= 7): break

                        current_sq = chess.square(next_file, next_rank)
                        if board.piece_at(current_sq):
                            if board.color_at(current_sq) != piece.color and current_sq in sq_to_idx:
                                edges.append([source_idx, sq_to_idx[current_sq]])
                                edge_attrs.append(EDGE_TYPE_TO_INT["XRAY"])
                            break

    edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.long)
    edge_attr_one_hot = torch.nn.functional.one_hot(edge_attr_tensor, num_classes=NUM_EDGE_FEATURES).float()

    return Data(
        x=torch.tensor(nodes, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
        edge_attr=edge_attr_one_hot if edge_attrs else torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float),
    )

if __name__ == '__main__':
    # This block is for self-contained validation of the script's logic.
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    g = fen_to_graph_data(fen)

    b = chess.Board(fen)
    assert g.x.shape[0] == b.occupied.bit_count(), "Node count should be equal to number of pieces"
    assert g.x.shape[1] == TOTAL_NODE_FEATURES, f"Expected {TOTAL_NODE_FEATURES} features per node"

    sq_map = {sq: i for i, (sq, p) in enumerate(sorted(b.piece_map().items()))}
    king_node_idx = sq_map[chess.E1]
    king_node = g.x[king_node_idx]

    assert king_node[3] == 1.0, "Turn should be White (1.0)"
    assert torch.all(king_node[4:8] == torch.tensor([1.0, 1.0, 1.0, 1.0])), "Castling rights incorrect"
    assert king_node[10] == 1.0, "Node should be marked as real"

    # Test X-Ray edge
    fen_xray = "8/8/8/4k3/4p3/4Q3/8/8 w - - 0 1"
    g_xray = fen_to_graph_data(fen_xray)
    b_xray = chess.Board(fen_xray)
    sq_map_xray = {sq: i for i, (sq, p) in enumerate(sorted(b_xray.piece_map().items()))}

    queen_idx = sq_map_xray[chess.E3]
    king_idx = sq_map_xray[chess.E5]

    found = any(s == queen_idx and t == king_idx and torch.argmax(g_xray.edge_attr[i]).item() == EDGE_TYPE_TO_INT["XRAY"]
                for i, (s, t) in enumerate(g_xray.edge_index.t().tolist()))

    assert found, "X-Ray edge from e3 to e5 not found"

    print("All tests passed!")
