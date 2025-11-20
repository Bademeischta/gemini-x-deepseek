# fen_to_graph_data_v2.py
# Purpose: convert a python-chess Board into a torch_geometric.data.Data object
# using bitboard-based operations and minimal hand-engineered node/global features.
# Assumes: python-chess, torch, torch_geometric installed.
# Notes: X-Ray/pins are NOT computed explicitly. Simple, stable node & global features included.

import numpy as np
import torch
from torch_geometric.data import Data
import chess

# Mapping piece types -> one-hot index base (6 types * 2 colors = 12 features)
PIECE_TYPE_TO_IDX = {
    (chess.PAWN, True): 0, (chess.KNIGHT, True): 1, (chess.BISHOP, True): 2,
    (chess.ROOK, True): 3, (chess.QUEEN, True): 4, (chess.KING, True): 5,
    (chess.PAWN, False): 6, (chess.KNIGHT, False): 7, (chess.BISHOP, False): 8,
    (chess.ROOK, False): 9, (chess.QUEEN, False): 10, (chess.KING, False): 11,
}

def piece_feature_vector(piece):
    """Return 12-d one-hot for piece type+color, plus file (0..7) and rank (0..7) normed, and mobility (int)."""
    vec = np.zeros(12 + 2 + 1, dtype=np.float32)  # 12 piece types, file, rank, mobility
    idx = PIECE_TYPE_TO_IDX[(piece.piece_type, piece.color)]
    vec[idx] = 1.0
    return vec

def fen_to_graph_data_v2(board: chess.Board) -> Data:
    """
    Inputs: python-chess Board object (already validated)
    Output: torch_geometric.data.Data with fields:
      - x: [num_nodes, node_feat_dim]
      - edge_index: [2, num_edges]
      - edge_attr: [num_edges, edge_attr_dim] (one-hot: ATTACK=1, DEFEND=0)
      - y: optional (value/policy placeholder)
      - meta: dict-like metadata (optional)
    Design choices:
      - Node-per-piece graph (only existing pieces become nodes)
      - Edges: directed edges from attacker to target for all attacks computed via bitboards
      - No explicit X-Ray/pin edges (removed as per v2)
      - Global features appended separately (returned via `data.u` or in data.meta`)
    """
    # --- Collect pieces and assign node indices ---
    piece_map = board.piece_map()  # dict square -> Piece
    squares = sorted(piece_map.keys())  # squares that have pieces (0..63)
    num_nodes = len(squares)
    if num_nodes == 0:
        # Extremely unlikely for valid chess positions, but guard anyway
        return Data(x=torch.zeros((0, 15), dtype=torch.float32))

    sq_to_idx = {sq: i for i, sq in enumerate(squares)}

    # --- Node features ---
    # We'll build a numpy array and convert to torch.tensor at the end
    node_feats = []
    for sq in squares:
        piece = piece_map[sq]
        vec = np.zeros(12 + 2 + 1, dtype=np.float32)  # 12 piece-types, file, rank, mobility
        # piece type + color one-hot
        vec_idx = PIECE_TYPE_TO_IDX[(piece.piece_type, piece.color)]
        vec[vec_idx] = 1.0
        # file and rank (normalized to 0..1)
        vec[12] = chess.square_file(sq) / 7.0
        vec[13] = chess.square_rank(sq) / 7.0
        # mobility: number of attacked squares from this square (small int normalized)
        mobility = bin(board.attacks_mask(sq)).count("1")
        vec[14] = mobility / 27.0  # 27 is max theoretical attacks (conservative)
        node_feats.append(vec)
    x = torch.tensor(np.stack(node_feats, axis=0), dtype=torch.float32)  # [N, 15]

    # --- Edges (attacks/defends) ---
    # We'll create directed edges for every attack (from attacker node -> target node).
    srcs = []
    dsts = []
    edge_types = []  # 1 = attack (capture/target occupied by opposite color), 0 = move/defend (same color)

    for source_sq in squares:
        source_idx = sq_to_idx[source_sq]
        attacks_bb = board.attacks_mask(source_sq)  # bitboard
        # iterate over target squares in the attack mask using chess.SquareSet (efficient)
        for target_sq in chess.SquareSet(attacks_bb):
            if target_sq in sq_to_idx:
                target_idx = sq_to_idx[target_sq]
                srcs.append(source_idx)
                dsts.append(target_idx)
                # if target has piece of opposite color -> attack edge (1), else defend (0)
                target_piece = board.piece_at(target_sq)
                if target_piece is not None:
                    is_attack = (target_piece.color != board.piece_at(source_sq).color)
                    edge_types.append(1 if is_attack else 0)
                else:
                    edge_types.append(0)  # unlikely because we only create nodes for occupied squares

    if len(srcs) == 0:
        # No edges — isolated pieces (weird but possible)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        # one-hot edge attr for attack/defend
        edge_attr = torch.nn.functional.one_hot(torch.tensor(edge_types, dtype=torch.long), num_classes=2).float()

    # --- Global features (u vector) ---
    # Side to move: +1 white, -1 black
    stm = 1.0 if board.turn == chess.WHITE else -1.0
    # Material diff: sum(piece values white) - sum(piece values black) normalized
    V = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
    mat_white = 0
    mat_black = 0
    for sq, piece in piece_map.items():
        if piece.color == chess.WHITE:
            mat_white += V.get(piece.piece_type, 0)
        else:
            mat_black += V.get(piece.piece_type, 0)
    material_diff = (mat_white - mat_black) / 20.0  # normalize arbitrarily

    castle_wk = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    castle_wq = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    castle_bk = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    castle_bq = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    is_check = 1.0 if board.is_check() else 0.0

    u = torch.tensor([stm, material_diff, castle_wk, castle_wq, castle_bk, castle_bq, is_check], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # attach global features conveniently (PyG doesn't enforce u, but many people use data.u or data.global_attr)
    data.u = u
    # data.squares = squares  # helpful mapping back to board squares
    # data.sq_to_idx = sq_to_idx  # useful for debugging / mapping moves

    return data
