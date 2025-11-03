import torch
import chess
from torch_geometric.data import Data
import numpy as np

# --- Mappings ---

# Map piece type and color to a single integer. Add a virtual piece for the turn node.
PIECE_TO_INT = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
    "TURN_NODE": 12, # Virtual piece type for the turn indicator node
}

# Map edge types to an integer
EDGE_TYPE_TO_INT = {
    "ATTACKS": 0,
    "DEFENDS": 1,
    "TURN_CONNECTION": 2, # Edge type for the virtual turn node
}

NUM_PIECE_TYPES = len(PIECE_TO_INT) # Should be 13 (0-12)
NUM_FEATURES_PER_NODE = 4 # piece_type, file, rank, turn_feature

def fen_to_graph_data(fen: str) -> Data:
    """
    Converts a FEN string into a torch_geometric.data.Data object with rich features.

    Nodes: Each piece on the board + a virtual 'turn' node.
    Node Features: [piece_type, file, rank, turn_feature]
    Global Graph Attributes:
        - turn: Single float indicating whose turn it is.
        - castling_rights: 4-bit tensor for castling availability.
        - en_passant: One-hot encoded tensor (64+1) for the en passant square.
        - halfmove_clock: Normalized halfmove clock.
        - fullmove_number: Fullmove number.
    Edges: Represent attack, defense, and turn relationships.
    """
    board = chess.Board(fen)
    turn_feature = 1.0 if board.turn == chess.WHITE else -1.0

    node_features = []
    square_to_node_idx = {}

    # --- 1. Piece Node Creation ---
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type_int = PIECE_TO_INT[(piece.piece_type, piece.color)]
            file = chess.square_file(square)
            rank = chess.square_rank(square)

            # ANSATZ 2: Node-level turn feature
            node_features.append([piece_type_int, file, rank, turn_feature])
            square_to_node_idx[square] = len(node_features) - 1

    # --- 2. Virtual Turn Node Creation (ANSATZ 3) ---
    turn_node_idx = len(node_features)
    # The turn node is placed at the center of the board (3.5, 3.5) for visualization/embedding purposes.
    turn_node_features = [PIECE_TO_INT["TURN_NODE"], 3.5, 3.5, turn_feature]
    node_features.append(turn_node_features)

    # --- 3. Edge Creation ---
    edge_indices = []
    edge_attrs = []

    # Piece-to-piece edges (Attacks/Defends)
    for source_square, source_node_idx in square_to_node_idx.items():
        source_piece = board.piece_at(source_square)
        attacked_squares = board.attacks(source_square)
        for target_square in attacked_squares:
            if target_square in square_to_node_idx:
                target_node_idx = square_to_node_idx[target_square]
                target_piece = board.piece_at(target_square)
                edge_type = EDGE_TYPE_TO_INT["ATTACKS"] if source_piece.color != target_piece.color else EDGE_TYPE_TO_INT["DEFENDS"]
                edge_indices.append([source_node_idx, target_node_idx])
                edge_attrs.append(edge_type)

    # Turn-node-to-piece edges
    for piece_node_idx in range(turn_node_idx): # All nodes except the turn node itself
        edge_indices.append([turn_node_idx, piece_node_idx])
        edge_attrs.append(EDGE_TYPE_TO_INT["TURN_CONNECTION"])

    # --- 4. Global Feature Creation ---
    # All global features must be 2D tensors [1, num_features] for correct batching.

    # ANSATZ 1: Global graph attribute for turn
    turn_global = torch.tensor([[turn_feature]], dtype=torch.float32)

    # Castling rights
    castling_rights = torch.tensor([[
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]], dtype=torch.float32)

    # En passant square (one-hot encoding)
    en_passant_vec = torch.zeros(1, 65, dtype=torch.float32)
    if board.ep_square:
        en_passant_vec[0, board.ep_square] = 1.0
    else:
        en_passant_vec[0, 64] = 1.0 # Index 64 for no en passant square

    # Halfmove clock (normalized) and fullmove number
    halfmove_clock = torch.tensor([[board.halfmove_clock / 100.0]], dtype=torch.float32)
    fullmove_number = torch.tensor([[board.fullmove_number]], dtype=torch.float32)

    # --- 5. Convert to Tensors and create Data object ---
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long) if edge_attrs else torch.empty((0,), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        # Global attributes
        turn=turn_global,
        castling_rights=castling_rights,
        en_passant=en_passant_vec,
        halfmove_clock=halfmove_clock,
        fullmove_number=fullmove_number
    )

    return data

if __name__ == '__main__':
    # Test with the starting position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    graph = fen_to_graph_data(start_fen)

    print("--- Testing fen_to_graph_data (Updated) ---")
    print(f"FEN: {start_fen}")
    print(f"Graph object: {graph}")

    # 1. Validate Node Information
    assert graph.num_nodes == 33, f"Expected 32 pieces + 1 turn node, but got {graph.num_nodes}"
    assert graph.x.shape[1] == NUM_FEATURES_PER_NODE, f"Expected {NUM_FEATURES_PER_NODE} node features, but got {graph.x.shape[1]}"
    print(f"Correct number of nodes (33) and node features ({NUM_FEATURES_PER_NODE}) found.")

    # 2. Validate Turn Feature in Nodes (Ansatz 2)
    # For 'w' (white to move), the turn feature should be 1.0
    assert torch.all(graph.x[:, 3] == 1.0).item(), "Turn feature in nodes should be 1.0 for white to move."
    print("Node-level turn feature is correct.")

    # 3. Validate Virtual Turn Node (Ansatz 3)
    turn_node_features = graph.x[-1] # The last node is the turn node
    assert turn_node_features[0] == PIECE_TO_INT["TURN_NODE"], "Virtual turn node has incorrect piece type."
    print("Virtual turn node is correctly identified.")

    # 4. Validate Global Attributes (Ansatz 1 and others)
    assert graph.turn.shape == (1, 1) and graph.turn.item() == 1.0, "Global turn attribute is incorrect."
    expected_castling = torch.tensor([[True, True, True, True]], dtype=torch.float32)
    assert torch.equal(graph.castling_rights, expected_castling), "Castling rights are incorrect."
    assert graph.en_passant.shape == (1, 65) and graph.en_passant[0, 64] == 1.0, "En passant vector is incorrect."
    assert graph.halfmove_clock.item() == 0.0, "Halfmove clock should be 0."
    assert graph.fullmove_number.item() == 1.0, "Fullmove number should be 1."
    print("All global graph attributes are correct for the starting FEN.")

    # 5. Validate Edges
    # 32 piece nodes, each connected to the turn node -> 32 edges
    # Plus piece-to-piece attacks/defenses
    num_turn_node_edges = 32
    assert graph.num_edges > num_turn_node_edges, f"Expected more than {num_turn_node_edges} edges, but got {graph.num_edges}."
    print(f"Edge count ({graph.num_edges}) seems reasonable.")

    print("\n--- All Tests Passed Successfully! ---")

    # A more complex mid-game position
    mid_game_fen = "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11"
    mid_game_graph = fen_to_graph_data(mid_game_fen)
    print("\n--- Testing mid-game position ---")
    print(f"FEN: {mid_game_fen}")
    print(f"Graph object: {mid_game_graph}")
    print(f"Number of nodes: {mid_game_graph.num_nodes}") # Should be 28 pieces + 1 turn node = 29
    assert mid_game_graph.num_nodes == 29
    print(f"Number of edges: {mid_game_graph.num_edges}")
    assert mid_game_graph.turn.item() == -1.0, "Turn should be black (-1.0) in mid-game FEN."
    assert torch.all(mid_game_graph.x[:, 3] == -1.0).item(), "Node-level turn feature should be -1.0."
    print("Mid-game position validation successful.")
