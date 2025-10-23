import torch
import chess
from torch_geometric.data import Data

# --- Mappings ---

# Map piece type and color to a single integer
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
}
# There are 12 unique piece types (6 pieces x 2 colors)

# Map edge types to an integer
EDGE_TYPE_TO_INT = {
    "ATTACKS": 0,
    "DEFENDS": 1,
}

def fen_to_graph_data(fen: str) -> Data:
    """
    Converts a FEN string representing a chess position into a
    torch_geometric.data.Data object.

    Nodes: Each piece on the board is a node.
    Node Features: [piece_type (int), file (x), rank (y)]
    Edges: Represent attack and defense relationships between pieces.
    Edge Features: The type of relationship (ATTACKS, DEFENDS).
    """
    board = chess.Board(fen)

    node_features = []
    square_to_node_idx = {} # Map chess.Square to the index in our node list

    # --- 1. Node Creation ---
    # Iterate through all squares to find pieces and create nodes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get piece features
            piece_type_int = PIECE_TO_INT[(piece.piece_type, piece.color)]
            file = chess.square_file(square) # 0 (a) to 7 (h)
            rank = chess.square_rank(square) # 0 (1) to 7 (8)

            # Append features for this node
            node_features.append([piece_type_int, file, rank])

            # Map this square to its node index
            square_to_node_idx[square] = len(node_features) - 1

    # --- 2. Edge Creation ---
    edge_indices = []
    edge_attrs = []

    # Iterate again through the squares that have pieces (our nodes)
    for source_square, source_node_idx in square_to_node_idx.items():
        source_piece = board.piece_at(source_square)

        # Get all squares this piece attacks
        attacked_squares = board.attacks(source_square)

        for target_square in attacked_squares:
            # Check if the attacked square is occupied by another piece
            if target_square in square_to_node_idx:
                target_node_idx = square_to_node_idx[target_square]
                target_piece = board.piece_at(target_square)

                # Determine edge type
                if source_piece.color != target_piece.color:
                    edge_type = EDGE_TYPE_TO_INT["ATTACKS"]
                else:
                    edge_type = EDGE_TYPE_TO_INT["DEFENDS"]

                # Add the edge
                edge_indices.append([source_node_idx, target_node_idx])
                edge_attrs.append(edge_type)

    # --- 3. Convert to Tensors ---
    x = torch.tensor(node_features, dtype=torch.float)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        # Handle cases with no edges (e.g., K vs K)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.long)

    # Create the graph data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

if __name__ == '__main__':
    # Test with the starting position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    graph = fen_to_graph_data(start_fen)

    print("--- Testing fen_to_graph_data ---")
    print(f"FEN: {start_fen}")
    print(f"Graph object: {graph}")
    print(f"Number of nodes (pieces): {graph.num_nodes}")
    print(f"Node feature shape: {graph.x.shape}")
    print(f"Number of edges (attacks/defenses): {graph.num_edges}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    print(f"Edge attribute shape: {graph.edge_attr.shape}")

    # Correctly validate node features for the white rook at a1 without using out-of-scope variables
    a1_square_coords = torch.tensor([0, 0], dtype=torch.float) # file=0, rank=0
    a1_node_idx = -1
    for i, features in enumerate(graph.x):
        # features are [piece_type, file, rank]
        if torch.equal(features[1:], a1_square_coords):
            a1_node_idx = i
            break

    assert a1_node_idx != -1, "Node for square a1 not found!"

    a1_rook_features = graph.x[a1_node_idx]
    expected_features = torch.tensor([PIECE_TO_INT[(chess.ROOK, chess.WHITE)], 0, 0], dtype=torch.float)
    assert torch.equal(a1_rook_features, expected_features)
    print("\nNode feature validation successful for a1 rook.")

    # A more complex mid-game position
    mid_game_fen = "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11"
    mid_game_graph = fen_to_graph_data(mid_game_fen)
    print("\n--- Testing mid-game position ---")
    print(f"FEN: {mid_game_fen}")
    print(f"Graph object: {mid_game_graph}")
    print(f"Number of nodes: {mid_game_graph.num_nodes}")
    print(f"Number of edges: {mid_game_graph.num_edges}")
