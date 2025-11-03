import torch
import chess
from torch_geometric.data import Data

# --- Mappings ---
PIECE_TO_INT = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}
NUM_NODE_FEATURES_BASE = 12 # 6 types * 2 colors

EDGE_TYPE_TO_INT = {
    "ATTACKS": 0,
    "DEFENDS": 1,
    "PIN": 2, # Fesselung
}
NUM_EDGE_FEATURES_BASE = 3

def fen_to_graph_data(fen: str) -> Data:
    """
    Converts a FEN string representing a chess position into a
    torch_geometric.data.Data object.
    """
    board = chess.Board(fen)
    turn_feature = 1.0 if board.turn == chess.WHITE else -1.0

    node_features = []
    square_to_node_idx = {} # Map chess.Square to the index in our node list

    # --- 1. Node Creation ---
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type_int = PIECE_TO_INT[(piece.piece_type, piece.color)]
            file = chess.square_file(square)
            rank = chess.square_rank(square)

            # Node Features: [piece_type, file, rank, turn (als node feature)]
            node_features.append([piece_type_int, file, rank, turn_feature])
            square_to_node_idx[square] = len(node_features) - 1

    # --- 2. Edge Creation (Attacks / Defends) ---
    edge_indices = []
    edge_attrs = []

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

    # --- 3. Edge Creation (Pins) ---
    # Diese Logik ist SEPARAT, da "is_pinned" eine globale Eigenschaft ist.
    for target_square, target_node_idx in square_to_node_idx.items():
        target_piece = board.piece_at(target_square)

        # Prüfe, ob die Figur (target) gefesselt ist
        if board.is_pinned(target_piece.color, target_square):
            # Finde die Figur, die fesselt (pinner)
            pinner_square = board.pinner(target_piece.color, target_square)

            # pinner_square ist None, wenn der König direkt angegriffen wird (Schach)
            if pinner_square is not None and pinner_square in square_to_node_idx:
                source_node_idx = square_to_node_idx[pinner_square]

                # Füge eine "PIN"-Kante vom Fessler (source) zur gefesselten Figur (target) hinzu
                edge_indices.append([source_node_idx, target_node_idx])
                edge_attrs.append(EDGE_TYPE_TO_INT["PIN"])


    # --- 4. Convert to Tensors ---
    if node_features:
        x = torch.tensor(node_features, dtype=torch.float)
    else:
        # Leeres Brett
        x = torch.empty((0, 4), dtype=torch.float)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Globale Features (korrekt als [1, N] tensor)
    data.turn = torch.tensor([[turn_feature]], dtype=torch.float)

    return data

if __name__ == '__main__':
    print("\n--- Testing Pin Detection ---")
    # FEN: Weißer Turm auf a2 fesselt schwarze Dame auf e4 an schwarzen König auf e8.
    pin_fen = "4k3/8/8/8/4q3/8/R7/4K3 w - - 0 1"
    pin_graph = fen_to_graph_data(pin_fen)
    board = chess.Board(pin_fen)

    print(f"FEN: {pin_fen}")

    # Finde die Knoten-Indizes
    square_to_idx = {}
    for i, features in enumerate(pin_graph.x):
        f, r = int(features[1].item()), int(features[2].item())
        sq = chess.square(f, r)
        square_to_idx[sq] = i

    ra2_idx = square_to_idx[chess.A2]
    qe4_idx = square_to_idx[chess.E4]
    ke8_idx = square_to_idx[chess.E8]

    print(f"Rook (Pinner) Index: {ra2_idx}")
    print(f"Queen (Pinned) Index: {qe4_idx}")

    # Suche nach der PIN-Kante
    pin_edge_found = False
    pin_edge_type = EDGE_TYPE_TO_INT["PIN"]

    for i in range(pin_graph.num_edges):
        source = pin_graph.edge_index[0, i].item()
        target = pin_graph.edge_index[1, i].item()
        attr = pin_graph.edge_attr[i].item()

        if source == ra2_idx and target == qe4_idx and attr == pin_edge_type:
            pin_edge_found = True
            break

    assert pin_edge_found, "FEHLER: Die PIN-Kante vom Turm (a2) zur Dame (e4) wurde nicht erstellt!"
    print("Pin-Kanten-Validierung erfolgreich.")
