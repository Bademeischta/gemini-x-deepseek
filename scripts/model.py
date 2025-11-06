import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from typing import Tuple

import config
from scripts.move_utils import MAX_FROM_SQUARES, MAX_TO_SQUARES, MAX_PROMOTION_PIECES
# Importieren Sie die Anzahl der Basisknoten-Merkmale (12 Typen)
from scripts.graph_utils import PIECE_TO_INT, TOTAL_NODE_FEATURES

# Die Anzahl der Basisknoten-Typen (Pawn..King * 2)
NUM_PIECE_TYPES = len(PIECE_TO_INT) # Sollte 12 sein

class RCNModel(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_edge_features: int, heads: int = config.GAT_HEADS):
        """
        Initializes the RCNModel.
        """
        super(RCNModel, self).__init__()

        # --- FIX: Embedding-Schicht wieder hinzufügen ---
        # Diese Schicht wandelt die Kategoriale ID (0-11) in einen dichten Vektor um.
        self.node_embedding = nn.Embedding(NUM_PIECE_TYPES, config.NODE_EMBEDDING_DIM)

        # Die Eingabekanäle für GATv2 sind jetzt:
        # (Embedding-Dimension) + (Restliche Features)
        # Restliche Features = TOTAL_NODE_FEATURES - 1 (da wir das 'piece_type'-Feature ersetzen)
        # z.B. 64 (Embedded) + 10 (Rest) = 74
        gat_input_dim = config.NODE_EMBEDDING_DIM + (TOTAL_NODE_FEATURES - 1)

        self.conv1 = GATv2Conv(gat_input_dim, 32, heads=heads, edge_dim=num_edge_features)
        self.norm1 = BatchNorm(32 * heads)
        self.conv2 = GATv2Conv(32 * heads, 64, heads=heads, edge_dim=num_edge_features)
        self.norm2 = BatchNorm(64 * heads)
        self.conv3 = GATv2Conv(64 * heads, out_channels, heads=1, concat=False, edge_dim=num_edge_features)
        self.norm3 = BatchNorm(out_channels)

        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)

        # --- Output Heads (unverändert) ---
        self.value_head = nn.Sequential(
            nn.Linear(out_channels, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh()
        )
        self.policy_from_head = nn.Sequential(nn.Linear(out_channels, MAX_FROM_SQUARES))
        self.policy_to_head = nn.Sequential(nn.Linear(out_channels, MAX_TO_SQUARES))
        self.policy_promo_head = nn.Sequential(nn.Linear(out_channels, MAX_PROMOTION_PIECES))

        self.tactic_head = nn.Sequential(nn.Linear(out_channels, 1), nn.Sigmoid())
        self.strategic_head = nn.Sequential(nn.Linear(out_channels, 1), nn.Sigmoid())

    def forward(self, data: Data) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the model.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # --- FIX: Merkmale aufteilen und 'piece_type' embedden ---

        # x[:, 0] ist der 'piece_type' (Index 0-11)
        node_piece_types = x[:, 0].long()

        # x[:, 1:] sind die restlichen 10 Merkmale (file, rank, turn, castling, etc.)
        node_other_features = x[:, 1:]

        # Wandle die kategorialen IDs in Vektoren um
        embedded_piece_types = self.node_embedding(node_piece_types)

        # Führe die Vektoren mit den restlichen Merkmalen zusammen
        x = torch.cat([embedded_piece_types, node_other_features], dim=1)

        # --- Restlicher Forward Pass (unverändert) ---

        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = nn.functional.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = nn.functional.elu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        node_features = nn.functional.elu(x)

        graph_embedding = global_mean_pool(node_features, batch)

        value = self.value_head(graph_embedding)

        policy_from_logits = self.policy_from_head(graph_embedding)
        policy_to_logits = self.policy_to_head(graph_embedding)
        policy_promo_logits = self.policy_promo_head(graph_embedding)

        tactic_flag = self.tactic_head(graph_embedding)
        strategic_flag = self.strategic_head(graph_embedding)

        return value, (policy_from_logits, policy_to_logits, policy_promo_logits), tactic_flag, strategic_flag
