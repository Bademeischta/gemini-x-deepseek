import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from typing import Tuple

import config
# No longer importing from graph_utils, constants are self-contained.

# 6 piece types x 2 colors
NUM_PIECE_TYPES = 12

class RCNModel(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_edge_features: int, heads: int = config.GAT_HEADS):
        """
        Initializes the RCNModel.
        """
        super(RCNModel, self).__init__()

        # The new fen_to_graph_data_v2 provides a 15-dim feature vector directly.
        # No embedding layer is needed.
        NEW_NODE_FEATURES = 15
        # The new edge_attr is a 2-dim one-hot vector (attack/defend).
        NEW_EDGE_FEATURES = 2

        self.conv1 = GATv2Conv(NEW_NODE_FEATURES, 32, heads=heads, edge_dim=NEW_EDGE_FEATURES)
        self.norm1 = BatchNorm(32 * heads)
        self.conv2 = GATv2Conv(32 * heads, 64, heads=heads, edge_dim=NEW_EDGE_FEATURES)
        self.norm2 = BatchNorm(64 * heads)
        self.conv3 = GATv2Conv(64 * heads, out_channels, heads=1, concat=False, edge_dim=NEW_EDGE_FEATURES)
        self.norm3 = BatchNorm(out_channels)

        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)

        # --- Output Heads (unverändert) ---
        self.value_head = nn.Sequential(
            nn.Linear(out_channels, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh()
        )
        # New: Joint policy head for 4096 move logits
        self.policy_head = nn.Linear(out_channels, 4096)

        # Initialize policy head weights with small values to prevent extreme logits
        nn.init.uniform_(self.policy_head.weight, -config.POLICY_HEAD_INIT_SCALE, config.POLICY_HEAD_INIT_SCALE)
        nn.init.zeros_(self.policy_head.bias)

        self.tactic_head = nn.Sequential(nn.Linear(out_channels, 1))
        self.strategic_head = nn.Sequential(nn.Linear(out_channels, 1))

    def forward(self, data: Data) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the model.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # The feature vector x is now ready to be used directly.

        if torch.isnan(x).any():
            raise ValueError("NaN detected in input features!")

        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = nn.functional.elu(x)
        x = self.dropout(x)

        if torch.isnan(x).any():
            raise ValueError("NaN detected after conv1!")

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = nn.functional.elu(x)
        x = self.dropout(x)

        if torch.isnan(x).any():
            raise ValueError("NaN detected after conv2!")

        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        node_features = nn.functional.elu(x)

        if torch.isnan(node_features).any():
            raise ValueError("NaN detected after conv3!")

        graph_embedding = global_mean_pool(node_features, batch)

        if torch.isnan(graph_embedding).any():
            raise ValueError("NaN detected after pooling!")

        value = self.value_head(graph_embedding)
        policy_logits = self.policy_head(graph_embedding)

        # Validate policy outputs
        if torch.isnan(policy_logits).any():
            raise ValueError("NaN in policy_logits!")

        tactic_flag = self.tactic_head(graph_embedding)
        strategic_flag = self.strategic_head(graph_embedding)

        return value, policy_logits, tactic_flag, strategic_flag
