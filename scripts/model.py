"""
This module defines the RCNModel, a Graph Neural Network for chess.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from typing import Tuple

import config
from scripts.move_utils import MAX_FROM_SQUARES, MAX_TO_SQUARES, MAX_PROMOTION_PIECES

class RCNModel(nn.Module):
    """
    The Relational Chess Net (RCN) model.

    This model uses Graph Attention Networks (GATv2) to process a graph
    representation of a chess board. It outputs predictions for the position's
    value, the policy (next move), and tactical/strategic flags.

    Attributes:
        conv1 (GATv2Conv): First graph attention layer.
        norm1 (BatchNorm): Batch normalization after the first convolution.
        conv2 (GATv2Conv): Second graph attention layer.
        norm2 (BatchNorm): Batch normalization after the second convolution.
        conv3 (GATv2Conv): Third graph attention layer.
        norm3 (BatchNorm): Batch normalization after the third convolution.
        dropout (Dropout): Dropout layer for regularization.
        value_head (Sequential): Predicts the position's value (-1 to 1).
        policy_from_head (Sequential): Predicts the starting square of the move.
        policy_to_head (Sequential): Predicts the ending square of the move.
        policy_promo_head (Sequential): Predicts the promotion piece.
        tactic_head (Sequential): Predicts if the position is tactical.
        strategic_head (Sequential): Predicts if the position is strategic.
    """
    def __init__(self, in_channels: int, out_channels: int, num_edge_features: int, heads: int = config.GAT_HEADS):
        """
        Initializes the RCNModel.

        Args:
            in_channels: Number of input features for each node.
            out_channels: Number of output features from the final graph convolution.
            num_edge_features: Number of features for each edge.
            heads: Number of attention heads in the GAT layers.
        """
        super(RCNModel, self).__init__()
        self.conv1 = GATv2Conv(in_channels, 32, heads=heads, edge_dim=num_edge_features)
        self.norm1 = BatchNorm(32 * heads)
        self.conv2 = GATv2Conv(32 * heads, 64, heads=heads, edge_dim=num_edge_features)
        self.norm2 = BatchNorm(64 * heads)
        self.conv3 = GATv2Conv(64 * heads, out_channels, heads=1, concat=False, edge_dim=num_edge_features)
        self.norm3 = BatchNorm(out_channels)

        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)

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

        Args:
            data: A PyG Data or Batch object containing the graph representation.

        Returns:
            A tuple containing:
            - value (torch.Tensor): The predicted value of the position.
            - policy_logits (Tuple[torch.Tensor, ...]): A tuple of logits for
              'from' square, 'to' square, and promotion piece.
            - tactic_flag (torch.Tensor): The probability of the position being tactical.
            - strategic_flag (torch.Tensor): The probability of the position being strategic.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

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
