import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
import config
from scripts.move_utils import POLICY_OUTPUT_SIZE, MAX_FROM_SQUARES, MAX_TO_SQUARES, MAX_PROMOTION_PIECES

class RCNModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_edge_features, heads=config.GAT_HEADS):
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

    def forward(self, data):
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
