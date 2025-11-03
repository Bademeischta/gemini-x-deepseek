import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data

from scripts.move_utils import POLICY_HEAD_FROM_SIZE, POLICY_HEAD_TO_SIZE
from scripts.graph_utils import NUM_PIECE_TYPES, TOTAL_NODE_FEATURES, NUM_EDGE_FEATURES
import config

# --- Model Configuration ---
NODE_EMBEDDING_DIM = config.NODE_EMBEDDING_DIM
EDGE_EMBEDDING_DIM = config.EDGE_EMBEDDING_DIM
GAT_HIDDEN_CHANNELS = config.GAT_HIDDEN_CHANNELS
GAT_HEADS = config.GAT_HEADS
DROPOUT_RATE = config.DROPOUT_RATE

class RCNModel(nn.Module):
    def __init__(self):
        super(RCNModel, self).__init__()
        self.node_embedding = nn.Embedding(NUM_PIECE_TYPES, NODE_EMBEDDING_DIM)
        self.edge_embedding = nn.Embedding(NUM_EDGE_FEATURES, EDGE_EMBEDDING_DIM)

        gat_input_dim = NODE_EMBEDDING_DIM + (TOTAL_NODE_FEATURES - 1)
        self.gat1 = GATv2Conv(gat_input_dim, GAT_HIDDEN_CHANNELS, heads=GAT_HEADS, edge_dim=EDGE_EMBEDDING_DIM)
        self.bn1 = nn.BatchNorm1d(GAT_HIDDEN_CHANNELS * GAT_HEADS)
        self.dropout1 = nn.Dropout(p=DROPOUT_RATE)

        self.gat2 = GATv2Conv(GAT_HIDDEN_CHANNELS * GAT_HEADS, GAT_HIDDEN_CHANNELS, heads=GAT_HEADS, edge_dim=EDGE_EMBEDDING_DIM, concat=False)
        self.bn2 = nn.BatchNorm1d(GAT_HIDDEN_CHANNELS)
        self.dropout2 = nn.Dropout(p=DROPOUT_RATE)

        mlp_input_dim = GAT_HIDDEN_CHANNELS
        self.value_head = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.policy_head_from = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, POLICY_HEAD_FROM_SIZE))
        self.policy_head_to = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, POLICY_HEAD_TO_SIZE))
        self.tactic_head = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.strategic_head = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, data: Data):
        node_types = data.x[:, 0].long()
        node_rest_features = data.x[:, 1:]

        embedded_types = self.node_embedding(node_types)
        x = torch.cat([embedded_types, node_rest_features], dim=1)

        edge_attr = self.edge_embedding(data.edge_attr.long())

        x = self.gat1(x, data.edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.gat2(x, data.edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        graph_embedding = global_mean_pool(x, data.batch)

        value_out = torch.tanh(self.value_head(graph_embedding))
        policy_from_out = self.policy_head_from(graph_embedding)
        policy_to_out = self.policy_head_to(graph_embedding)
        tactic_out = torch.sigmoid(self.tactic_head(graph_embedding))
        strategic_out = torch.sigmoid(self.strategic_head(graph_embedding))

        return {
            'value': value_out.squeeze(-1),
            'policy_from': policy_from_out,
            'policy_to': policy_to_out,
            'tactic': tactic_out.squeeze(-1),
            'strategic': strategic_out.squeeze(-1)
        }

if __name__ == '__main__':
    print("--- Testing RCNModel with From-To Policy Head ---")
    model = RCNModel()
    print(model)

    num_nodes = 33
    dummy_x = torch.rand(num_nodes, TOTAL_NODE_FEATURES)
    dummy_x[:, 0] = torch.randint(0, NUM_PIECE_TYPES, (num_nodes,))
    dummy_edge_index = torch.randint(0, num_nodes, (2, 100))
    dummy_edge_attr = torch.randint(0, NUM_EDGE_FEATURES, (100,))
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long)

    dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, batch=dummy_batch)

    output = model(dummy_data)
    assert output['value'].shape == (1,)
    assert output['policy_from'].shape == (1, POLICY_HEAD_FROM_SIZE)
    assert output['policy_to'].shape == (1, POLICY_HEAD_TO_SIZE)
    print("\nForward pass successful with correct output shapes.")
