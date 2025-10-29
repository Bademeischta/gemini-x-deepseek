import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data

# Critical: Import the correct policy output size from the move mapping utility
from scripts.move_utils import POLICY_OUTPUT_SIZE

# --- Model Configuration ---
NUM_NODE_FEATURES = 12
NODE_EMBEDDING_DIM = 64
NUM_EDGE_FEATURES = 2
EDGE_EMBEDDING_DIM = 32
GAT_HIDDEN_CHANNELS = 128
GAT_HEADS = 4

class RCNModel(nn.Module):
    """
    Relational Chess Net (RCN) - A Graph Attention Network for chess.
    """
    def __init__(self):
        super(RCNModel, self).__init__()

        # --- 1. Embedding Layers ---
        self.node_embedding = nn.Embedding(NUM_NODE_FEATURES, NODE_EMBEDDING_DIM)
        self.edge_embedding = nn.Embedding(NUM_EDGE_FEATURES, EDGE_EMBEDDING_DIM)

        # --- 2. Graph Encoder (GATv2) ---
        # The input dimension includes the embedded piece type and the two coordinate features.
        self.gat1 = GATv2Conv(
            NODE_EMBEDDING_DIM + 2,
            GAT_HIDDEN_CHANNELS,
            heads=GAT_HEADS,
            edge_dim=EDGE_EMBEDDING_DIM,
            concat=True
        )
        self.norm1 = nn.LayerNorm(GAT_HIDDEN_CHANNELS * GAT_HEADS)

        self.gat2 = GATv2Conv(
            GAT_HIDDEN_CHANNELS * GAT_HEADS,
            GAT_HIDDEN_CHANNELS,
            heads=GAT_HEADS,
            edge_dim=EDGE_EMBEDDING_DIM,
            concat=False
        )
        self.norm2 = nn.LayerNorm(GAT_HIDDEN_CHANNELS)

        # --- 3. Multi-Task Output Heads ---
        mlp_input_dim = GAT_HIDDEN_CHANNELS

        self.value_head = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.policy_head = nn.Sequential(nn.Linear(mlp_input_dim, 256), nn.ReLU(), nn.Linear(256, POLICY_OUTPUT_SIZE))
        self.tactic_head = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.strategic_head = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 1. Process & Embed Node Features
        node_types = x[:, 0].long()
        node_coords = x[:, 1:] # File and rank (already floats)

        embedded_types = self.node_embedding(node_types)

        x = torch.cat([embedded_types, node_coords], dim=1)

        edge_attr = self.edge_embedding(edge_attr.long())

        # 2. Pass through Graph Encoder
        x = F.relu(self.norm1(self.gat1(x, edge_index, edge_attr)))
        x = F.relu(self.norm2(self.gat2(x, edge_index, edge_attr)))

        # 3. Global Pooling
        graph_embedding = global_mean_pool(x, batch)

        # 4. Pass through Output Heads
        value_out = torch.tanh(self.value_head(graph_embedding))
        policy_out = self.policy_head(graph_embedding)
        tactic_out = torch.sigmoid(self.tactic_head(graph_embedding))
        strategic_out = torch.sigmoid(self.strategic_head(graph_embedding))

        return {
            'value': value_out.squeeze(-1),
            'policy': policy_out,
            'tactic': tactic_out.squeeze(-1),
            'strategic': strategic_out.squeeze(-1)
        }

if __name__ == '__main__':
    print("--- Testing RCNModel ---")
    model = RCNModel()
    print("Model architecture:")
    print(model)

    num_nodes = 32
    # Node features are [piece_type, file, rank]
    dummy_x = torch.rand(num_nodes, 3, dtype=torch.float)
    dummy_x[:, 0] = torch.randint(0, NUM_NODE_FEATURES, (num_nodes,))
    dummy_edge_index = torch.randint(0, num_nodes, (2, 128))
    dummy_edge_attr = torch.randint(0, NUM_EDGE_FEATURES, (128,))
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long)

    dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, batch=dummy_batch)

    try:
        output = model(dummy_data)
        print("\nForward pass successful!")
        print("Output shapes:")
        for head, tensor in output.items():
            print(f"- {head}: {tensor.shape}")

        assert output['value'].shape == (1,)
        assert output['policy'].shape == (1, POLICY_OUTPUT_SIZE)
        assert output['tactic'].shape == (1,)
        assert output['strategic'].shape == (1,)
        print("\nOutput shapes are correct.")

    except Exception as e:
        print(f"\nAn error occurred during the forward pass: {e}")
