import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data

# Critical: Import the correct policy output size from the move mapping utility
from scripts.move_utils import POLICY_OUTPUT_SIZE
from scripts.graph_utils import NUM_PIECE_TYPES

# --- Model Configuration ---
# Now 13 piece types: 12 pieces + 1 virtual turn node
NUM_NODE_TYPES = NUM_PIECE_TYPES
NODE_EMBEDDING_DIM = 64
# Now 3 edge types: ATTACKS, DEFENDS, TURN_CONNECTION
NUM_EDGE_FEATURES = 3
EDGE_EMBEDDING_DIM = 16
GAT_HIDDEN_CHANNELS = 128
GAT_HEADS = 4
# Total size of global features: turn (1) + castling (4) + en_passant (65) + halfmove (1) + fullmove (1) = 72
GLOBAL_FEATURES_DIM = 1 + 4 + 65 + 1 + 1
GLOBAL_EMBEDDING_DIM = 32

class RCNModel(nn.Module):
    """
    Relational Chess Net (RCN) - A Graph Attention Network for chess,
    updated to process rich global and node-level features.
    """
    def __init__(self):
        super(RCNModel, self).__init__()

        # --- 1. Embedding Layers ---
        self.node_embedding = nn.Embedding(NUM_NODE_TYPES, NODE_EMBEDDING_DIM)
        self.edge_embedding = nn.Embedding(NUM_EDGE_FEATURES, EDGE_EMBEDDING_DIM)
        self.global_feature_encoder = nn.Linear(GLOBAL_FEATURES_DIM, GLOBAL_EMBEDDING_DIM)

        # --- 2. Graph Encoder (GATv2) ---
        # Input dim: embedded piece type + file + rank + turn feature
        gat_input_dim = NODE_EMBEDDING_DIM + 3
        self.gat1 = GATv2Conv(
            gat_input_dim,
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
        # The input now includes both the graph embedding and the global feature embedding
        mlp_input_dim = GAT_HIDDEN_CHANNELS + GLOBAL_EMBEDDING_DIM

        self.value_head = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.policy_head = nn.Sequential(nn.Linear(mlp_input_dim, 256), nn.ReLU(), nn.Linear(256, POLICY_OUTPUT_SIZE))
        self.tactic_head = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.strategic_head = nn.Sequential(nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, data: Data):
        # 1. Process Global Features
        global_features = torch.cat([
            data.turn,
            data.castling_rights,
            data.en_passant,
            data.halfmove_clock,
            data.fullmove_number
        ], dim=-1)
        global_embedding = F.relu(self.global_feature_encoder(global_features))

        # 2. Process Node Features
        node_types = data.x[:, 0].long()
        node_rest_features = data.x[:, 1:] # file, rank, turn_feature
        embedded_types = self.node_embedding(node_types)
        x = torch.cat([embedded_types, node_rest_features], dim=1)

        # 3. Process Edge Features
        edge_attr = self.edge_embedding(data.edge_attr.long())

        # 4. Pass through Graph Encoder
        x = F.relu(self.norm1(self.gat1(x, data.edge_index, edge_attr)))
        x = F.relu(self.norm2(self.gat2(x, data.edge_index, edge_attr)))

        # 5. Global Pooling
        graph_embedding = global_mean_pool(x, data.batch)

        # 6. Combine Graph and Global Embeddings
        combined_embedding = torch.cat([graph_embedding, global_embedding], dim=1)

        # 7. Pass through Output Heads
        value_out = torch.tanh(self.value_head(combined_embedding))
        policy_out = self.policy_head(combined_embedding)
        tactic_out = torch.sigmoid(self.tactic_head(combined_embedding))
        strategic_out = torch.sigmoid(self.strategic_head(combined_embedding))

        return {
            'value': value_out.squeeze(-1),
            'policy': policy_out,
            'tactic': tactic_out.squeeze(-1),
            'strategic': strategic_out.squeeze(-1)
        }

if __name__ == '__main__':
    print("--- Testing RCNModel (Updated) ---")
    model = RCNModel()
    print("Model architecture:")
    print(model)

    # Test with a dummy graph that matches the new feature structure
    num_nodes = 33 # 32 pieces + 1 turn node
    # Node features: [piece_type, file, rank, turn_feature]
    dummy_x = torch.rand(num_nodes, 4, dtype=torch.float)
    dummy_x[:, 0] = torch.randint(0, NUM_NODE_TYPES, (num_nodes,))
    dummy_edge_index = torch.randint(0, num_nodes, (2, 128))
    dummy_edge_attr = torch.randint(0, NUM_EDGE_FEATURES, (128,))
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long)

    # Dummy global features
    dummy_turn = torch.tensor([[1.0]])
    dummy_castling = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    dummy_ep = torch.zeros(1, 65)
    dummy_ep[0, 64] = 1.0
    dummy_halfmove = torch.tensor([[0.1]])
    dummy_fullmove = torch.tensor([[10.0]])

    dummy_data = Data(
        x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, batch=dummy_batch,
        turn=dummy_turn, castling_rights=dummy_castling, en_passant=dummy_ep,
        halfmove_clock=dummy_halfmove, fullmove_number=dummy_fullmove
    )

    try:
        output = model(dummy_data)
        print("\nForward pass successful!")
        print("Output shapes:")
        for head, tensor in output.items():
            print(f"- {head}: {tensor.shape}")

        # The batch size is 1
        assert output['value'].shape == (1,)
        assert output['policy'].shape == (1, POLICY_OUTPUT_SIZE)
        assert output['tactic'].shape == (1,)
        assert output['strategic'].shape == (1,)
        print("\nOutput shapes are correct.")

    except Exception as e:
        print(f"\nAn error occurred during the forward pass: {e}")
