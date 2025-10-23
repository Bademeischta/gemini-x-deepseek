import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data

# --- Model Configuration ---
# These are example dimensions. The final dimensions will be tuned.
NUM_NODE_FEATURES = 12  # e.g., 6 piece types x 2 colors
NODE_EMBEDDING_DIM = 64
NUM_EDGE_FEATURES = 8  # e.g., 8 different types of relationships
EDGE_EMBEDDING_DIM = 32
GAT_HIDDEN_CHANNELS = 128
GAT_HEADS = 4
POLICY_OUTPUT_SIZE = 4672 # Max possible moves in chess, a common upper bound

class RCNModel(nn.Module):
    """
    Relational Chess Net (RCN) - A Graph Attention Network for chess.
    This model processes a graph representation of a chess board and outputs
    evaluations for value, policy, and tactical/strategic flags.
    """
    def __init__(self):
        super(RCNModel, self).__init__()

        # --- 1. Embedding Layers ---
        self.node_embedding = nn.Embedding(NUM_NODE_FEATURES, NODE_EMBEDDING_DIM)
        self.edge_embedding = nn.Embedding(NUM_EDGE_FEATURES, EDGE_EMBEDDING_DIM)

        # --- 2. Graph Encoder (GATv2) ---
        self.gat1 = GATv2Conv(
            NODE_EMBEDDING_DIM,
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
            concat=False # Use averaging for the final layer
        )
        self.norm2 = nn.LayerNorm(GAT_HIDDEN_CHANNELS)

        # --- 3. Multi-Task Output Heads ---
        # Shared MLP input dimension
        mlp_input_dim = GAT_HIDDEN_CHANNELS

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, POLICY_OUTPUT_SIZE)
        )

        # Tactic Head
        self.tactic_head = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Strategic Head
        self.strategic_head = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, data: Data):
        """
        Forward pass for the RCNModel.

        Args:
            data (torch_geometric.data.Data): A graph data object with attributes:
                - x: Node features [num_nodes, feature_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge attributes [num_edges, edge_feature_dim]
                - batch: Batch vector [num_nodes]

        Returns:
            dict: A dictionary containing the outputs of the four heads.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 1. Apply Embeddings
        # Ensure input tensors are of the correct type (long for embeddings)
        x = self.node_embedding(x.long()).squeeze(1) # Squeeze to remove extra dimension
        edge_attr = self.edge_embedding(edge_attr.long()).squeeze(1)

        # 2. Pass through Graph Encoder
        x = self.gat1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.relu(x)

        # 3. Global Pooling
        # Aggregate node features to get a single graph-level representation
        graph_embedding = global_mean_pool(x, batch)

        # 4. Pass through Output Heads
        value_out = torch.tanh(self.value_head(graph_embedding))
        policy_out = self.policy_head(graph_embedding) # No softmax here
        tactic_out = torch.sigmoid(self.tactic_head(graph_embedding))
        strategic_out = torch.sigmoid(self.strategic_head(graph_embedding))

        return {
            'value': value_out.squeeze(-1),
            'policy': policy_out,
            'tactic': tactic_out.squeeze(-1),
            'strategic': strategic_out.squeeze(-1)
        }

if __name__ == '__main__':
    # Example of how to create and run the model (for testing purposes)
    print("RCNModel class defined successfully.")

    # Create a dummy graph to test the forward pass
    # 32 nodes (pieces), max 8 edges per node
    num_nodes = 32
    num_edges = 128

    dummy_x = torch.randint(0, NUM_NODE_FEATURES, (num_nodes, 1))
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges))
    dummy_edge_attr = torch.randint(0, NUM_EDGE_FEATURES, (num_edges, 1))
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long)

    dummy_data = Data(
        x=dummy_x,
        edge_index=dummy_edge_index,
        edge_attr=dummy_edge_attr,
        batch=dummy_batch
    )

    model = RCNModel()
    print("\nModel architecture:")
    print(model)

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
