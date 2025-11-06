import torch
import os
import sys

# Add project root to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.model import RCNModel
from scripts.graph_utils import TOTAL_NODE_FEATURES, NUM_EDGE_FEATURES
import config

def create_and_save_dummy_model():
    """
    Initializes an RCNModel with random weights and saves it to the path
    specified in the config file.
    """
    print("Initializing dummy RCNModel...")

    # Ensure the target directory exists
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    # Correctly initialize the model with required parameters
    dummy_model = RCNModel(
        in_channels=TOTAL_NODE_FEATURES,
        out_channels=config.MODEL_OUT_CHANNELS,
        num_edge_features=NUM_EDGE_FEATURES
    )

    # Save the initialized model
    torch.save(dummy_model.state_dict(), config.MODEL_SAVE_PATH)

    print(f"Dummy model saved successfully to: {config.MODEL_SAVE_PATH}")
    print("The engine can now be run using this randomly initialized model.")

if __name__ == '__main__':
    create_and_save_dummy_model()
