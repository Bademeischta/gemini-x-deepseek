
import torch
import os
from model import RCNModel

def create_dummy_model():
    """
    Creates and saves a dummy RCN model with random weights.

    This is useful for development and testing when a fully trained model is not
    yet available, allowing other parts of the system (like the engine) to be
    built and tested independently.
    """
    print("Creating a dummy RCN model with random initial weights...")

    # Ensure the models directory exists
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # Instantiate the model from the model definition
    # The model parameters (e.g., in_channels) should match the real model
    # For this dummy version, we can use default or placeholder values.
    dummy_model = RCNModel(
        in_channels=6,  # Standard feature count: piece type, color, coords etc.
        hidden_channels=128,
        num_heads=4,
        num_layers=6
    )

    # The model is initialized with random weights by default in PyTorch

    model_path = os.path.join(model_dir, "rcn_model.pth")
    try:
        torch.save(dummy_model.state_dict(), model_path)
        print(f"Successfully saved dummy model to: {model_path}")
    except Exception as e:
        print(f"Error saving the dummy model: {e}")

if __name__ == '__main__':
    # This allows the script to be run directly for verification
    create_dummy_model()
