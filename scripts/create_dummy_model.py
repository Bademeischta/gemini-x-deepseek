
import torch
import os
from scripts.model import RCNModel

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

    # Instantiate the model from the model definition.
    # The constructor takes no arguments.
    dummy_model = RCNModel()

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
