import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import os
import json

# This script assumes that 'scripts/model.py' is already present in the environment
from scripts.model import RCNModel
from scripts.dataset import ChessGraphDataset

# --- Configuration ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
TRAIN_TEST_SPLIT = 0.9
MODEL_SAVE_PATH = "models/rcn_model.pth"
DATA_PUZZLES_PATH = "data/kkk_subset_puzzles.jsonl"
DATA_STRATEGIC_PATH = "data/kkk_subset_strategic.jsonl"


def train():
    """Main function to run the training and validation loops."""
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Data Setup ---
    print("Setting up data loaders...")

    # Check for data files and create dummy ones if they don't exist
    if not os.path.exists(DATA_PUZZLES_PATH) or not os.path.exists(DATA_STRATEGIC_PATH):
        print("One or more data files are missing. Creating dummy data for a test run.")
        os.makedirs("data", exist_ok=True)
        dummy_puzzle_data = {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4", "tactic_flag": 1.0}
        dummy_strategic_data = {"fen": "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11", "value": -0.5, "policy_target": "a7a5", "strategic_flag": 1.0}
        with open(DATA_PUZZLES_PATH, 'w') as f:
            f.write(json.dumps(dummy_puzzle_data) + '\n')
        with open(DATA_STRATEGIC_PATH, 'w') as f:
            f.write(json.dumps(dummy_strategic_data) + '\n')

    # The dataset must be used within a 'with' block to ensure file handles are closed.
    with ChessGraphDataset(jsonl_paths=[DATA_PUZZLES_PATH, DATA_STRATEGIC_PATH]) as dataset:

        train_size = int(TRAIN_TEST_SPLIT * len(dataset))
        val_size = len(dataset) - train_size

        # Note: random_split returns Subset objects, not a new Dataset.
        # We'll use our wrapper to handle this gracefully.
        train_subset, val_subset = random_split(dataset, [train_size, val_size])

        # Import the wrapper from the dataset script
        from scripts.dataset import DatasetWrapper
        train_dataset = DatasetWrapper(dataset, train_subset.indices)
        val_dataset = DatasetWrapper(dataset, val_subset.indices)

        print(f"Dataset size: {len(dataset)}")
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # --- 2. Model Setup ---
        # Model setup and training loop must be inside the 'with' block
        model = RCNModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        loss_value_fn = nn.MSELoss()
        loss_policy_fn = nn.CrossEntropyLoss()
        loss_tactic_fn = nn.BCELoss()
        loss_strategic_fn = nn.BCELoss()

        best_val_loss = float('inf')

        # --- 3. Training Loop ---
        print("\nStarting training...")
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_losses = {'total': 0, 'value': 0, 'policy': 0, 'tactic': 0, 'strategic': 0}

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)

                loss_v = loss_value_fn(out['value'], batch.y)
                loss_p = loss_policy_fn(out['policy'], batch.policy_target)
                loss_t = loss_tactic_fn(out['tactic'], batch.tactic_flag)
                loss_s = loss_strategic_fn(out['strategic'], batch.strategic_flag)

                loss = loss_v + loss_p + loss_t + loss_s
                loss.backward()
                optimizer.step()

                train_losses['total'] += loss.item()
                train_losses['value'] += loss_v.item()
                train_losses['policy'] += loss_p.item()
                train_losses['tactic'] += loss_t.item()
                train_losses['strategic'] += loss_s.item()

            model.eval()
            val_losses = {'total': 0, 'value': 0, 'policy': 0, 'tactic': 0, 'strategic': 0}
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)

                    loss_v = loss_value_fn(out['value'], batch.y)
                    loss_p = loss_policy_fn(out['policy'], batch.policy_target)
                    loss_t = loss_tactic_fn(out['tactic'], batch.tactic_flag)
                    loss_s = loss_strategic_fn(out['strategic'], batch.strategic_flag)

                    loss = loss_v + loss_p + loss_t + loss_s
                    val_losses['total'] += loss.item()
                    val_losses['value'] += loss_v.item()
                    val_losses['policy'] += loss_p.item()
                    val_losses['tactic'] += loss_t.item()
                    val_losses['strategic'] += loss_s.item()

            # Averaging losses
            for k in train_losses: train_losses[k] /= max(1, len(train_loader))
            for k in val_losses: val_losses[k] /= max(1, len(val_loader))

            print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
            print(f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f}")

            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"New best model saved to {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})")

        print("\nTraining finished.")

if __name__ == '__main__':
    train()
