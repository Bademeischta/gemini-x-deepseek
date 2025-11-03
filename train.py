import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import os
import json

from scripts.model import RCNModel
from scripts.dataset import ChessGraphDataset, DatasetWrapper

import config

def train():
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Data Setup ---
    print("Setting up data loaders...")
    if not os.path.exists(config.DATA_PUZZLES_PATH) or not os.path.exists(config.DATA_STRATEGIC_PATH):
        # Create dummy data if not present
        os.makedirs("data", exist_ok=True)
        dummy_data = [
            {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4", "tactic_flag": 1.0},
            {"fen": "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11", "value": -0.5, "policy_target": "a7a5", "strategic_flag": 1.0}
        ]
        with open(config.DATA_PUZZLES_PATH, 'w') as f: f.write(json.dumps(dummy_data[0]) + '\n')
        with open(config.DATA_STRATEGIC_PATH, 'w') as f: f.write(json.dumps(dummy_data[1]) + '\n')

    with ChessGraphDataset(jsonl_paths=[config.DATA_PUZZLES_PATH, config.DATA_STRATEGIC_PATH]) as dataset:
        train_size = int(config.TRAIN_TEST_SPLIT * len(dataset))
        val_size = len(dataset) - train_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size])

        train_dataset = DatasetWrapper(dataset, train_subset.indices)
        val_dataset = DatasetWrapper(dataset, val_subset.indices)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # --- 2. Model Setup ---
        model = RCNModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        loss_value_fn = nn.MSELoss()
        loss_policy_fn = nn.CrossEntropyLoss(ignore_index=-1)
        loss_tactic_fn = nn.BCELoss()
        loss_strategic_fn = nn.BCELoss()
        best_val_loss = float('inf')

        # --- 3. Training Loop ---
        print("\nStarting training...")
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)

                loss_v = loss_value_fn(out['value'], batch.y)
                loss_p = loss_policy_fn(out['policy_from'], batch.policy_target_from) + \
                         loss_policy_fn(out['policy_to'], batch.policy_target_to)
                loss_t = loss_tactic_fn(out['tactic'], batch.tactic_flag)
                loss_s = loss_strategic_fn(out['strategic'], batch.strategic_flag)
                loss = loss_v + loss_p + loss_t + loss_s

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
                optimizer.step()
                total_train_loss += loss.item()

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    loss_v = loss_value_fn(out['value'], batch.y)
                    loss_p = loss_policy_fn(out['policy_from'], batch.policy_target_from) + \
                             loss_policy_fn(out['policy_to'], batch.policy_target_to)
                    loss_t = loss_tactic_fn(out['tactic'], batch.tactic_flag)
                    loss_s = loss_strategic_fn(out['strategic'], batch.strategic_flag)
                    loss = loss_v + loss_p + loss_t + loss_s
                    total_val_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_loader))
            avg_val_loss = total_val_loss / max(1, len(val_loader))

            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"New best model saved to {config.MODEL_SAVE_PATH}")

        print("\nTraining finished.")

if __name__ == '__main__':
    train()
