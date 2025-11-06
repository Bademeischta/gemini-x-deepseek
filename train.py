import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import os
import json

from scripts.model import RCNModel
from scripts.dataset import ChessGraphDataset, DatasetWrapper
from scripts.graph_utils import TOTAL_NODE_FEATURES, NUM_EDGE_FEATURES
import config

def train():
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Data Setup ---
    print("Setting up data loaders...")
    jsonl_paths = [p for p in [config.DATA_PUZZLES_PATH, config.DATA_STRATEGIC_PATH] if p]
    if not all(os.path.exists(p) for p in jsonl_paths):
        os.makedirs("data", exist_ok=True)
        dummy_data = [
            {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4", "tactic_flag": 1.0},
            {"fen": "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11", "value": -0.5, "policy_target": "a7a5", "strategic_flag": 1.0}
        ]
        if config.DATA_PUZZLES_PATH and not os.path.exists(config.DATA_PUZZLES_PATH):
            with open(config.DATA_PUZZLES_PATH, 'w') as f: f.write(json.dumps(dummy_data[0]) + '\n')
        if config.DATA_STRATEGIC_PATH and not os.path.exists(config.DATA_STRATEGIC_PATH):
            with open(config.DATA_STRATEGIC_PATH, 'w') as f: f.write(json.dumps(dummy_data[1]) + '\n')

    with ChessGraphDataset(jsonl_paths=jsonl_paths) as dataset:
        # Use a smaller subset for quick testing if the dataset is large
        dataset_len = len(dataset)
        train_size = int(config.TRAIN_TEST_SPLIT * dataset_len)
        val_size = dataset_len - train_size

        # Ensure we have at least one sample in validation
        if val_size == 0 and train_size > 0:
            train_size -= 1
            val_size += 1

        if train_size == 0 or val_size == 0:
            print("Not enough data to create train/val split. Using all data for training.")
            train_indices = list(range(dataset_len))
            val_indices = []
        else:
            train_subset, val_subset = random_split(dataset, [train_size, val_size])
            train_indices = train_subset.indices
            val_indices = val_subset.indices

        train_dataset = DatasetWrapper(dataset, train_indices)
        val_dataset = DatasetWrapper(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # --- 2. Model Setup ---
        model = RCNModel(
            in_channels=TOTAL_NODE_FEATURES,
            out_channels=config.MODEL_OUT_CHANNELS,
            num_edge_features=NUM_EDGE_FEATURES
        ).to(device)
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

                value, policy_logits, tactic, strategic = model(batch)
                policy_from, policy_to, policy_promo = policy_logits

                loss_v = loss_value_fn(value, batch.y.view(-1, 1))
                loss_p_from = loss_policy_fn(policy_from, batch.policy_target_from)
                loss_p_to = loss_policy_fn(policy_to, batch.policy_target_to)
                loss_p_promo = loss_policy_fn(policy_promo, batch.policy_target_promo)
                loss_p = loss_p_from + loss_p_to + loss_p_promo

                loss_t = loss_tactic_fn(tactic, batch.tactic_flag.view(-1, 1))
                loss_s = loss_strategic_fn(strategic, batch.strategic_flag.view(-1, 1))
                loss = loss_v + loss_p + loss_t + loss_s

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_loader))

            # Validation
            total_val_loss = 0
            if val_loader:
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        value, policy_logits, tactic, strategic = model(batch)
                        policy_from, policy_to, policy_promo = policy_logits

                        loss_v = loss_value_fn(value, batch.y.view(-1, 1))
                        loss_p_from = loss_policy_fn(policy_from, batch.policy_target_from)
                        loss_p_to = loss_policy_fn(policy_to, batch.policy_target_to)
                        loss_p_promo = loss_policy_fn(policy_promo, batch.policy_target_promo)
                        loss_p = loss_p_from + loss_p_to + loss_p_promo

                        loss_t = loss_tactic_fn(tactic, batch.tactic_flag.view(-1, 1))
                        loss_s = loss_strategic_fn(strategic, batch.strategic_flag.view(-1, 1))
                        loss = loss_v + loss_p + loss_t + loss_s
                        total_val_loss += loss.item()

            avg_val_loss = total_val_loss / max(1, len(val_loader))

            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"New best model saved to {config.MODEL_SAVE_PATH}")

        print("\nTraining finished.")

if __name__ == '__main__':
    train()
