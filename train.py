"""
This module contains the main training loop for the RCNModel.

It handles data loading, model setup, training, validation, and saving the
best model checkpoint.
"""
import torch
import torch.nn as nn
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
import os
import json
from typing import List

from scripts.model import RCNModel
from scripts.dataset import ChessGraphDataset, DatasetWrapper
from scripts.graph_utils import TOTAL_NODE_FEATURES, NUM_EDGE_FEATURES
import config

def save_checkpoint(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, best_val_loss: float) -> None:
    """Saves the training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, config.TRAINING_CHECKPOINT_PATH)
    print(f"Checkpoint for epoch {epoch+1} saved to {config.TRAINING_CHECKPOINT_PATH}")

def train() -> None:
    """
    Main training function with resume capability.

    This function orchestrates the entire training process:
    1. Sets up the device (GPU or CPU).
    2. Initializes and prepares the datasets and data loaders.
    3. Initializes the model, optimizer, and loss functions.
    4. Loads a checkpoint if one exists to resume training.
    5. Runs the training and validation loop for a configured number of epochs.
    6. Saves a checkpoint at the end of each epoch.
    7. Saves the model with the best validation loss separately.
    8. Handles interruptions gracefully by saving a final checkpoint.
    """
    # --- 0. Setup ---
    # Create directories based on the actual file paths, which might be patched in tests.
    if config.MODEL_SAVE_PATH:
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    if config.TRAINING_CHECKPOINT_PATH:
        os.makedirs(os.path.dirname(config.TRAINING_CHECKPOINT_PATH), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Data Setup ---
    print("Setting up data loaders...")
    jsonl_paths: List[str] = [p for p in [config.DATA_PUZZLES_PATH, config.DATA_STRATEGIC_PATH] if p]
    if not all(os.path.exists(p) for p in jsonl_paths):
        # Note: This dummy data logic might not be ideal with Drive persistence,
        # but we keep it to prevent crashes if files are missing.
        print("One or more data files not found. Creating dummy files.")
        dummy_data = [
            {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4", "tactic_flag": 1.0},
            {"fen": "r1b2rk1/pp1p1p1p/1qn2np1/4p3/4P3/1N1B1N2/PPPQ1PPP/R3K2R b KQ - 1 11", "value": -0.5, "policy_target": "a7a5", "strategic_flag": 1.0}
        ]
        # We need at least 16 samples for a batch size of 8, so create 8 of each.
        if config.DATA_PUZZLES_PATH and not os.path.exists(config.DATA_PUZZLES_PATH):
            with open(config.DATA_PUZZLES_PATH, 'w') as f:
                for _ in range(8):
                    f.write(json.dumps(dummy_data[0]) + '\n')
        if config.DATA_STRATEGIC_PATH and not os.path.exists(config.DATA_STRATEGIC_PATH):
            with open(config.DATA_STRATEGIC_PATH, 'w') as f:
                for _ in range(8):
                    f.write(json.dumps(dummy_data[1]) + '\n')

    with ChessGraphDataset(jsonl_paths=jsonl_paths) as dataset:
        dataset_len = len(dataset)

        # Enforce minimum batch size for BatchNorm
        MIN_BATCH_SIZE = 4
        if config.BATCH_SIZE < MIN_BATCH_SIZE:
            print(f"WARNING: batch_size {config.BATCH_SIZE} too small for BatchNorm. Setting to {MIN_BATCH_SIZE}")
            batch_size = MIN_BATCH_SIZE
        else:
            batch_size = config.BATCH_SIZE

        train_size = int(config.TRAIN_TEST_SPLIT * dataset_len)
        val_size = dataset_len - train_size

        # Need at least batch_size samples in each set for BatchNorm
        if val_size < batch_size and config.TRAIN_TEST_SPLIT < 1.0:
            val_size = min(batch_size, dataset_len // 5)
            train_size = dataset_len - val_size

        if train_size < batch_size:
            raise ValueError(f"Dataset too small ({dataset_len} samples). Need at least {batch_size * 2} samples for training.")

        print(f"Split: {train_size} train, {val_size} val (batch_size={batch_size})")

        if train_size == 0 or (val_size == 0 and config.TRAIN_TEST_SPLIT < 1.0):
            print("Not enough data for a meaningful train/val split. Using all data for training.")
            train_indices, val_indices = list(range(dataset_len)), []
        else:
            train_subset, val_subset = random_split(dataset, [train_size, val_size])
            train_indices, val_indices = train_subset.indices, val_subset.indices

        train_dataset = DatasetWrapper(dataset, train_indices)
        val_dataset = DatasetWrapper(dataset, val_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True) if val_dataset else None

        # --- 2. Model Setup ---
        model = RCNModel(
            in_channels=TOTAL_NODE_FEATURES,
            out_channels=config.MODEL_OUT_CHANNELS,
            num_edge_features=NUM_EDGE_FEATURES
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
        loss_value_fn, loss_policy_fn = nn.MSELoss(), nn.CrossEntropyLoss(ignore_index=-1)
        loss_tactic_fn, loss_strategic_fn = nn.BCELoss(), nn.BCELoss()

        # --- 3. Load Checkpoint ---
        start_epoch = 0
        best_val_loss = float('inf')
        if os.path.exists(config.TRAINING_CHECKPOINT_PATH):
            checkpoint = torch.load(config.TRAINING_CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Training wird ab Epoche {start_epoch + 1} fortgesetzt.")

        # --- 4. Training Loop ---
        print("\nStarting training...")
        epoch = start_epoch
        try:
            for epoch in range(start_epoch, config.NUM_EPOCHS):
                model.train()
                total_train_loss: float = 0.0
                batch_count = 0
                for batch in train_loader:
                    # Skip batches with invalid data
                    if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                        print(f"WARNING: Skipping batch with NaN/Inf in input")
                        continue

                    batch = batch.to(device)
                    optimizer.zero_grad()
                    try:
                        value, (policy_from, policy_to, policy_promo), tactic, strategic = model(batch)

                        # Validate outputs before loss calculation
                        if torch.isnan(value).any():
                            raise ValueError("NaN in value output")

                        loss_v = loss_value_fn(value, batch.y.view(-1, 1))
                        # Policy loss with NaN protection
                        loss_p_from = loss_policy_fn(policy_from, batch.policy_target_from)
                        loss_p_to = loss_policy_fn(policy_to, batch.policy_target_to)
                        loss_p_promo = loss_policy_fn(policy_promo, batch.policy_target_promo)

                        # Check each component
                        if torch.isnan(loss_p_from):
                            print(f"WARNING: NaN in policy_from loss")
                            continue
                        if torch.isnan(loss_p_to):
                            print(f"WARNING: NaN in policy_to loss")
                            continue
                        if torch.isnan(loss_p_promo):
                            print(f"WARNING: NaN in policy_promo loss")
                            continue

                        loss_p = loss_p_from + loss_p_to + loss_p_promo

                        loss_t = loss_tactic_fn(tactic, batch.tactic_flag.view(-1, 1))
                        loss_s = loss_strategic_fn(strategic, batch.strategic_flag.view(-1, 1))
                        loss = loss_v + loss_p + loss_t + loss_s

                        # Check for NaN loss before backward
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"WARNING: NaN/Inf loss detected (v:{loss_v:.4f}, p:{loss_p:.4f}, t:{loss_t:.4f}, s:{loss_s:.4f})")
                            continue

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
                        optimizer.step()
                        total_train_loss += loss.item()
                        batch_count += 1
                    except Exception as e:
                        print(f"ERROR in training batch: {e}")
                        continue
                avg_train_loss = total_train_loss / max(1, batch_count)

                total_val_loss: float = 0.0
                val_batch_count = 0
                if val_loader and len(val_loader) > 0:
                    model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            # Skip batches with invalid data
                            if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                                print(f"WARNING: Skipping validation batch with NaN/Inf in input")
                                continue

                            batch = batch.to(device)
                            try:
                                value, (policy_from, policy_to, policy_promo), tactic, strategic = model(batch)

                                # Validate outputs before loss calculation
                                if torch.isnan(value).any():
                                    raise ValueError("NaN in validation value output")

                                loss_v = loss_value_fn(value, batch.y.view(-1, 1))
                                # Policy loss with NaN protection
                                loss_p_from = loss_policy_fn(policy_from, batch.policy_target_from)
                                loss_p_to = loss_policy_fn(policy_to, batch.policy_target_to)
                                loss_p_promo = loss_policy_fn(policy_promo, batch.policy_target_promo)

                                # Check each component
                                if torch.isnan(loss_p_from):
                                    print(f"WARNING: NaN in policy_from loss")
                                    continue
                                if torch.isnan(loss_p_to):
                                    print(f"WARNING: NaN in policy_to loss")
                                    continue
                                if torch.isnan(loss_p_promo):
                                    print(f"WARNING: NaN in policy_promo loss")
                                    continue

                                loss_p = loss_p_from + loss_p_to + loss_p_promo
                                loss_t = loss_tactic_fn(tactic, batch.tactic_flag.view(-1, 1))
                                loss_s = loss_strategic_fn(strategic, batch.strategic_flag.view(-1, 1))
                                loss = loss_v + loss_p + loss_t + loss_s

                                if torch.isnan(loss) or torch.isinf(loss):
                                    print(f"WARNING: NaN/Inf loss in validation")
                                    continue

                                total_val_loss += loss.item()
                                val_batch_count += 1
                            except Exception as e:
                                print(f"ERROR in validation batch: {e}")
                                continue
                avg_val_loss = total_val_loss / max(1, val_batch_count)
                print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                    print(f"New best model saved to {config.MODEL_SAVE_PATH}")

                # Save a checkpoint at the end of every epoch
                save_checkpoint(epoch, model, optimizer, best_val_loss)

        except KeyboardInterrupt:
            print("\nTraining unterbrochen. Speichere letzten Checkpoint...")
            save_checkpoint(epoch, model, optimizer, best_val_loss)
        except Exception as e:
            print(f"\nFehler aufgetreten: {e}. Speichere Notfall-Checkpoint...")
            save_checkpoint(epoch, model, optimizer, best_val_loss)

        print("\nTraining finished.")

if __name__ == '__main__':
    train()
