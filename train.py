"""
This module contains the main training loop for the RCNModel.

It handles data loading, model setup, training, validation, and saving the
best model checkpoint.
"""
import torch
import torch.nn as nn
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from torch.cuda.amp import autocast, GradScaler  # <-- NEU: Mixed Precision
import os
import json
from typing import List
from tqdm import tqdm  # <-- NEU: Progress Bars
import chess

from scripts.model import RCNModel
from scripts.dataset import ChessGraphDataset, DatasetWrapper
from scripts.uci_index import uci_to_index_4096
import config

def print_gpu_memory_stats(device: torch.device, prefix: str = ""):
    """Zeigt GPU Memory Stats für Debugging."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def save_checkpoint(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                   best_val_loss: float, scaler: GradScaler = None) -> None:
    """Saves the training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, config.TRAINING_CHECKPOINT_PATH)
    print(f"✓ Checkpoint for epoch {epoch+1} saved to {config.TRAINING_CHECKPOINT_PATH}")

def create_legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Creates a boolean mask for legal moves (4096 indices)."""
    mask = torch.zeros(4096, dtype=torch.bool)
    for move in board.legal_moves:
        idx = uci_to_index_4096(move.uci())
        mask[idx] = True
    return mask

def train() -> None:
    """
    Main training function with resume capability, GPU optimization, and progress bars.
    """
    # --- 0. Setup ---
    if config.MODEL_SAVE_PATH:
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    if config.TRAINING_CHECKPOINT_PATH:
        os.makedirs(os.path.dirname(config.TRAINING_CHECKPOINT_PATH), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Device: {device}")

    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}")
        print(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("⚠ WARNING: Training on CPU - This will be VERY slow!")
        print("   Consider enabling GPU in Colab or checking CUDA installation")

    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"{'='*60}\n")

    # --- 1. Data Setup ---
    print("Setting up data loaders...")
    jsonl_paths: List[str] = [p for p in [config.DATA_PUZZLES_PATH, config.DATA_STRATEGIC_PATH] if p]

    if not all(os.path.exists(p) for p in jsonl_paths):
        print("One or more data files not found. Creating dummy files.")
        dummy_data = [
            {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "policy_target": "e2e4", "tactic_flag": 0.0},
            {"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "value": 0.1, "policy_target": "g1f3", "tactic_flag": 0.0},
            {"fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 3", "value": 0.1, "policy_target": "f1c4", "tactic_flag": 0.0},
            {"fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 4", "value": 0.1, "policy_target": "f6e4", "tactic_flag": 1.0},
            {"fen": "rnbqk2r/pppp1ppp/5n2/4p3/1bB1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 2 5", "value": 0.2, "policy_target": "d2d3", "tactic_flag": 0.0},
            {"fen": "r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 6", "value": 0.1, "policy_target": "e8g8", "tactic_flag": 0.0},
            {"fen": "r1bq1rk1/ppppbppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 1 7", "value": 0.1, "policy_target": "a2a4", "tactic_flag": 0.0},
            {"fen": "r1bq1rk1/ppppbppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b - - 0 7", "value": 0.1, "policy_target": "d7d6", "tactic_flag": 0.0},
            {"fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "value": 0.0, "policy_target": "c2c3", "strategic_flag": 1.0},
            {"fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/2P5/PP1P1PPP/RNBQKBNR b KQkq - 0 2", "value": 0.0, "policy_target": "d7d5", "strategic_flag": 1.0},
            {"fen": "rnbqkbnr/pp2pppp/3p4/2p5/4P3/2P5/PP1P1PPP/RNBQKBNR w KQkq - 0 3", "value": 0.0, "policy_target": "d2d4", "strategic_flag": 1.0},
            {"fen": "rnbqkbnr/pp2pppp/3p4/8/3pP3/2P5/PP3PPP/RNBQKBNR w KQkq - 0 4", "value": 0.1, "policy_target": "c3d4", "strategic_flag": 1.0},
            {"fen": "rnbqkbnr/pp2pppp/3p4/8/3PP3/8/PP3PPP/RNBQKBNR b KQkq - 0 4", "value": 0.1, "policy_target": "g8f6", "strategic_flag": 1.0},
            {"fen": "rnbqkb1r/pp2pppp/3p1n2/8/3PP3/8/PP3PPP/RNBQKBNR w KQkq - 1 5", "value": 0.1, "policy_target": "b1c3", "strategic_flag": 1.0},
            {"fen": "rnbqkb1r/pp2pppp/3p1n2/8/3PP3/2N5/PP3PPP/R1BQKBNR b KQkq - 2 5", "value": 0.1, "policy_target": "a7a6", "strategic_flag": 1.0},
            {"fen": "rnbqkb1r/1p2pppp/p2p1n2/8/3PP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 6", "value": 0.1, "policy_target": "f2f4", "strategic_flag": 1.0},
        ]

        if config.DATA_PUZZLES_PATH and not os.path.exists(config.DATA_PUZZLES_PATH):
            with open(config.DATA_PUZZLES_PATH, 'w') as f:
                for item in dummy_data[:8]:
                    f.write(json.dumps(item) + '\n')
        if config.DATA_STRATEGIC_PATH and not os.path.exists(config.DATA_STRATEGIC_PATH):
            with open(config.DATA_STRATEGIC_PATH, 'w') as f:
                for item in dummy_data[8:]:
                    f.write(json.dumps(item) + '\n')

    dataset = ChessGraphDataset(jsonl_paths=jsonl_paths)
    dataset_len = len(dataset)
    print(f"Total dataset size: {dataset_len} samples")

    MIN_BATCH_SIZE = 4
    if config.BATCH_SIZE < MIN_BATCH_SIZE:
        print(f"WARNING: batch_size {config.BATCH_SIZE} too small for BatchNorm. Setting to {MIN_BATCH_SIZE}")
        batch_size = MIN_BATCH_SIZE
    else:
        batch_size = config.BATCH_SIZE

    train_size = int(config.TRAIN_TEST_SPLIT * dataset_len)
    val_size = dataset_len - train_size

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
    val_dataset = DatasetWrapper(dataset, val_indices) if val_indices else None

    # OPTIMIZATION: num_workers und pin_memory für schnelleren GPU Transfer
    num_workers = 2 if device.type == 'cuda' else 0
    pin_memory = device.type == 'cuda'

    print(f"DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    ) if val_dataset else None

    # --- 2. Model Setup ---
    print("\nInitializing model...")
    model = RCNModel(
        in_channels=15, # Corresponds to NEW_NODE_FEATURES in model
        out_channels=config.MODEL_OUT_CHANNELS,
        num_edge_features=2 # Corresponds to NEW_EDGE_FEATURES in model
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)

    # OPTIMIZATION: Mixed Precision Training für ~2x Speedup
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("✓ Using Mixed Precision Training (AMP)")

    loss_value_fn = nn.MSELoss()
    loss_policy_fn = nn.CrossEntropyLoss(ignore_index=-1)
    loss_tactic_fn = nn.BCEWithLogitsLoss()
    loss_strategic_fn = nn.BCEWithLogitsLoss()

    # --- 3. Load Checkpoint ---
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(config.TRAINING_CHECKPOINT_PATH):
        print(f"\nLoading checkpoint from {config.TRAINING_CHECKPOINT_PATH}")
        checkpoint = torch.load(config.TRAINING_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"✓ Resuming from epoch {start_epoch + 1}, best val loss: {best_val_loss:.4f}")

    if device.type == 'cuda':
        print_gpu_memory_stats(device, "\nInitial ")

    # --- 4. Training Loop mit Progress Bars ---
    print("\n" + "="*60)
    print("  STARTING TRAINING")
    print("="*60 + "\n")

    # PROGRESS BAR: Für gesamtes Training über alle Epochen
    epoch_pbar = tqdm(
        range(start_epoch, config.NUM_EPOCHS),
        desc="Training Progress",
        unit="epoch",
        position=0,
        leave=True,
        ncols=100
    )

    epoch = start_epoch
    try:
        for epoch in epoch_pbar:
            model.train()
            total_train_loss: float = 0.0
            batch_count = 0

            # PROGRESS BAR: Für Batches innerhalb einer Epoche
            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]",
                unit="batch",
                position=1,
                leave=False,
                ncols=100
            )

            for batch_idx, batch in enumerate(train_pbar):
                # Skip batches with invalid data
                if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                    train_pbar.set_postfix({"status": "SKIP:NaN"})
                    continue

                batch = batch.to(device)
                optimizer.zero_grad()

                try:
                    # OPTIMIZATION: Mixed Precision Forward Pass
                    with autocast(enabled=use_amp):
                        value, policy_logits, tactic, strategic = model(batch)

                        # Create legal move mask
                        legal_move_masks = torch.stack([create_legal_move_mask(chess.Board(data.fen)) for data in batch.to_data_list()]).to(device)

                        # Apply mask to logits
                        policy_logits[~legal_move_masks] = -1e9

                        # Validate outputs
                        if torch.isnan(value).any():
                            raise ValueError("NaN in value output")

                        loss_v = loss_value_fn(value, batch.y.view(-1, 1))

                        # New: Joint Policy Loss Calculation
                        policy_targets = batch.policy_target
                        loss_p = loss_policy_fn(policy_logits, policy_targets)

                        loss_t = loss_tactic_fn(tactic, batch.tactic_flag.view(-1, 1))
                        loss_s = loss_strategic_fn(strategic, batch.strategic_flag.view(-1, 1))
                        loss = loss_v + loss_p + loss_t + loss_s

                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        train_pbar.set_postfix({"loss": "NaN", "status": "SKIP"})
                        continue

                    # OPTIMIZATION: Mixed Precision Backward Pass
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
                        optimizer.step()

                    total_train_loss += loss.item()
                    batch_count += 1

                    # UPDATE PROGRESS BAR
                    postfix = {
                        "loss": f"{loss.item():.4f}",
                        "avg": f"{total_train_loss/batch_count:.4f}"
                    }

                    # GPU Memory jeden 10. Batch
                    if device.type == 'cuda' and batch_idx % 10 == 0:
                        gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
                        postfix["GPU"] = f"{gpu_mem:.1f}GB"

                    train_pbar.set_postfix(postfix)

                except Exception as e:
                    train_pbar.set_postfix({"error": str(e)[:20]})
                    print(f"\n  ERROR in batch {batch_idx}: {e}")
                    continue

            avg_train_loss = total_train_loss / max(1, batch_count)

            # Validation Loop mit Progress Bar
            total_val_loss: float = 0.0
            val_batch_count = 0

            if val_loader and len(val_loader) > 0:
                model.eval()

                val_pbar = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]",
                    unit="batch",
                    position=1,
                    leave=False,
                    ncols=100
                )

                with torch.no_grad():
                    for batch in val_pbar:
                        if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                            val_pbar.set_postfix({"status": "SKIP:NaN"})
                            continue

                        batch = batch.to(device)

                        try:
                            with autocast(enabled=use_amp):
                                value, policy_logits, tactic, strategic = model(batch)

                                # Create legal move mask
                                legal_move_masks = torch.stack([create_legal_move_mask(chess.Board(data.fen)) for data in batch.to_data_list()]).to(device)

                                # Apply mask to logits
                                policy_logits[~legal_move_masks] = -1e9

                                if torch.isnan(value).any():
                                    raise ValueError("NaN in validation value")

                                loss_v = loss_value_fn(value, batch.y.view(-1, 1))

                                policy_targets = batch.policy_target
                                loss_p = loss_policy_fn(policy_logits, policy_targets)

                                loss_t = loss_tactic_fn(tactic, batch.tactic_flag.view(-1, 1))
                                loss_s = loss_strategic_fn(strategic, batch.strategic_flag.view(-1, 1))
                                loss = loss_v + loss_p + loss_t + loss_s

                            if torch.isnan(loss) or torch.isinf(loss):
                                val_pbar.set_postfix({"status": "NaN"})
                                continue

                            total_val_loss += loss.item()
                            val_batch_count += 1

                            val_pbar.set_postfix({
                                "loss": f"{loss.item():.4f}",
                                "avg": f"{total_val_loss/val_batch_count:.4f}"
                            })

                        except Exception as e:
                            val_pbar.set_postfix({"error": str(e)[:20]})
                            continue

            avg_val_loss = total_val_loss / max(1, val_batch_count) if val_batch_count > 0 else float('inf')

            # UPDATE EPOCH PROGRESS BAR
            epoch_pbar.set_postfix({
                "train": f"{avg_train_loss:.4f}",
                "val": f"{avg_val_loss:.4f}" if avg_val_loss != float('inf') else "N/A"
            })

            # Console Output (bleibt sichtbar)
            print(f"\n{'─'*60}")
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} Complete")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.4f}" if avg_val_loss != float('inf') else "  Val Loss:   N/A")

            if device.type == 'cuda':
                print_gpu_memory_stats(device, "  ")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"  ✓ New best model saved!")

            # Save checkpoint
            save_checkpoint(epoch, model, optimizer, best_val_loss, scaler)
            print(f"{'─'*60}\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user!")
        save_checkpoint(epoch, model, optimizer, best_val_loss, scaler)
        print("✓ Checkpoint saved")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        save_checkpoint(epoch, model, optimizer, best_val_loss, scaler)
        print("✓ Emergency checkpoint saved")
    finally:
        epoch_pbar.close()

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print("="*60 + "\n")


if __name__ == '__main__':
    train()
