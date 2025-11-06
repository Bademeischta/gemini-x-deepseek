# --- Search Configuration ---
SEARCH_DEPTH = 4
QUIESCENCE_SEARCH_DEPTH = 4
TRANSPOSITION_TABLE_SIZE = 1_000_000 # Max entries

# --- Model Configuration ---
NODE_EMBEDDING_DIM = 64
EDGE_EMBEDDING_DIM = 16
GAT_HIDDEN_CHANNELS = 128
MODEL_OUT_CHANNELS = 128
GAT_HEADS = 4
DROPOUT_RATE = 0.3

# --- Training Configuration ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
TRAIN_TEST_SPLIT = 0.9
GRADIENT_CLIP_NORM = 1.0

import os

# --- File Paths (Google Drive Persistence) ---
# Base path for the project on Google Drive.
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/RCN_Project"

# Model and checkpoint paths
MODEL_DIR = os.path.join(DRIVE_PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "rcn_model.pth")
TRAINING_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "training_checkpoint.pth")

# Data paths
DATA_DIR = os.path.join(DRIVE_PROJECT_ROOT, "data")
DATA_PUZZLES_PATH = os.path.join(DATA_DIR, "kkk_subset_puzzles.jsonl")
DATA_STRATEGIC_PATH = os.path.join(DATA_DIR, "kkk_subset_strategic.jsonl")
PROCESSING_STATE_PATH = os.path.join(DATA_DIR, "processing_state.json")

# Log path
ENGINE_LOG_PATH = os.path.join(DRIVE_PROJECT_ROOT, "engine.log")

# Directory creation is handled by the scripts that need them (train.py, process_elite_games.py)
# to make the code more modular and testable.


# --- Logging Configuration ---
LOG_MAX_BYTES = 10 * 1024 * 1024 # 10 MB
LOG_BACKUP_COUNT = 5
