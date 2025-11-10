# --- Search Configuration ---
SEARCH_DEPTH = 4
QUIESCENCE_SEARCH_DEPTH = 4
TRANSPOSITION_TABLE_SIZE = 1_000_000 # Max entries

# --- Model Configuration ---
NODE_EMBEDDING_DIM = 64
EDGE_EMBEDDING_DIM = 16
GAT_HIDDEN_CHANNELS = 128
MODEL_OUT_CHANNELS = 128
POLICY_HEAD_INIT_SCALE = 0.01  # Small initialization for policy heads
GAT_HEADS = 4
DROPOUT_RATE = 0.2

# --- Training Configuration ---
LEARNING_RATE = 0.0005
BATCH_SIZE = 8
NUM_EPOCHS = 20
TRAIN_TEST_SPLIT = 0.85
GRADIENT_CLIP_NORM = 1.0

import os

# --- File Paths (Dynamische Persistenz) ---

def _is_in_colab():
    """Prüft, ob das Skript in Google Colab ausgeführt wird."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

IS_COLAB = _is_in_colab()

if IS_COLAB:
    # Wir sind in Google Colab, verwenden den Google Drive Pfad
    DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/RCN_Project"
else:
    # Wir sind lokal oder in Codespaces, verwenden einen relativen Pfad
    # "." bedeutet das aktuelle Verzeichnis, in dem das Skript läuft.
    DRIVE_PROJECT_ROOT = "."

# Der Rest der Pfade (MODEL_SAVE_PATH, DATA_DIR etc.)
# wird von DRIVE_PROJECT_ROOT abgeleitet und ist automatisch korrekt.
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
