# --- Search Configuration ---
SEARCH_DEPTH = 4
QUIESCENCE_SEARCH_DEPTH = 4
TRANSPOSITION_TABLE_SIZE = 1_000_000 # Max entries

# --- Model Configuration ---
NODE_EMBEDDING_DIM = 64
EDGE_EMBEDDING_DIM = 16
GAT_HIDDEN_CHANNELS = 128
GAT_HEADS = 4
DROPOUT_RATE = 0.3

# --- Training Configuration ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
TRAIN_TEST_SPLIT = 0.9
GRADIENT_CLIP_NORM = 1.0

# --- File Paths ---
MODEL_SAVE_PATH = "models/rcn_model.pth"
DATA_PUZZLES_PATH = "data/kkk_subset_puzzles.jsonl"
DATA_STRATEGIC_PATH = "data/kkk_subset_strategic.jsonl"
ENGINE_LOG_PATH = "engine.log"

# --- Logging Configuration ---
LOG_MAX_BYTES = 10 * 1024 * 1024 # 10 MB
LOG_BACKUP_COUNT = 5
