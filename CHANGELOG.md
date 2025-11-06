# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-11-06

This is the first stable release after a comprehensive overhaul of the RCN chess engine. The entire codebase was refactored to fix critical bugs, improve performance, and establish a solid foundation for future development.

### Added
- **New Search Algorithm:** Implemented a complete Negamax search with alpha-beta pruning, replacing the previous broken implementation.
- **Advanced Search Heuristics:** Added a Transposition Table, Killer Move heuristic, and MVV-LVA move ordering for captures.
- **Quiescence Search:** Added a depth-limited quiescence search to stabilize evaluations in tactical positions.
- **"From-To-Promotion" Policy Head:** Redesigned the model's policy output for greater efficiency and accuracy.
- **Expanded Graph Features:** The graph representation now includes Pin and X-Ray edges, as well as global game state information (turn, castling rights, en-passant, 50-move rule) in every node.
- **Batch Normalization:** Added `BatchNorm` layers to the model to improve training stability.
- **Gradient Clipping:** Implemented gradient clipping in the training loop to prevent exploding gradients.
- **Configuration File:** Centralized all major parameters into `config.py`.
- **Comprehensive Test Suite:** Created a robust test suite with unit and integration tests, achieving significant code coverage. This includes tests for the UCI handshake, data loading, and graph creation.
- **End-to-End Training Test:** Added a test to validate the entire training pipeline.
- **Dummy Model Creation Script:** Added a script (`scripts/create_dummy_model.py`) to generate a model with random weights for development.
- **Code Quality:** Added extensive type hints and Google-style docstrings to all major modules.
- **Move Generation Caching:** Implemented caching for sorted move lists to improve search performance.

### Fixed
- **Critical `uci_to_index` Crash:** The function no longer crashes on invalid moves.
- **PV Reconstruction:** The search now correctly reconstructs and returns the Principal Variation.
- **Dataset Memory Leak:** The `ChessGraphDataset` is now a proper context manager, preventing resource leaks.
- **UCI Race Condition:** The engine now waits for full initialization before sending `readyok`.
- **Time Management Precision:** Time calculations now use integer arithmetic (milliseconds) to avoid floating-point errors.
- **Model Initialization:** The `RCNModel` is now correctly initialized with all required parameters in `train.py` and other scripts.
- **Training Loss Calculation:** Correctly unpacked the model's tuple output and added the missing loss for the promotion head.
- **Log Rotation:** Implemented `RotatingFileHandler` to prevent the engine log from growing indefinitely.
- **Duplicate Position Prevention:** Added logic to prevent duplicate positions from being added to the dataset.
- **Tensor Shape Mismatches:** Corrected several tensor shape errors in the training loop and model forward pass.
