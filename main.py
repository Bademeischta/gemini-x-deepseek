
import argparse
import subprocess
import sys

def run_script(script_path, as_module=False):
    """Executes a Python script and handles errors."""
    try:
        command = [sys.executable]
        if as_module:
            # Replace file path separators with dots for module path
            module_path = script_path.replace('.py', '').replace('/', '.')
            command.extend(["-m", module_path])
        else:
            command.append(script_path)

        print(f"Executing command: {' '.join(command)}")
        subprocess.run(command, check=True)
        print(f"Successfully executed: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_path}: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        return False

def main():
    """Main entry point for the RCN project."""
    parser = argparse.ArgumentParser(
        description="RCN Chess Project: A unified command-line interface."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Command: process-data
    parser_process = subparsers.add_parser(
        "process-data", help="Run the data processing pipeline."
    )
    parser_process.set_defaults(func=process_data)

    # Command: create-dummy-model
    parser_dummy = subparsers.add_parser(
        "create-dummy-model", help="Create a dummy model for development."
    )
    parser_dummy.set_defaults(func=create_dummy_model)

    # Command: train
    parser_train = subparsers.add_parser(
        "train", help="Train the RCN model."
    )
    parser_train.set_defaults(func=train_model)

    # Command: run-engine
    parser_engine = subparsers.add_parser(
        "run-engine", help="Run the UCI chess engine."
    )
    parser_engine.set_defaults(func=run_engine)

    args = parser.parse_args()
    args.func()

def process_data():
    """Runs the full data processing pipeline."""
    print("--- Starting Data Processing Pipeline ---")
    if run_script("scripts/process_puzzles.py", as_module=True):
        run_script("scripts/process_elite_games.py", as_module=True)
    print("--- Data Processing Finished ---")

def create_dummy_model():
    """Creates a dummy model."""
    print("--- Creating Dummy Model ---")
    run_script("scripts/create_dummy_model.py", as_module=True)
    print("--- Dummy Model Creation Finished ---")

def train_model():
    """Trains the model."""
    print("--- Starting Model Training ---")
    # train.py is in the root directory
    run_script("train.py", as_module=False)
    print("--- Model Training Finished ---")

def run_engine():
    """Runs the chess engine."""
    print("--- Starting UCI Engine ---")
    # engine.py is in the root, not a module in /scripts
    run_script("engine.py", as_module=False)
    print("--- Engine Terminated ---")

if __name__ == "__main__":
    main()
