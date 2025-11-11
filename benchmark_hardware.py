"""
Hardware Benchmark Script - Testet GPU/CPU Performance und Auslastung
Führe dieses Skript VOR dem Training aus um Hardware-Probleme zu identifizieren!
"""
import torch
import time
import psutil
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.model import RCNModel
from scripts.graph_utils import fen_to_graph_data, TOTAL_NODE_FEATURES, NUM_EDGE_FEATURES
from torch_geometric.data import Batch
import config

def print_separator(title: str = ""):
    """Druckt eine formatierte Trennlinie."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"{'='*60}")

def check_cuda_availability() -> Dict[str, Any]:
    """Überprüft CUDA/GPU Verfügbarkeit und Details."""
    print_separator("GPU INFORMATION")

    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if info["cuda_available"]:
        print(f"✓ CUDA is available")
        print(f"  CUDA Version: {info['cuda_version']}")
        print(f"  Number of GPUs: {info['device_count']}")

        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")

            # Aktuelle GPU Auslastung
            if torch.cuda.is_available():
                print(f"    Current Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                print(f"    Current Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("❌ CUDA is NOT available - Training will use CPU only!")
        print("   This will be EXTREMELY slow for deep learning!")

    return info

def check_cpu_info():
    """Zeigt CPU Informationen."""
    print_separator("CPU INFORMATION")

    print(f"  CPU Count (Physical): {psutil.cpu_count(logical=False)}")
    print(f"  CPU Count (Logical): {psutil.cpu_count(logical=True)}")
    print(f"  CPU Usage: {psutil.cpu_percent(interval=1)}%")

    memory = psutil.virtual_memory()
    print(f"  RAM Total: {memory.total / 1024**3:.2f} GB")
    print(f"  RAM Available: {memory.available / 1024**3:.2f} GB")
    print(f"  RAM Usage: {memory.percent}%")

def benchmark_data_transfer(device: torch.device, num_samples: int = 100):
    """Benchmarkt den Datentransfer von CPU zu GPU."""
    print_separator("DATA TRANSFER BENCHMARK")

    if device.type == 'cpu':
        print("⚠ Skipping (CPU mode)")
        return

    # Test mit realistischen Graph-Daten
    print(f"Testing transfer of {num_samples} chess positions to GPU...")

    fen_samples = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
    ]

    graphs = [fen_to_graph_data(fen_samples[i % len(fen_samples)]) for i in range(num_samples)]

    # Test 1: Einzeltransfer
    start = time.time()
    for graph in graphs:
        _ = graph.to(device)
    single_time = time.time() - start

    print(f"  Individual Transfer: {single_time:.3f}s ({single_time/num_samples*1000:.2f}ms per sample)")

    # Test 2: Batch Transfer
    batch = Batch.from_data_list(graphs)
    start = time.time()
    batch_gpu = batch.to(device)
    batch_time = time.time() - start

    print(f"  Batch Transfer: {batch_time:.3f}s for {num_samples} samples")
    print(f"  → Speedup: {single_time/batch_time:.2f}x faster with batching")

    if batch_time > 1.0:
        print(f"  ⚠ WARNING: Data transfer is slow! Consider:")
        print(f"     - Using pin_memory=True in DataLoader")
        print(f"     - Increasing batch size")
        print(f"     - Moving data files to faster storage")

def benchmark_model_forward(device: torch.device, batch_sizes: list = [4, 8, 16, 32]):
    """Benchmarkt die Model Forward-Pass Geschwindigkeit."""
    print_separator("MODEL FORWARD PASS BENCHMARK")

    model = RCNModel(
        in_channels=TOTAL_NODE_FEATURES,
        out_channels=config.MODEL_OUT_CHANNELS,
        num_edge_features=NUM_EDGE_FEATURES
    ).to(device)
    model.eval()

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    print(f"Device: {device}")
    print(f"\nBatch Size | Time/Batch | Samples/sec | GPU Util")
    print(f"{'-'*60}")

    for bs in batch_sizes:
        graphs = [fen_to_graph_data(fen) for _ in range(bs)]
        batch = Batch.from_data_list(graphs).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(batch)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Actual benchmark
        num_iterations = 50
        start = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(batch)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        time_per_batch = elapsed / num_iterations
        samples_per_sec = (bs * num_iterations) / elapsed

        # GPU Memory
        gpu_mem = ""
        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            gpu_mem = f"{mem_allocated:.2f}GB"

        print(f"  {bs:4d}     | {time_per_batch*1000:8.2f}ms | {samples_per_sec:10.1f} | {gpu_mem}")

    if device.type == 'cuda':
        print(f"\n  Peak GPU Memory: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")

        # Reset für echtes Training
        torch.cuda.reset_peak_memory_stats(device)

def benchmark_training_step(device: torch.device, batch_size: int = 8, num_steps: int = 20):
    """Benchmarkt einen kompletten Training-Schritt (Forward + Backward + Optimizer)."""
    print_separator("FULL TRAINING STEP BENCHMARK")

    model = RCNModel(
        in_channels=TOTAL_NODE_FEATURES,
        out_channels=config.MODEL_OUT_CHANNELS,
        num_edge_features=NUM_EDGE_FEATURES
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    graphs = [fen_to_graph_data(fen) for _ in range(batch_size)]
    batch = Batch.from_data_list(graphs).to(device)

    # Add dummy targets
    batch.y = torch.zeros(batch_size, 1).to(device)
    batch.policy_target_from = torch.randint(0, 64, (batch_size,)).to(device)
    batch.policy_target_to = torch.randint(0, 64, (batch_size,)).to(device)
    batch.policy_target_promo = torch.full((batch_size,), -1, dtype=torch.long).to(device)
    batch.tactic_flag = torch.zeros(batch_size, 1).to(device)
    batch.strategic_flag = torch.zeros(batch_size, 1).to(device)

    print(f"Running {num_steps} training steps with batch_size={batch_size}...")

    model.train()
    times = []

    for step in range(num_steps):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()

        optimizer.zero_grad()
        value, policy_logits, tactic, strategic = model(batch)
        loss = loss_fn(value, batch.y)
        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)

        if step == 0:
            print(f"  First step (includes JIT compilation): {elapsed*1000:.2f}ms")

    avg_time = sum(times[1:]) / len(times[1:])  # Exclude first step
    throughput = batch_size / avg_time

    print(f"\n  Average time per step: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    print(f"  Estimated time for 1000 batches: {avg_time*1000/60:.1f} minutes")

    if device.type == 'cuda':
        print(f"  GPU Memory Used: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

        if throughput < 10:
            print(f"\n  ⚠ WARNING: Very low throughput detected!")
            print(f"     - Consider increasing batch size")
            print(f"     - Check if GPU is actually being used")
            print(f"     - Profile with nvidia-smi to see GPU utilization")

def suggest_optimizations(device: torch.device):
    """Gibt Optimierungsvorschläge basierend auf Hardware."""
    print_separator("OPTIMIZATION RECOMMENDATIONS")

    if device.type == 'cpu':
        print("❌ CRITICAL: You are using CPU for training!")
        print("\nImmediate Actions:")
        print("  1. Enable GPU in Colab: Runtime → Change runtime type → GPU")
        print("  2. Verify CUDA installation if running locally")
        print("  3. CPU training will be 50-100x slower than GPU!")
        return

    # GPU Optimierungen
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / 1024**3

    print("✓ GPU is available. Optimization suggestions:\n")

    print("1. BATCH SIZE:")
    if total_gb < 6:
        print(f"   Your GPU has {total_gb:.1f}GB VRAM → Recommend batch_size=4-8")
    elif total_gb < 12:
        print(f"   Your GPU has {total_gb:.1f}GB VRAM → Recommend batch_size=8-16")
    else:
        print(f"   Your GPU has {total_gb:.1f}GB VRAM → Recommend batch_size=16-32")

    print(f"   Current config.BATCH_SIZE = {config.BATCH_SIZE}")

    print("\n2. DATALOADER SETTINGS:")
    print("   Add to DataLoader initialization in train.py:")
    print("   ```python")
    print("   train_loader = DataLoader(")
    print("       train_dataset,")
    print("       batch_size=batch_size,")
    print("       shuffle=True,")
    print("       num_workers=2,        # ← ADD THIS (use 2-4 workers)")
    print("       pin_memory=True,       # ← ADD THIS (faster GPU transfer)")
    print("       persistent_workers=True # ← ADD THIS (reuse workers)")
    print("   )")
    print("   ```")

    print("\n3. MIXED PRECISION TRAINING:")
    print("   Use torch.cuda.amp for ~2x speedup:")
    print("   ```python")
    print("   from torch.cuda.amp import autocast, GradScaler")
    print("   scaler = GradScaler()")
    print("   ")
    print("   # In training loop:")
    print("   with autocast():")
    print("       value, policy, tactic, strategic = model(batch)")
    print("       loss = calculate_loss(...)")
    print("   scaler.scale(loss).backward()")
    print("   scaler.step(optimizer)")
    print("   scaler.update()")
    print("   ```")

    print("\n4. GRADIENT ACCUMULATION (if OOM errors):")
    print("   Simulate larger batches:")
    print("   ```python")
    print("   accumulation_steps = 4")
    print("   for i, batch in enumerate(train_loader):")
    print("       loss = loss / accumulation_steps")
    print("       loss.backward()")
    print("       if (i + 1) % accumulation_steps == 0:")
    print("           optimizer.step()")
    print("           optimizer.zero_grad()")
    print("   ```")

def monitor_gpu_realtime():
    """Zeigt wie man GPU-Auslastung in Echtzeit überwacht."""
    print_separator("REAL-TIME GPU MONITORING")

    if not torch.cuda.is_available():
        print("⚠ GPU not available, skipping")
        return

    print("To monitor GPU usage during training, run this in a SEPARATE terminal:\n")
    print("  watch -n 1 nvidia-smi")
    print("\nOR in Python (add to train.py):")
    print("```python")
    print("# Add at top of training loop")
    print("if epoch % 5 == 0 and torch.cuda.is_available():")
    print("    print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / '")
    print("          f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB')")
    print("```")

    print("\nKey metrics to watch in nvidia-smi:")
    print("  - GPU-Util: Should be >70% during training")
    print("  - Memory-Usage: Should be 50-90% of total")
    print("  - Power: Should be close to max TDP")
    print("\nIf GPU-Util is <30%, you have a CPU bottleneck!")

def main():
    """Hauptfunktion für Hardware-Benchmark."""
    print("\n" + "="*60)
    print("  RCN CHESS ENGINE - HARDWARE BENCHMARK")
    print("  Run this BEFORE training to diagnose performance issues")
    print("="*60)

    # 1. System Check
    cuda_info = check_cuda_availability()
    check_cpu_info()

    # 2. Bestimme Device
    device = torch.device('cuda' if cuda_info["cuda_available"] else 'cpu')

    # 3. Benchmarks
    if cuda_info["cuda_available"]:
        benchmark_data_transfer(device)

    benchmark_model_forward(device)
    benchmark_training_step(device)

    # 4. Monitoring Info
    monitor_gpu_realtime()

    # 5. Recommendations
    suggest_optimizations(device)

    print_separator()
    print("✓ Benchmark complete!")
    print("\nNext steps:")
    print("  1. Review the recommendations above")
    print("  2. Update train.py with suggested optimizations")
    print("  3. Run: python train.py")
    print("  4. Monitor GPU with: watch -n 1 nvidia-smi")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
