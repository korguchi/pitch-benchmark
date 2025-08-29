import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import List, Type

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from algorithms import get_algorithm, list_algorithms

# Get available algorithms once
AVAILABLE_ALGORITHMS = list_algorithms()


def generate_harmonic_signal(sample_rate: int, duration: float) -> np.ndarray:
    """
    Generate a test signal with fundamental frequency and harmonics.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.zeros_like(t)

    for harmonic in range(1, 4):
        amplitude = 1.0 / harmonic
        signal += amplitude * np.sin(2 * np.pi * 440 * harmonic * t)

    signal = signal.astype(np.float32)
    signal = signal / np.abs(signal).max()

    return signal


def benchmark_algorithm(
    algorithm_class: Type,
    audio_signal: np.ndarray,
    sample_rate: int,
    hop_length: int,
    device: str,
    n_runs: int = 20,
) -> float:
    """
    Benchmark a single pitch detection algorithm.
    """
    supports_device = "device" in algorithm_class.__init__.__code__.co_varnames

    if supports_device:
        algorithm = algorithm_class(
            sample_rate=sample_rate,
            hop_size=hop_length,
            fmin=50.0,
            fmax=1000.0,
            device=device,
        )
    else:
        if device == "cuda":
            return float("inf")
        algorithm = algorithm_class(
            sample_rate=sample_rate,
            hop_size=hop_length,
            fmin=50.0,
            fmax=1000.0,
        )

    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_runs):
        start_time = time.time()
        algorithm.extract_pitch(audio_signal)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        latencies.append(elapsed_time)

    return sum(latencies) / len(latencies)


def run_benchmark(
    algorithm_classes: List[Type],
    baseline_algorithm: Type,
    output_dir: str = "results",
    sample_rate: int = 22050,
    hop_length: int = 256,
    signal_length_sec: float = 5.0,
    n_runs: int = 20,
) -> None:
    """
    Run benchmarks using relative performance measurements.
    """
    print(f"Generating test signal ({signal_length_sec}s at {sample_rate}Hz)...")
    audio_signal = generate_harmonic_signal(sample_rate, signal_length_sec)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name()}")

    print(f"\nBenchmarking baseline: {baseline_algorithm.get_name()}")
    baseline_times = {}
    for device in devices:
        print(f"  Running {baseline_algorithm.get_name()} on {device.upper()}...")
        baseline_times[device] = benchmark_algorithm(
            baseline_algorithm,
            audio_signal,
            sample_rate,
            hop_length,
            device,
            n_runs,
        )

    print("\nBenchmarking all algorithms...")
    results = []
    algorithm_results = {}  # Store results for JSON output

    for algorithm_class in tqdm(algorithm_classes, desc="Testing algorithms"):
        # Update the progress bar description to show current algorithm
        tqdm.write(f"Testing {algorithm_class.get_name()}...")

        row = [algorithm_class.get_name()]
        algo_data = {"device_results": {}, "supports_cuda": False}

        for device in devices:
            latency = benchmark_algorithm(
                algorithm_class,
                audio_signal,
                sample_rate,
                hop_length,
                device,
                n_runs,
            )

            if latency == float("inf"):
                row.append("CPU only")
                algo_data["device_results"][device] = {
                    "supported": False,
                    "absolute_time_ms": None,
                    "relative_speed": None,
                    "baseline_time_ms": baseline_times[device] * 1000,
                }
            else:
                if device == "cuda":
                    algo_data["supports_cuda"] = True

                if algorithm_class == baseline_algorithm:
                    relative_speed = 1.0  # Force baseline to exactly 1.0x
                else:
                    relative_speed = baseline_times[device] / latency
                abs_time = latency * 1000  # Convert to ms
                row.append(f"{relative_speed:.2f}x ({abs_time:.1f}ms)")

                algo_data["device_results"][device] = {
                    "supported": True,
                    "absolute_time_ms": float(abs_time),
                    "relative_speed": float(relative_speed),
                    "baseline_time_ms": float(baseline_times[device] * 1000),
                }

        results.append(row)
        algorithm_results[algorithm_class.get_name()] = algo_data

    # Print console table as before
    headers = ["Algorithm"] + [f"{dev.upper()}" for dev in devices]
    print("\nBenchmark Results")
    print(f"Baseline: {baseline_algorithm.get_name()}")
    print(tabulate(results, headers=headers, tablefmt="grid"))

    # Save JSON results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    for algo_name, algo_data in algorithm_results.items():
        # Create parameter string for filename consistency
        param_str = (
            f"sr{int(sample_rate / 1000)}k_"
            f"hop{hop_length}_"
            f"len{signal_length_sec}s_"
            f"runs{n_runs}"
        )

        result_path = os.path.join(output_dir, f"speed_{algo_name}_{param_str}.json")

        full_result = {
            "metadata": {
                "benchmark_type": "speed",
                "algorithm_name": algo_name,
                "baseline_algorithm": baseline_algorithm.get_name(),
                "timestamp_utc": timestamp,
                "devices_tested": devices,
                "cuda_available": torch.cuda.is_available(),
            },
            "parameters": {
                "sample_rate": sample_rate,
                "hop_length": hop_length,
                "signal_length_seconds": signal_length_sec,
                "n_runs": n_runs,
                "signal_type": "harmonic",
                "fundamental_frequency": 440,
                "harmonics": [1, 2, 3],
            },
            "results": {
                "supports_cuda": algo_data["supports_cuda"],
                "device_performance": algo_data["device_results"],
            },
        }

        with open(result_path, "w") as f:
            json.dump(full_result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark pitch detection algorithms for speed performance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=AVAILABLE_ALGORITHMS,
        choices=AVAILABLE_ALGORITHMS,
        help="List of pitch detection algorithms to benchmark",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="CREPE",
        choices=AVAILABLE_ALGORITHMS,
        help="Baseline algorithm for relative speed comparison",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=22050, help="Audio sample rate in Hz"
    )
    parser.add_argument(
        "--hop-length", type=int, default=256, help="Hop length in samples"
    )
    parser.add_argument(
        "--signal-length",
        type=float,
        default=1.0,
        help="Length of test signal in seconds",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of benchmark runs per algorithm",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save JSON results",
    )

    args = parser.parse_args()

    # Map algorithm names to classes
    algorithms = [get_algorithm(name) for name in args.algorithms]
    baseline_algorithm = get_algorithm(args.baseline)

    print(f"Benchmarking {len(algorithms)} algorithms:")
    for algo in algorithms:
        print(f"  - {algo.get_name()}")

    run_benchmark(
        algorithm_classes=algorithms,
        baseline_algorithm=baseline_algorithm,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        signal_length_sec=args.signal_length,
        n_runs=args.n_runs,
    )
