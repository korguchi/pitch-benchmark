import torch
import time
import numpy as np
from typing import List, Type
from tabulate import tabulate
from algorithms import (
    YAAPTPitchAlgorithm,
    PraatPitchAlgorithm,
    TorchCREPEPitchAlgorithm,
    PENNPitchAlgorithm,
    SWIPEPitchAlgorithm,
    RAPTPitchAlgorithm,
    pYINPitchAlgorithm,
)


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
        algorithm(audio_signal)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        latencies.append(elapsed_time)

    return sum(latencies) / len(latencies)


def run_benchmark(
    algorithm_classes: List[Type],
    baseline_algorithm: Type,
    sample_rate: int = 22050,
    hop_length: int = 256,
    signal_length_sec: float = 5.0,
    n_runs: int = 20,
) -> None:
    """
    Run benchmarks using relative performance measurements.
    """
    audio_signal = generate_harmonic_signal(sample_rate, signal_length_sec)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    baseline_times = {}
    for device in devices:
        baseline_times[device] = benchmark_algorithm(
            baseline_algorithm,
            audio_signal,
            sample_rate,
            hop_length,
            device,
            n_runs,
        )

    results = []
    for algorithm_class in algorithm_classes:
        row = [algorithm_class.__name__.replace("PitchAlgorithm", "")]

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
            else:
                if algorithm_class == baseline_algorithm:
                    relative_speed = 1.0  # Force baseline to exactly 1.0x
                else:
                    relative_speed = baseline_times[device] / latency
                abs_time = latency * 1000  # Convert to ms
                row.append(f"{relative_speed:.2f}x ({abs_time:.1f}ms)")

        results.append(row)

    headers = ["Algorithm"] + [f"{dev.upper()}" for dev in devices]
    print("\nBenchmark Results")
    print(f"Baseline: {baseline_algorithm.__name__.replace('PitchAlgorithm', '')}")
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    algorithms = [
        YAAPTPitchAlgorithm,
        PraatPitchAlgorithm,
        TorchCREPEPitchAlgorithm,
        PENNPitchAlgorithm,
        SWIPEPitchAlgorithm,
        RAPTPitchAlgorithm,
        pYINPitchAlgorithm,
    ]

    run_benchmark(
        algorithm_classes=algorithms,
        baseline_algorithm=TorchCREPEPitchAlgorithm,
        sample_rate=22050,
        hop_length=256,
        signal_length_sec=5.0,
        n_runs=20,
    )
