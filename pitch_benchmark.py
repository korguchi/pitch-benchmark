import numpy as np
import argparse
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import torch
from algorithms import (
    YAAPTPitchAlgorithm,
    PraatPitchAlgorithm,
    TorchCREPEPitchAlgorithm,
    SWIPEPitchAlgorithm,
    RAPTPitchAlgorithm,
    pYINPitchAlgorithm,
    PENNPitchAlgorithm,
)
from datasets import PitchDatasetPTDB, PitchDatasetNSynth, PitchDatasetMDBStemSynth
from noise import ESC50Noise, WhiteNoise


def optimize_thresholds(
    dataset: Dataset,
    validation_size: float,
    fmin: int,
    fmax: int,
    algorithms: List,
    threshold_range: List[float] = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
) -> Dict:
    """
    Find optimal thresholds for each algorithm using a validation set.

    Args:
        dataset: Dataset instance providing audio and ground truth
        validation_size: Fraction of dataset to use for validation (e.g., 0.2)
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz
        algorithms: List of algorithm_class
        threshold_range: List of threshold values to try

    Returns:
        Dictionary mapping algorithm names to their optimal thresholds
    """
    # Create validation split
    dataset_size = len(dataset)
    validation_length = max(int(dataset_size * validation_size), 1)
    print(f"Using {validation_length} samples for optimizing thresholds")
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    validation_indices = indices[:validation_length]
    validation_dataset = Subset(dataset, validation_indices)

    optimal_thresholds = {}

    for algo_class in tqdm(algorithms, desc="Optimizing thresholds"):
        algo_name = algo_class.__name__.replace("PitchAlgorithm", "")
        best_f1 = -1
        best_threshold = None

        # Initialize algorithm
        algo = algo_class(
            sample_rate=dataset.sample_rate,
            hop_size=dataset.hop_size,
            fmin=fmin,
            fmax=fmax,
        )

        # Try different thresholds
        for threshold in threshold_range:
            f1_scores = []

            for idx in range(len(validation_dataset)):
                try:
                    sample = validation_dataset[idx]
                    audio = sample["audio"].numpy()
                    true_voicing = sample["periodicity"].numpy()

                    _, pred_voicing = algo(audio, threshold)
                    metrics = evaluate_voicing_detection(pred_voicing, true_voicing)
                    f1_scores.append(metrics["f1"])

                except Exception as e:
                    print(
                        f"Error processing {algo_class.__name__} with threshold {threshold} on sample {idx}: {e}"
                    )
                    continue

            if not f1_scores or np.isnan(f1_scores).any():
                continue

            mean_f1 = np.mean(f1_scores)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_threshold = threshold

        optimal_thresholds[algo_name] = best_threshold

    return optimal_thresholds


def evaluate_voicing_detection(
    pred_voiced: np.ndarray, true_voiced: np.ndarray
) -> Dict:
    """
    Evaluate voicing detection using precision-recall metrics.
    Handles edge cases where algorithm detects no voiced segments.

    Args:
        pred_voiced: Predicted voicing values (T,)
        true_voiced: Ground truth voicing values (T,)

    Returns:
        Dictionary containing precision, recall, and F1 metrics.
    """
    true_pos = np.sum(pred_voiced & true_voiced)
    false_pos = np.sum(pred_voiced & ~true_voiced)
    false_neg = np.sum(~pred_voiced & true_voiced)

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0

    # Calculate F1 score
    f1 = (
        0.0
        if (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_pitch_accuracy(
    pitch_pred: np.ndarray,
    pitch_true: np.ndarray,
    valid_mask: np.ndarray,
    epsilon: float = 50.0,
) -> Dict:
    """
    Evaluate pitch accuracy between predicted and ground truth pitch values.

    Args:
        pitch_pred: Predicted pitch values in Hz (T,)
        pitch_true: Ground truth pitch values in Hz (T,)
        valid_mask: Boolean array indicating which time steps contain valid pitch values for evaluation
        epsilon: Tolerance for RPA/RCA calculation in cents

    Returns:
        Dictionary containing pitch accuracy metrics
    """
    if len(pitch_pred) != len(pitch_true):
        raise ValueError(
            f"Length mismatch: pred={len(pitch_pred)}, true={len(pitch_true)}"
        )

    if not np.any(valid_mask):
        return {
            "rmse": np.nan,
            "cents_error": np.nan,
            "rpa": np.nan,
            "rca": np.nan,
            "valid_frames": 0,
        }

    # Extract valid frequencies
    pred = pitch_pred[valid_mask]
    true = pitch_true[valid_mask]

    # Calculate RMSE in Hz
    rmse = np.sqrt(np.mean((pred - true) ** 2))

    # Calculate cents difference
    cents_diff = np.abs(1200 * np.log2(pred / (true + np.finfo(float).eps)))

    # Raw Pitch Accuracy (RPA)
    rpa = np.mean(cents_diff < epsilon)

    # Raw Chroma Accuracy (RCA)
    wrapped_cents_diff = cents_diff % 1200
    chroma_diff = np.minimum(wrapped_cents_diff, 1200 - wrapped_cents_diff)
    rca = np.mean(chroma_diff < epsilon)

    return {
        "rmse": rmse,
        "cents_error": np.mean(cents_diff),
        "rpa": rpa,
        "rca": rca,
        "valid_frames": np.sum(valid_mask),
    }


def evaluate_pitch_algorithms(
    dataset: Dataset, fmin: int, fmax: int, algorithms: List[Tuple], noise_class
) -> Dict:
    """
    Evaluate multiple pitch detection algorithms with comprehensive metrics.

    Args:
        dataset: Dataset instance providing audio and ground truth
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz
        algorithms: List of tuples (algorithm_class, threshold)
        noise_class: Noise class

    Returns:
        Dictionary containing comprehensive evaluation metrics for each algorithm
    """
    results = {}
    for algo_class, threshold in tqdm(algorithms, desc="Evaluating algorithms"):
        algo_name = algo_class.__name__.replace("PitchAlgorithm", "")
        algo = algo_class(
            sample_rate=dataset.sample_rate,
            hop_size=dataset.hop_size,
            fmin=fmin,
            fmax=fmax,
        )

        # Global accumulators for all predictions and ground truth
        all_pred_voicing = []
        all_true_voicing = []
        all_pred_pitch = []
        all_true_pitch = []
        all_voiced_mask = []  # Combined mask for voiced regions

        for idx in tqdm(
            range(len(dataset)), desc=f"Processing {algo_name}", leave=False
        ):
            try:
                sample = dataset[idx]
                audio = sample["audio"]
                # Apply noise if specified
                if noise_class is not None:
                    audio = noise_class.mix_noise(audio).squeeze(0)
                audio = audio.numpy()
                true_pitch = sample["pitch"].numpy()
                true_voicing = sample["periodicity"].numpy()

                # Skip any files without pitch
                # This happens when the pitch is not in [fmin, fmax]
                # For example: NSynth
                if not true_voicing.any():
                    continue

                pred_pitch, pred_voicing = algo(audio, threshold)

                # Accumulate all predictions and ground truth
                all_pred_voicing.append(pred_voicing)
                all_true_voicing.append(true_voicing)
                all_pred_pitch.append(pred_pitch)
                all_true_pitch.append(true_pitch)
                all_voiced_mask.append(pred_voicing & true_voicing)

            except Exception as e:
                print(f"Error processing file {idx} with {algo_name}: {str(e)}")
                continue

        # Concatenate all arrays for global evaluation
        if all_pred_voicing:  # Check if we have any valid data
            global_pred_voicing = np.concatenate(all_pred_voicing)
            global_true_voicing = np.concatenate(all_true_voicing)
            global_pred_pitch = np.concatenate(all_pred_pitch)
            global_true_pitch = np.concatenate(all_true_pitch)
            global_voiced_mask = np.concatenate(all_voiced_mask)

            voicing_metrics = evaluate_voicing_detection(
                global_pred_voicing, global_true_voicing
            )
            pitch_metrics = evaluate_pitch_accuracy(
                global_pred_pitch, global_true_pitch, global_voiced_mask
            )

            results[algo_name] = {
                "voicing_detection": voicing_metrics,
                "pitch_accuracy": pitch_metrics,
                "num_files_processed": len(all_pred_voicing),
                "total_frames": len(global_pred_voicing),
            }
        else:
            # Handle case where no valid files were processed
            results[algo_name] = {
                "voicing_detection": {
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                },
                "pitch_accuracy": {
                    "rmse": np.nan,
                    "cents_error": np.nan,
                    "rpa": np.nan,
                    "rca": np.nan,
                },
                "num_files_processed": 0,
                "total_frames": 0,
            }

    return results


def print_evaluation_results(metrics: Dict):
    print("\nPitch Detection Evaluation Results")
    print("=" * 120)

    def format_metric(value: float, percentage: bool = False) -> str:
        if np.isnan(value):
            return "N/A".center(16)
        if percentage:
            return f"{value * 100:.1f}%".center(16)
        return f"{value:.2f}".center(16)

    sections = [
        (
            "Voicing Detection Performance",
            [
                (
                    "Precision ↑",
                    lambda m: format_metric(
                        m["voicing_detection"]["precision"], percentage=True
                    ),
                ),
                (
                    "Recall ↑",
                    lambda m: format_metric(
                        m["voicing_detection"]["recall"], percentage=True
                    ),
                ),
                (
                    "F1 ↑",
                    lambda m: format_metric(
                        m["voicing_detection"]["f1"], percentage=True
                    ),
                ),
            ],
        ),
        (
            "Pitch Accuracy Performance",
            [
                ("RMSE (Hz) ↓", lambda m: format_metric(m["pitch_accuracy"]["rmse"])),
                (
                    "Cents Err (Δ¢) ↓",
                    lambda m: format_metric(m["pitch_accuracy"]["cents_error"]),
                ),
                (
                    "RPA ↑",
                    lambda m: format_metric(
                        m["pitch_accuracy"]["rpa"], percentage=True
                    ),
                ),
                (
                    "RCA ↑",
                    lambda m: format_metric(
                        m["pitch_accuracy"]["rca"], percentage=True
                    ),
                ),
            ],
        ),
        (
            "Combined Score",
            [
                (
                    "Harmonic Mean ↑",
                    lambda m: format_metric(
                        calculate_combined_score(m), percentage=True
                    ),
                ),
            ],
        ),
    ]

    for section_name, metrics_info in sections:
        n_metrics = len(metrics_info)
        # Calculate table width: 20 (alg) + 16 per metric + (n_metrics) spaces
        table_width = 20 + 16 * n_metrics + n_metrics

        print(f"\n{section_name}:")
        print("-" * table_width)

        # Prepare headers with truncation
        headers = ["Algorithm"] + [
            name[:16] for name, _ in metrics_info
        ]  # Truncate long headers

        # Create format string (all columns centered)
        format_str = "{:^20} " + " ".join(["{:^16}"] * n_metrics)
        print(format_str.format(*headers))
        print("-" * table_width)

        # Process each algorithm
        for algo_name, algo_metrics in metrics.items():
            algo_display = algo_name[:20]  # Truncate long algorithm names

            try:
                values = [func(algo_metrics) for _, func in metrics_info]
                print(format_str.format(algo_display, *values))
            except (KeyError, TypeError) as e:
                error_values = ["N/A".center(16)] * len(metrics_info)
                print(format_str.format(algo_display, *error_values))
                print(f"  Warning: Error formatting {algo_name}: {e}")


def calculate_combined_score(metrics: Dict) -> float:
    """
    Calculate a combined performance score using the harmonic mean of multiple pitch detection metrics.

    The harmonic mean is used instead of arithmetic mean because it requires balanced performance
    across ALL metrics - a system must perform well on both voicing detection and pitch accuracy
    to achieve a good score. Poor performance in any single metric heavily impacts the final score.

    The metrics included are:
    - Precision (voicing detection)
    - Recall (voicing detection)
    - RPA (Raw Pitch Accuracy)
    - RCA (Raw Chroma Accuracy)

    All metrics are already in [0,1] range where 1 is best.

    Args:
        metrics (Dict): A dictionary with the following structure:
            {
                "voicing_detection": {
                    "precision": float,
                    "recall": float
                },
                "pitch_accuracy": {
                    "rpa": float,
                    "rca": float
                }
            }

    Returns:
        float: Harmonic mean of the selected metrics, or np.nan if any metric is missing or invalid.
    """
    EPSILON = 1e-10  # Small constant to prevent division by zero

    try:
        # Extract the same metrics as the original function
        values = [
            metrics["voicing_detection"]["precision"],
            metrics["voicing_detection"]["recall"],
            metrics["pitch_accuracy"]["rpa"],
            metrics["pitch_accuracy"]["rca"],
        ]

        values = np.array(values)

        # Check for NaN values
        if np.any(np.isnan(values)):
            return np.nan

        # Ensure no zeros (same as original)
        values = np.clip(values, EPSILON, 1.0)

        # Calculate harmonic mean (same formula as original)
        harmonic_mean = len(values) / np.sum(1 / values)
        return harmonic_mean

    except (KeyError, TypeError, ZeroDivisionError):
        return np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pitch detection algorithms on a dataset with optional noise condition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["PTDB", "NSynth", "MDBStemSynth"],
        help="Dataset to evaluate on",
    )
    required.add_argument(
        "--data-dir", type=str, required=True, help="Path to dataset directory"
    )
    required.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["YAAPT", "Praat", "SWIPE", "RAPT", "pYIN", "CREPE", "PENN"],
        choices=["YAAPT", "Praat", "SWIPE", "RAPT", "pYIN", "CREPE", "PENN"],
        help="List of pitch detection algorithms to evaluate",
    )

    # Optional arguments
    parser.add_argument(
        "--noise-type",
        type=str,
        choices=["white", "esc50"],
        help="Type of noise to apply. If not specified, no noise is applied",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=15.0,
        help="Signal-to-Noise Ratio in dB for noise addition",
    )
    parser.add_argument(
        "--esc50-dir",
        type=str,
        help="Path to ESC-50 dataset (required if noise-type is esc50)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=22050, help="Audio sample rate in Hz"
    )
    parser.add_argument("--hop-size", type=int, default=256, help="Hop size in samples")
    parser.add_argument(
        "--fmin", type=float, default=65.0, help="Minimum frequency in Hz"
    )
    parser.add_argument(
        "--fmax", type=float, default=300.0, help="Maximum frequency in Hz"
    )
    parser.add_argument(
        "--seed", type=int, default=3, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.01,
        help="Fraction of dataset to use for threshold optimization",
    )

    args = parser.parse_args()

    # Validate ESC50 directory if ESC50 noise is selected
    if args.noise_type == "esc50" and not args.esc50_dir:
        parser.error("--esc50-dir is required when using esc50 noise")

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize dataset
    dataset_classes = {
        "PTDB": PitchDatasetPTDB,
        "NSynth": PitchDatasetNSynth,
        "MDBStemSynth": PitchDatasetMDBStemSynth,
    }
    dataset = dataset_classes[args.dataset](
        root_dir=args.data_dir,
        use_cache=False,
        sample_rate=args.sample_rate,
        hop_size=args.hop_size,
    )

    # Map algorithm names to classes
    algo_classes = {
        "YAAPT": YAAPTPitchAlgorithm,
        "Praat": PraatPitchAlgorithm,
        "SWIPE": SWIPEPitchAlgorithm,
        "RAPT": RAPTPitchAlgorithm,
        "pYIN": pYINPitchAlgorithm,
        "CREPE": TorchCREPEPitchAlgorithm,
        "PENN": PENNPitchAlgorithm,
    }
    algorithms = [algo_classes[name] for name in args.algorithms]

    # Find optimal thresholds
    optimal_thresholds = optimize_thresholds(
        dataset=dataset,
        validation_size=args.validation_size,
        fmin=args.fmin,
        fmax=args.fmax,
        algorithms=algorithms,
    )
    print(f"Thresholds: {optimal_thresholds}")

    # Update algorithms with optimal thresholds
    optimized_algorithms = [
        (
            algo_class,
            optimal_thresholds[algo_class.__name__.replace("PitchAlgorithm", "")],
        )
        for algo_class in algorithms
    ]

    # Set up noise configuration
    noise = None
    if args.noise_type == "white":
        noise = WhiteNoise(snr_range=[args.snr, args.snr])
    elif args.noise_type == "esc50":
        noise = ESC50Noise(
            data_dir=args.esc50_dir,
            snr_range=[args.snr, args.snr],
            target_sample_rate=args.sample_rate,
        )

    # Run evaluation
    print(
        f"\nTesting with{' ' + args.noise_type if args.noise_type else 'out'} noise"
        + (f" (SNR: {args.snr}dB)" if args.noise_type else "")
    )
    metrics = evaluate_pitch_algorithms(
        fmin=args.fmin,
        fmax=args.fmax,
        dataset=dataset,
        algorithms=optimized_algorithms,
        noise_class=noise,
    )
    print_evaluation_results(metrics)
