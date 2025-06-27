import numpy as np
import argparse
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import torch
from algorithms import get_algorithm, list_algorithms
from datasets import get_dataset, list_datasets, CHiMeNoiseDataset


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
        algo_name = algo_class.get_name()
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

                    _, pred_voicing = algo.extract_pitch(audio, threshold)
                    metrics = evaluate_voicing_detection(pred_voicing, true_voicing)
                    f1_scores.append(metrics["f1"])

                except Exception as e:
                    print(
                        f"Error processing {algo_class.get_name()} with threshold {threshold} on sample {idx}: {e}"
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
    gross_error_threshold: float = 200.0,
) -> Dict:
    """
    Evaluate pitch accuracy between predicted and ground truth pitch values.

    Args:
        pitch_pred: Predicted pitch values in Hz (T,)
        pitch_true: Ground truth pitch values in Hz (T,)
        valid_mask: Boolean array indicating which time steps contain valid pitch values for evaluation
        epsilon: Tolerance for RPA/RCA calculation in cents
        gross_error_threshold: Threshold for gross error calculation in cents

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
            "octave_error_rate": np.nan,
            "gross_error_rate": np.nan,
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

    # Gross error rate (errors > gross_error_threshold cents)
    gross_error_rate = np.mean(cents_diff > gross_error_threshold)

    # Octave error detection
    relative_error = np.abs(pred - true) / (true + np.finfo(float).eps)
    octave_errors = np.logical_or(
        relative_error > 0.4,  # More than 40% error
        (cents_diff > 1100) & (cents_diff < 1300),  # Near octave (1200 cents)
    )
    octave_error_rate = np.mean(octave_errors)

    return {
        "rmse": rmse,
        "cents_error": np.mean(cents_diff),
        "rpa": rpa,
        "rca": rca,
        "octave_error_rate": octave_error_rate,
        "gross_error_rate": gross_error_rate,
        "valid_frames": np.sum(valid_mask),
    }


def evaluate_pitch_algorithms(
    dataset: Dataset,
    fmin: int,
    fmax: int,
    algorithms: List[Tuple],
) -> Dict:
    """
    Evaluate multiple pitch detection algorithms with comprehensive metrics.

    Args:
        dataset: Dataset instance providing audio and ground truth
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz
        algorithms: List of tuples (algorithm_class, threshold)

    Returns:
        Dictionary containing comprehensive evaluation metrics for each algorithm
    """
    results = {}
    for algo_class, threshold in tqdm(algorithms, desc="Evaluating algorithms"):
        algo_name = algo_class.get_name()
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
                audio = sample["audio"].numpy()
                true_pitch = sample["pitch"].numpy()
                true_voicing = sample["periodicity"].numpy()

                # Skip any files without pitch
                # This happens when the pitch is not in [fmin, fmax]
                # For example: NSynth
                if not true_voicing.any():
                    continue

                pred_pitch, pred_voicing = algo.extract_pitch(audio, threshold)

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

            # Calculate enhanced harmonic mean (6-component)
            rpa = pitch_metrics["rpa"]
            cents_error = pitch_metrics["cents_error"]
            octave_error_rate = pitch_metrics["octave_error_rate"]
            gross_error_rate = pitch_metrics["gross_error_rate"]
            voicing_recall = voicing_metrics["recall"]
            voicing_precision = voicing_metrics["precision"]

            # Transform error rates to accuracy scores using exponential decay
            cents_accuracy = np.exp(-cents_error / 500.0)
            octave_accuracy = np.exp(-octave_error_rate * 10.0)
            gross_error_accuracy = np.exp(-gross_error_rate * 5.0)

            # Collect all six components for harmonic mean calculation
            components = [
                rpa,  # Raw pitch accuracy (frames with <50¢ error)
                cents_accuracy,  # Fine-grained pitch precision
                voicing_recall,  # Voiced segment detection completeness
                voicing_precision,  # Voiced segment detection accuracy
                octave_accuracy,  # Octave error robustness
                gross_error_accuracy,  # Overall tracking stability
            ]

            # Compute 6-component harmonic mean with safety check
            if any(component <= 0 for component in components):
                combined_score = 0.0
            else:
                combined_score = 6.0 / sum(1.0 / component for component in components)

            results[algo_name] = {
                "voicing_detection": voicing_metrics,
                "pitch_accuracy": pitch_metrics,
                "combined_score": combined_score,
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
                    "octave_error_rate": np.nan,
                    "gross_error_rate": np.nan,
                },
                "combined_score": np.nan,
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
            "Pitch Robustness",
            [
                (
                    "Octave Err % ↓",
                    lambda m: format_metric(
                        m["pitch_accuracy"]["octave_error_rate"], percentage=True
                    ),
                ),
                (
                    "Gross Err % ↓",
                    lambda m: format_metric(
                        m["pitch_accuracy"]["gross_error_rate"], percentage=True
                    ),
                ),
            ],
        ),
        (
            "Combined Score",
            [
                (
                    "Harmonic Mean ↑",
                    lambda m: format_metric(m["combined_score"], percentage=True),
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
        choices=list_datasets(),
        help="Dataset to evaluate on",
    )
    required.add_argument(
        "--data-dir", type=str, required=True, help="Path to dataset directory"
    )
    required.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=list_algorithms(),
        choices=list_algorithms(),
        help="List of pitch detection algorithms to evaluate",
    )

    # Optional arguments
    parser.add_argument(
        "--noise-dir",
        type=str,
        help="Path to CHiME-Home dir containing background audio files",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=10.0,
        help="Signal-to-Noise Ratio in dB for noise addition",
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

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize base dataset
    dataset_class = get_dataset(args.dataset)
    base_dataset = dataset_class(
        root_dir=args.data_dir,
        sample_rate=args.sample_rate,
        hop_size=args.hop_size,
    )

    # Wrap with noise augmentation if requested
    if args.noise_dir:
        dataset = CHiMeNoiseDataset(
            base_dataset=base_dataset,
            chime_home_dir=args.noise_dir,
            chime_sample_rate=16000,
            snr_db=args.snr,
            noise_probability=1.0,
        )
    else:
        dataset = base_dataset

    # Map algorithm names to classes
    algorithms = [get_algorithm(name) for name in args.algorithms]

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
            optimal_thresholds[algo_class.get_name()],
        )
        for algo_class in algorithms
    ]

    # Run evaluation
    noise_desc = ""
    if args.noise_dir:
        noise_desc = f" with {args.noise_dir} noise (SNR: {args.snr}dB)"
    else:
        noise_desc = " without noise"

    print(f"\nTesting{noise_desc}")
    metrics = evaluate_pitch_algorithms(
        fmin=args.fmin,
        fmax=args.fmax,
        dataset=dataset,
        algorithms=optimized_algorithms,
    )
    print_evaluation_results(metrics)
