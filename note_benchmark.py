import numpy as np
import argparse
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import torch
from algorithms import get_algorithm, list_algorithms
from datasets import (
    get_transcription_dataset,
    list_transcription_datasets,
    CHiMeNoiseDataset,
)
from mir_eval.transcription import evaluate as transcription_evaluate


def midi_to_hz(midi_pitch: float) -> float:
    """Converts MIDI pitch to Hz."""
    return 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))


def evaluate_transcription(
    true_notes: List[Dict], pred_notes: List[Dict]
) -> Dict[str, float]:
    """
    Evaluate note transcription performance using mir_eval standards.

    Args:
        true_notes: Ground truth notes with 'start', 'end', 'midi_pitch'
        pred_notes: Predicted notes with 'start', 'end', 'midi_pitch'

    Returns:
        Dictionary with precision, recall, and F1 scores (Onset-Offset-Pitch).
        Note: mir_eval's default tolerances are 0.05s for onset/offset
        and 0.5 semitones for pitch.
    """
    # Convert list of dicts to mir_eval's expected numpy array format
    # mir_eval expects (onsets, offsets) in a (N, 2) array for intervals
    # and pitches in Hz in a (N,) array.

    # Handle empty true/predicted notes lists to prevent errors in np.array creation
    true_intervals = (
        np.array([[n["start"], n["end"]] for n in true_notes])
        if true_notes
        else np.empty((0, 2))
    )
    # Convert MIDI pitch to Hz for mir_eval
    true_pitches_hz = (
        np.array([midi_to_hz(n["midi_pitch"]) for n in true_notes])
        if true_notes
        else np.array([])
    )

    pred_intervals = (
        np.array([[n["start"], n["end"]] for n in pred_notes])
        if pred_notes
        else np.empty((0, 2))
    )
    # Convert MIDI pitch to Hz for mir_eval
    pred_pitches_hz = (
        np.array([midi_to_hz(n["midi_pitch"]) for n in pred_notes])
        if pred_notes
        else np.array([])
    )

    # Call mir_eval.transcription.evaluate with the correct arguments
    results = transcription_evaluate(
        true_intervals, true_pitches_hz, pred_intervals, pred_pitches_hz
    )

    return {
        "precision": results["Precision"],
        "recall": results["Recall"],
        "f1": results["F-measure"],
    }


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
    Optimizes based on transcription F1 score.
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
                    true_notes = sample["notes"]

                    # Get predicted notes using current threshold
                    _, _, pred_notes = algo.extract_pitch(audio, threshold)

                    # Evaluate transcription performance
                    metrics = evaluate_transcription(true_notes, pred_notes)
                    f1_scores.append(metrics["f1"])

                except Exception as e:
                    print(
                        f"Error processing {algo_class.get_name()} with threshold {threshold} on sample {idx}: {e}"
                    )
                    continue

            if not f1_scores:
                continue

            mean_f1 = np.mean(f1_scores)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_threshold = threshold

        optimal_thresholds[algo_name] = best_threshold
        print(
            f"{algo_name}: optimal threshold = {best_threshold:.3f} (F1={best_f1:.3f})"
        )

    return optimal_thresholds


def evaluate_pitch_algorithms(
    dataset: Dataset,
    fmin: int,
    fmax: int,
    algorithms: List[Tuple],
) -> Dict:
    """
    Evaluate multiple pitch detection algorithms with comprehensive metrics.
    Focuses solely on transcription evaluation.
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

        # For transcription evaluation
        all_true_notes = []
        all_pred_notes = []
        num_files_processed = 0

        for idx in tqdm(
            range(len(dataset)), desc=f"Processing {algo_name}", leave=False
        ):
            try:
                sample = dataset[idx]
                audio = sample["audio"].numpy()
                true_notes = sample["notes"]

                # Skip files without any notes
                if not true_notes:
                    continue

                _, _, pred_notes = algo.extract_pitch(audio, threshold)

                # Accumulate note data
                all_true_notes.extend(true_notes)
                all_pred_notes.extend(pred_notes)
                num_files_processed += 1

            except Exception as e:
                print(f"Error processing file {idx} with {algo_name}: {str(e)}")
                continue

        # Skip if no valid data
        if num_files_processed == 0:
            results[algo_name] = {
                "transcription": {"f1": np.nan, "precision": np.nan, "recall": np.nan},
                "num_files_processed": 0,
                "total_notes": 0,
            }
            continue

        # Compute transcription metrics
        transcription_metrics = evaluate_transcription(all_true_notes, all_pred_notes)

        results[algo_name] = {
            "transcription": transcription_metrics,
            "num_files_processed": num_files_processed,
            "total_notes": len(all_true_notes),
        }

    return results


def print_evaluation_results(metrics: Dict):
    print("\nTranscription Evaluation Results")
    print("=" * 80)

    def format_metric(value: float, percentage: bool = False) -> str:
        if np.isnan(value):
            return "N/A".center(12)
        if percentage:
            return f"{value * 100:.1f}%".center(12)
        return f"{value:.2f}".center(12)

    # Define metrics to display
    headers = ["Algorithm", "Precision ↑", "Recall ↑", "F1 ↑"]
    format_str = "{:<20} " + " ".join(["{:^12}"] * (len(headers) - 1))

    print(format_str.format(*headers))
    print("-" * 80)

    for algo_name, algo_metrics in metrics.items():
        try:
            t_metrics = algo_metrics["transcription"]
            precision = format_metric(t_metrics["precision"], True)
            recall = format_metric(t_metrics["recall"], True)
            f1 = format_metric(t_metrics["f1"], True)

            print(format_str.format(algo_name[:20], precision, recall, f1))
        except (KeyError, TypeError) as e:
            error_values = ["N/A".center(12)] * 5
            print(format_str.format(algo_name[:20], *error_values))
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
        choices=list_transcription_datasets(),
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
        "--fmax", type=float, default=1200.0, help="Maximum frequency in Hz"
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
    dataset_class = get_transcription_dataset(args.dataset)
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
    print(f"Optimal thresholds: {optimal_thresholds}")

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
