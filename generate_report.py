#!/usr/bin/env python3
"""
Comprehensive analysis script for pitch detection benchmark results.
Generates a detailed markdown report with tables and insights.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_all_results(results_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load all JSON result files from the results directory."""
    pitch_results = []
    speed_results = []

    for file_path in Path(results_dir).glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Distinguish between speed and pitch results
            if data.get("metadata", {}).get("benchmark_type") == "speed":
                speed_results.append(data)
            elif "results" in data and "combined_score" in data.get(
                "results", {}
            ):
                # This looks like a pitch benchmark result
                pitch_results.append(data)
            else:
                print(
                    f"Warning: Unrecognized result format in {file_path.name}"
                )

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    return pitch_results, speed_results


def aggregate_pitch_results(
    pitch_results: List[Dict],
) -> Dict[str, Dict[str, List[float]]]:
    """Aggregate pitch results by algorithm and dataset."""
    # Structure: {algorithm: {dataset: [scores]}}
    aggregated = defaultdict(lambda: defaultdict(list))

    for result in pitch_results:
        try:
            algo_name = result["metadata"]["algorithm_name"]
            dataset_name = result["metadata"]["dataset_name"]
            combined_score = result["results"]["combined_score"]

            if combined_score is not None and not np.isnan(combined_score):
                aggregated[algo_name][dataset_name].append(combined_score)
        except KeyError as e:
            print(f"Warning: Missing key in pitch result: {e}")
            continue

    return aggregated


def aggregate_speed_results(
    speed_results: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """Aggregate speed results by algorithm."""
    # Structure: {algorithm: {metric: value}}
    aggregated = {}

    for result in speed_results:
        try:
            algo_name = result["metadata"]["algorithm_name"]
            cpu_perf = result["results"]["device_performance"].get("cpu", {})

            if cpu_perf.get("supported", False):
                aggregated[algo_name] = {
                    "absolute_time_ms": cpu_perf["absolute_time_ms"],
                    "relative_speed": cpu_perf.get("relative_speed", 1.0),
                    "supports_cuda": result["results"]["supports_cuda"],
                }
        except KeyError as e:
            print(f"Warning: Missing key in speed result: {e}")
            continue

    return aggregated


def collect_detailed_metrics(
    pitch_results: List[Dict],
) -> Dict[str, Dict[str, List[Any]]]:
    """Collect detailed metrics for in-depth analysis."""
    # Structure: {algorithm: {metric: [values]}}
    metrics = defaultdict(lambda: defaultdict(list))

    for result in pitch_results:
        try:
            algo_name = result["metadata"]["algorithm_name"]
            results_data = result["results"]

            # Voicing detection metrics
            voicing = results_data.get("voicing_detection", {})
            for metric in ["precision", "recall", "f1"]:
                if metric in voicing and voicing[metric] is not None:
                    metrics[algo_name][f"voicing_{metric}"].append(
                        voicing[metric]
                    )

            # Pitch accuracy metrics
            pitch = results_data.get("pitch_accuracy", {})
            for metric in [
                "rpa",
                "rca",
                "cents_error",
                "rmse",
                "octave_error_rate",
                "gross_error_rate",
            ]:
                if metric in pitch and pitch[metric] is not None:
                    metrics[algo_name][f"pitch_{metric}"].append(pitch[metric])

            # Smoothness metrics
            smoothness = results_data.get("smoothness_metrics", {})
            for metric in ["relative_smoothness", "continuity_breaks"]:
                if metric in smoothness and smoothness[metric] is not None:
                    metrics[algo_name][metric].append(smoothness[metric])

            # Optimal threshold
            if (
                "optimal_threshold" in results_data
                and results_data["optimal_threshold"] is not None
            ):
                metrics[algo_name]["optimal_threshold"].append(
                    results_data["optimal_threshold"]
                )

            # Combined score for consistency analysis
            if (
                "combined_score" in results_data
                and results_data["combined_score"] is not None
            ):
                metrics[algo_name]["combined_score"].append(
                    results_data["combined_score"]
                )

            # Execution time
            if "execution_time_seconds" in result["metadata"]:
                metrics[algo_name]["execution_time"].append(
                    result["metadata"]["execution_time_seconds"]
                )

        except KeyError as e:
            print(f"Warning: Missing key in detailed metrics: {e}")
            continue

    return metrics


def generate_dataset_descriptions() -> str:
    """Generate dataset descriptions section."""
    descriptions = "## Dataset Descriptions\n\n"

    descriptions += "The benchmark evaluates algorithms across diverse datasets covering speech, music, synthetic, and real-world conditions:\n\n"

    descriptions += (
        "| **Dataset** | **Domain** | **Type** | **Description** |\n"
    )
    descriptions += "|---|---|---|---|\n"

    dataset_info = [
        (
            "NSynth",
            "Music",
            "Synthetic",
            "Single-note synthetic audio from musical instruments with accurate pitch labels. Lacks temporal/spectral complexity of real-world environments.",
        ),
        (
            "PTDB",
            "Speech",
            "Real",
            "Speech recordings with laryngograph signals capturing vocal fold vibrations. Ground truth derived from high-pass filtered laryngograph signals processed with RAPT algorithm.",
        ),
        (
            "PTDBNoisy",
            "Speech",
            "Real",
            "Subset of 347 PTDB files (7.4%) with noticeable noise that were excluded from main evaluation.",
        ),
        (
            "MIR1K",
            "Music",
            "Real",
            "Vocal excerpts with pitch contours initially extracted algorithmically (e.g., YIN) followed by manual correction. Labels still reflect some algorithmic biases.",
        ),
        (
            "MDBStemSynth",
            "Music",
            "Synthetic",
            "Musically structured synthetic audio with accurate pitch annotations. Valuable for controlled evaluation but lacks real-world acoustic variability.",
        ),
        (
            "Vocadito",
            "Music",
            "Real",
            "Solo vocal recordings with pitch annotations derived from pYIN algorithm, refined through manual verification process.",
        ),
        (
            "Bach10Synth",
            "Music",
            "Synthetic",
            "High-quality pitch labels for synthesized musical performances. Similar to MDB-STEM-Synth but focused on Bach compositions.",
        ),
        (
            "SpeechSynth",
            "Speech",
            "Synthetic",
            "Synthetic Mandarin speech generated using LightSpeech TTS model. Trained on 97.48 hours from AISHELL-3 and Biaobei datasets, providing exact pitch ground truth.",
        ),
    ]

    for dataset, domain, type_str, desc in dataset_info:
        descriptions += f"| **{dataset}** | {domain} | {type_str} | {desc} |\n"

    descriptions += "\n**Key Characteristics:**\n"
    descriptions += "- **Synthetic datasets** provide perfect ground truth but may lack real-world complexity\n"
    descriptions += "- **Real datasets** capture natural acoustic variations but have imperfect ground truth annotations\n"
    descriptions += (
        "- **Speech datasets** focus on vocal pitch tracking challenges\n"
    )
    descriptions += "- **Music datasets** encompass instrumental and vocal music scenarios\n"
    descriptions += "- **SpeechSynth** addresses the gap of lacking synthetic speech data with accurate pitch labels\n\n"

    return descriptions


def generate_methodology_section() -> str:
    """Generate methodology and metric definitions section."""
    methodology = "## Benchmark Methodology\n\n"

    methodology += "### Evaluation Setup\n"
    methodology += "This benchmark evaluates pitch detection algorithms across multiple datasets with different characteristics, "
    methodology += "including synthetic and real audio from speech and music domains. Each algorithm is tested on noisy audio "
    methodology += "generated by mixing clean datasets with CHiME background noise at various signal-to-noise ratios (10-30 dB) "
    methodology += "and voice gain variations (-6 to +6 dB).\n\n"

    methodology += "### Performance Metric Definition\n"
    methodology += "The **Overall Performance Rankings** show the **Harmonic Mean (HM)** score as percentages, computed from six complementary components:\n\n"
    methodology += "**HM = 6 / (1/RPA + 1/CA + 1/P + 1/R + 1/OA + 1/GEA)**\n\n"
    methodology += "Where:\n"
    methodology += "- **RPA** (Raw Pitch Accuracy): Fraction of voiced frames within 50 cents of ground truth\n"
    methodology += "- **CA** (Cents Accuracy): exp(-mean_cents_error/500), penalizing larger deviations exponentially\n"
    methodology += "- **P** (Voicing Precision): TP/(TP+FP), fraction of predicted voiced frames that are truly voiced\n"
    methodology += "- **R** (Voicing Recall): TP/(TP+FN), fraction of truly voiced frames detected\n"
    methodology += "- **OA** (Octave Accuracy): exp(-10×octave_error_rate), robustness against octave errors\n"
    methodology += "- **GEA** (Gross Error Accuracy): exp(-5×gross_error_rate), penalizing deviations >200 cents\n\n"

    methodology += "### Speed Benchmark Details\n"
    methodology += "CPU timing measurements are performed on 1-second audio signals at 22.05 kHz sample rate with 256-sample hop length. "
    methodology += "The reported **CPU Time (ms)** represents the average processing time per 1-second audio segment across multiple runs. "
    methodology += "**Relative Speed** shows performance relative to CREPE as the baseline algorithm.\n\n"

    methodology += "### Optimal Threshold Analysis\n"
    methodology += "The **Optimal Threshold** refers to the voicing confidence threshold that maximizes the Harmonic Mean score. "
    methodology += "Algorithms test multiple thresholds (0.0 to 1.0 in steps of 0.1) and select the one yielding the highest combined score. "
    methodology += "**CV** stands for Coefficient of Variation (std/mean), measuring consistency across datasets.\n\n"

    return methodology


def generate_combined_score_table(
    aggregated_results: Dict[str, Dict[str, List[float]]],
) -> str:
    """Generate the main performance table showing combined scores as percentages."""
    if not aggregated_results:
        return "No pitch benchmark results found.\n"

    # Get all unique datasets
    all_datasets = set()
    for algo_data in aggregated_results.values():
        all_datasets.update(algo_data.keys())
    all_datasets = sorted(all_datasets)

    # Calculate averages for each algorithm-dataset combination
    table_data = []
    for algo_name in sorted(aggregated_results.keys()):
        row_data = {"algorithm": algo_name, "scores": [], "average": 0}
        dataset_scores = []

        for dataset in all_datasets:
            scores = aggregated_results[algo_name].get(dataset, [])
            if scores:
                avg_score = np.mean(scores) * 100  # Convert to percentage
                row_data["scores"].append(f"{avg_score:.1f}%")
                dataset_scores.append(avg_score)
            else:
                row_data["scores"].append("N/A")

        if dataset_scores:
            row_data["average"] = np.mean(dataset_scores)

        table_data.append(row_data)

    # Sort by average performance (descending)
    table_data.sort(key=lambda x: x["average"], reverse=True)

    # Find best score in each column for highlighting
    best_scores_per_dataset = {}
    for dataset_idx, dataset in enumerate(all_datasets):
        best_score = -1
        for row in table_data:
            if (
                dataset_idx < len(row["scores"])
                and row["scores"][dataset_idx] != "N/A"
            ):
                score_val = float(row["scores"][dataset_idx].rstrip("%"))
                if score_val > best_score:
                    best_score = score_val
        best_scores_per_dataset[dataset] = best_score

    # Generate markdown table
    header = (
        "| **Algorithm** | "
        + " | ".join([f"**{dataset}**" for dataset in all_datasets])
        + " | **Average** |\n"
    )
    separator = "|" + "---|" * (len(all_datasets) + 2) + "\n"

    rows = []
    for i, row in enumerate(table_data):
        algo_name = row["algorithm"]

        # Highlight best scores in each column
        formatted_scores = []
        for dataset_idx, (dataset, score_str) in enumerate(
            zip(all_datasets, row["scores"])
        ):
            if score_str != "N/A":
                score_val = float(score_str.rstrip("%"))
                if (
                    abs(score_val - best_scores_per_dataset[dataset]) < 0.1
                ):  # Best score
                    formatted_scores.append(f"**{score_str}**")
                else:
                    formatted_scores.append(score_str)
            else:
                formatted_scores.append(score_str)

        scores_str = " | ".join(formatted_scores)

        # Highlight best average (first row after sorting)
        if i == 0:
            avg_str = (
                f"**{row['average']:.1f}%**" if row["average"] > 0 else "N/A"
            )
            algo_name = (
                f"**{algo_name}**"  # Bold the best overall algorithm name
            )
        else:
            avg_str = f"{row['average']:.1f}%" if row["average"] > 0 else "N/A"

        rows.append(f"| {algo_name} | {scores_str} | {avg_str} |\n")

    return (
        "## Overall Performance Rankings\n\n"
        + header
        + separator
        + "".join(rows)
        + "\n"
    )


def generate_speed_table(speed_results: Dict[str, Dict[str, float]]) -> str:
    """Generate CPU speed performance table."""
    if not speed_results:
        return "No speed benchmark results found.\n"

    # Sort by absolute time (ascending - faster is better)
    sorted_algos = sorted(
        speed_results.items(), key=lambda x: x[1]["absolute_time_ms"]
    )

    header = "| **Algorithm** | **CPU Time (ms) ↓** | **Relative Speed ↑** |\n"
    separator = "|---|---|---|\n"

    rows = []
    for algo_name, metrics in sorted_algos:
        time_ms = f"{metrics['absolute_time_ms']:.1f}"
        rel_speed = f"{metrics['relative_speed']:.2f}x"

        # Bold the fastest
        if metrics == sorted_algos[0][1]:
            time_ms = f"**{time_ms}**"
            rel_speed = f"**{rel_speed}**"
            algo_name = f"**{algo_name}**"

        rows.append(f"| {algo_name} | {time_ms} | {rel_speed} |\n")

    return (
        "## Speed Performance (CPU)\n\n"
        + header
        + separator
        + "".join(rows)
        + "\n"
    )


def generate_detailed_analysis(
    detailed_metrics: Dict[str, Dict[str, List[Any]]],
) -> str:
    """Generate detailed analysis of various metrics."""
    if not detailed_metrics:
        return "No detailed metrics available.\n"

    analysis = "## Detailed Performance Analysis\n\n"

    # Voicing Detection Analysis
    analysis += "### Voicing Detection Performance\n"
    analysis += "Measures how well algorithms distinguish between voiced (pitched) and unvoiced (unpitched) audio segments.\n\n"
    analysis += (
        "| **Algorithm** | **Precision ↑** | **Recall ↑** | **F1-Score ↑** |\n"
    )
    analysis += "|---|---|---|---|\n"

    voicing_data = []
    for algo_name in sorted(detailed_metrics.keys()):
        metrics = detailed_metrics[algo_name]
        precision = (
            np.mean(metrics.get("voicing_precision", [0]))
            if metrics.get("voicing_precision")
            else 0
        )
        recall = (
            np.mean(metrics.get("voicing_recall", [0]))
            if metrics.get("voicing_recall")
            else 0
        )
        f1 = (
            np.mean(metrics.get("voicing_f1", [0]))
            if metrics.get("voicing_f1")
            else 0
        )

        voicing_data.append(
            {
                "name": algo_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    # Find best in each column
    best_precision = max(voicing_data, key=lambda x: x["precision"])[
        "precision"
    ]
    best_recall = max(voicing_data, key=lambda x: x["recall"])["recall"]
    best_f1 = max(voicing_data, key=lambda x: x["f1"])["f1"]

    for data in voicing_data:
        precision_str = (
            f"**{data['precision']:.3f}**"
            if abs(data["precision"] - best_precision) < 1e-6
            else f"{data['precision']:.3f}"
        )
        recall_str = (
            f"**{data['recall']:.3f}**"
            if abs(data["recall"] - best_recall) < 1e-6
            else f"{data['recall']:.3f}"
        )
        f1_str = (
            f"**{data['f1']:.3f}**"
            if abs(data["f1"] - best_f1) < 1e-6
            else f"{data['f1']:.3f}"
        )

        algo_name = (
            f"**{data['name']}**"
            if abs(data["f1"] - best_f1) < 1e-6
            else data["name"]
        )

        analysis += (
            f"| {algo_name} | {precision_str} | {recall_str} | {f1_str} |\n"
        )

    analysis += "\n"

    # Pitch Accuracy Analysis
    analysis += "### Pitch Accuracy Metrics\n"
    analysis += "Detailed pitch estimation accuracy across different error types and magnitudes.\n\n"
    analysis += "| **Algorithm** | **RPA ↑** | **RCA ↑** | **Cents Error ↓** | **RMSE (Hz) ↓** | **Octave Error ↓** | **Gross Error ↓** |\n"
    analysis += "|---|---|---|---|---|---|---|\n"

    pitch_data = []
    for algo_name in sorted(detailed_metrics.keys()):
        metrics = detailed_metrics[algo_name]
        rpa = (
            np.mean(metrics.get("pitch_rpa", [0]))
            if metrics.get("pitch_rpa")
            else 0
        )
        rca = (
            np.mean(metrics.get("pitch_rca", [0]))
            if metrics.get("pitch_rca")
            else 0
        )
        cents = (
            np.mean(metrics.get("pitch_cents_error", [0]))
            if metrics.get("pitch_cents_error")
            else 0
        )
        rmse = (
            np.mean(metrics.get("pitch_rmse", [0]))
            if metrics.get("pitch_rmse")
            else 0
        )
        octave_err = (
            np.mean(metrics.get("pitch_octave_error_rate", [0]))
            if metrics.get("pitch_octave_error_rate")
            else 0
        )
        gross_err = (
            np.mean(metrics.get("pitch_gross_error_rate", [0]))
            if metrics.get("pitch_gross_error_rate")
            else 0
        )

        pitch_data.append(
            {
                "name": algo_name,
                "rpa": rpa,
                "rca": rca,
                "cents": cents,
                "rmse": rmse,
                "octave_err": octave_err,
                "gross_err": gross_err,
            }
        )

    # Find best in each column (higher is better for RPA/RCA, lower is better for errors)
    best_rpa = max(pitch_data, key=lambda x: x["rpa"])["rpa"]
    best_rca = max(pitch_data, key=lambda x: x["rca"])["rca"]
    best_cents = min(
        [x["cents"] for x in pitch_data if x["cents"] > 0], default=0
    )
    best_rmse = min([x["rmse"] for x in pitch_data if x["rmse"] > 0], default=0)
    best_octave = min(
        [x["octave_err"] for x in pitch_data if x["octave_err"] >= 0], default=0
    )
    best_gross = min(
        [x["gross_err"] for x in pitch_data if x["gross_err"] >= 0], default=0
    )

    for data in pitch_data:
        rpa_str = (
            f"**{data['rpa']:.3f}**"
            if abs(data["rpa"] - best_rpa) < 1e-6
            else f"{data['rpa']:.3f}"
        )
        rca_str = (
            f"**{data['rca']:.3f}**"
            if abs(data["rca"] - best_rca) < 1e-6
            else f"{data['rca']:.3f}"
        )
        cents_str = (
            f"**{data['cents']:.1f}**"
            if data["cents"] > 0 and abs(data["cents"] - best_cents) < 1e-6
            else f"{data['cents']:.1f}"
        )
        rmse_str = (
            f"**{data['rmse']:.1f}**"
            if data["rmse"] > 0 and abs(data["rmse"] - best_rmse) < 1e-6
            else f"{data['rmse']:.1f}"
        )
        octave_str = (
            f"**{data['octave_err']:.3f}**"
            if abs(data["octave_err"] - best_octave) < 1e-6
            else f"{data['octave_err']:.3f}"
        )
        gross_str = (
            f"**{data['gross_err']:.3f}**"
            if abs(data["gross_err"] - best_gross) < 1e-6
            else f"{data['gross_err']:.3f}"
        )

        analysis += f"| {data['name']} | {rpa_str} | {rca_str} | {cents_str} | {rmse_str} | {octave_str} | {gross_str} |\n"

    analysis += "\n**Additional Metric Definitions:**\n"
    analysis += "- **RCA** (Raw Chroma Accuracy): Fraction with correct pitch class (note name), ignoring octave\n"
    analysis += "- **Cents Error**: Mean absolute pitch deviation in cents (raw error, before exponential transform used in CA)\n"
    analysis += "- **RMSE**: Root Mean Square Error in Hz\n\n"

    # Smoothness Analysis
    analysis += "### Pitch Contour Smoothness\n"
    analysis += (
        "Measures the temporal stability and continuity of pitch tracks.\n\n"
    )
    analysis += "| **Algorithm** | **Relative Smoothness ↓** | **Continuity Breaks ↓** | **Overall Smoothness Rank ↓** |\n"
    analysis += "|---|---|---|---|\n"

    smoothness_data = []
    for algo_name in sorted(detailed_metrics.keys()):
        metrics = detailed_metrics[algo_name]
        rel_smooth = metrics.get("relative_smoothness", [])
        breaks = metrics.get("continuity_breaks", [])

        avg_smooth = np.nanmean(rel_smooth) if rel_smooth else np.nan
        avg_breaks = np.nanmean(breaks) if breaks else np.nan

        smoothness_data.append(
            {
                "name": algo_name,
                "rel_smooth": avg_smooth,
                "breaks": avg_breaks,
                "smooth_rank": np.inf,
                "breaks_rank": np.inf,
            }
        )

    # Calculate ranks for each metric (1 = best, i.e., lowest value)
    # Rank relative smoothness
    valid_smooth_algos = [
        d for d in smoothness_data if not np.isnan(d["rel_smooth"])
    ]
    valid_smooth_algos.sort(key=lambda x: x["rel_smooth"])
    for rank, data in enumerate(valid_smooth_algos, 1):
        data["smooth_rank"] = rank

    # Rank continuity breaks
    valid_breaks_algos = [
        d for d in smoothness_data if not np.isnan(d["breaks"])
    ]
    valid_breaks_algos.sort(key=lambda x: x["breaks"])
    for rank, data in enumerate(valid_breaks_algos, 1):
        data["breaks_rank"] = rank

    # Calculate combined rank (average of individual ranks)
    for data in smoothness_data:
        ranks = []
        if data["smooth_rank"] != np.inf:
            ranks.append(data["smooth_rank"])
        if data["breaks_rank"] != np.inf:
            ranks.append(data["breaks_rank"])

        if ranks:
            data["combined_rank"] = np.mean(ranks)
        else:
            data["combined_rank"] = np.inf

    # Sort by combined rank for display
    smoothness_data.sort(key=lambda x: x["combined_rank"])

    # Find best values for highlighting
    valid_smooth = [
        x["rel_smooth"]
        for x in smoothness_data
        if not np.isnan(x["rel_smooth"])
    ]
    valid_breaks = [
        x["breaks"] for x in smoothness_data if not np.isnan(x["breaks"])
    ]

    best_smooth = min(valid_smooth) if valid_smooth else np.nan
    best_breaks = min(valid_breaks) if valid_breaks else np.nan
    best_combined = min(
        [
            x["combined_rank"]
            for x in smoothness_data
            if x["combined_rank"] != np.inf
        ],
        default=np.inf,
    )

    for data in smoothness_data:
        smooth_str = (
            f"{data['rel_smooth']:.3f}"
            if not np.isnan(data["rel_smooth"])
            else "N/A"
        )
        breaks_str = (
            f"{data['breaks']:.3f}" if not np.isnan(data["breaks"]) else "N/A"
        )

        # Highlight best individual metrics
        if (
            not np.isnan(data["rel_smooth"])
            and abs(data["rel_smooth"] - best_smooth) < 1e-6
        ):
            smooth_str = f"**{smooth_str}**"
        if (
            not np.isnan(data["breaks"])
            and abs(data["breaks"] - best_breaks) < 1e-6
        ):
            breaks_str = f"**{breaks_str}**"

        # Combined rank display and highlighting
        if data["combined_rank"] != np.inf:
            combined_str = f"{data['combined_rank']:.1f}"
            if abs(data["combined_rank"] - best_combined) < 1e-6:
                combined_str = f"**{combined_str}**"
                algo_name = f"**{data['name']}**"  # Bold algorithm name for best overall
            else:
                algo_name = data["name"]
        else:
            combined_str = "N/A"
            algo_name = data["name"]

        analysis += (
            f"| {algo_name} | {smooth_str} | {breaks_str} | {combined_str} |\n"
        )

    analysis += "\n**Metric Definitions:**\n"
    analysis += "- **Relative Smoothness**: Coefficient of variation of consecutive pitch changes (std/mean of relative frame-to-frame changes)\n"
    analysis += "- **Continuity Breaks**: Fraction of ground-truth voiced segments where predicted voicing has gaps\n"
    analysis += "- **Overall Smoothness Rank**: Average rank across both smoothness metrics (1=best, lower is better)\n\n"

    # Threshold Analysis
    analysis += "### Optimal Threshold Analysis\n"
    analysis += "Voicing confidence thresholds that maximize overall performance scores.\n\n"
    analysis += (
        "| **Algorithm** | **Mean Threshold** | **Std Dev ↓** | **Range** |\n"
    )
    analysis += "|---|---|---|---|\n"

    threshold_data = []
    for algo_name in sorted(detailed_metrics.keys()):
        thresholds = detailed_metrics[algo_name].get("optimal_threshold", [])
        if thresholds:
            mean_thresh = np.mean(thresholds)
            std_thresh = np.std(thresholds)
            min_thresh = np.min(thresholds)
            max_thresh = np.max(thresholds)
            threshold_data.append(
                {
                    "name": algo_name,
                    "mean": mean_thresh,
                    "std": std_thresh,
                    "range": f"{min_thresh:.2f}-{max_thresh:.2f}",
                }
            )
        else:
            threshold_data.append(
                {"name": algo_name, "mean": None, "std": None, "range": "N/A"}
            )

    # Find best (lowest std dev)
    valid_stds = [x["std"] for x in threshold_data if x["std"] is not None]
    best_std = min(valid_stds) if valid_stds else None

    for data in threshold_data:
        if data["mean"] is not None:
            mean_str = f"{data['mean']:.3f}"
            std_str = f"{data['std']:.3f}"
            if best_std is not None and abs(data["std"] - best_std) < 1e-6:
                std_str = f"**{std_str}**"
            analysis += f"| {data['name']} | {mean_str} | {std_str} | {data['range']} |\n"
        else:
            analysis += f"| {data['name']} | N/A | N/A | N/A |\n"

    analysis += "\n"

    # Consistency Analysis
    analysis += "### Algorithm Consistency\n"
    analysis += "Measures performance stability across different datasets using Coefficient of Variation (CV = std/mean).\n\n"
    analysis += (
        "| **Algorithm** | **Performance CV ↓** | **Threshold CV ↓** |\n"
    )
    analysis += "|---|---|---|\n"

    consistency_data = []
    for algo_name in sorted(detailed_metrics.keys()):
        # Get all combined scores for this algorithm across datasets
        all_scores = []
        combined_scores = detailed_metrics[algo_name].get("combined_score", [])
        for score in combined_scores:
            if score is not None and not np.isnan(score):
                all_scores.append(score)

        thresholds = detailed_metrics[algo_name].get("optimal_threshold", [])

        perf_cv = (
            np.std(all_scores) / np.mean(all_scores)
            if len(all_scores) > 1 and np.mean(all_scores) > 0
            else np.nan
        )
        thresh_cv = (
            np.std(thresholds) / np.mean(thresholds)
            if len(thresholds) > 1 and np.mean(thresholds) > 0
            else np.nan
        )

        consistency_data.append(
            {"name": algo_name, "perf_cv": perf_cv, "thresh_cv": thresh_cv}
        )

    # Find best (lowest CV values)
    valid_perf_cvs = [
        x["perf_cv"] for x in consistency_data if not np.isnan(x["perf_cv"])
    ]
    valid_thresh_cvs = [
        x["thresh_cv"] for x in consistency_data if not np.isnan(x["thresh_cv"])
    ]

    best_perf_cv = min(valid_perf_cvs) if valid_perf_cvs else np.nan
    best_thresh_cv = min(valid_thresh_cvs) if valid_thresh_cvs else np.nan

    for data in consistency_data:
        perf_cv_str = (
            f"{data['perf_cv']:.3f}" if not np.isnan(data["perf_cv"]) else "N/A"
        )
        thresh_cv_str = (
            f"{data['thresh_cv']:.3f}"
            if not np.isnan(data["thresh_cv"])
            else "N/A"
        )

        if (
            not np.isnan(data["perf_cv"])
            and abs(data["perf_cv"] - best_perf_cv) < 1e-6
        ):
            perf_cv_str = f"**{perf_cv_str}**"
        if (
            not np.isnan(data["thresh_cv"])
            and abs(data["thresh_cv"] - best_thresh_cv) < 1e-6
        ):
            thresh_cv_str = f"**{thresh_cv_str}**"

        analysis += f"| {data['name']} | {perf_cv_str} | {thresh_cv_str} |\n"

    return analysis + "\n"


def generate_subset_analysis(
    aggregated_results: Dict[str, Dict[str, List[float]]],
) -> str:
    """Generate analysis by dataset subsets (Origin, Domain, Cross-Dimension)."""
    if not aggregated_results:
        return ""

    # Define dataset categories
    categories = {
        "origin": {
            "Synthetic": [
                "Bach10Synth",
                "MDBStemSynth",
                "SpeechSynth",
                "NSynth",
            ],
            "Real": ["MIR1K", "PTDB", "PTDBNoisy", "Vocadito"],
        },
        "domain": {
            "Speech": ["PTDB", "PTDBNoisy", "SpeechSynth"],
            "Music": [
                "Bach10Synth",
                "MDBStemSynth",
                "NSynth",
                "Vocadito",
                "MIR1K",
            ],
        },
        "cross_dimension": {
            "Synthetic + Speech": ["SpeechSynth"],
            "Synthetic + Music": ["Bach10Synth", "MDBStemSynth", "NSynth"],
            "Real + Speech": ["PTDB", "PTDBNoisy"],
            "Real + Music": ["Vocadito", "MIR1K"],
        },
    }

    analysis = "## Performance by Dataset Subsets\n\n"

    for category_name, subcategories in categories.items():
        if category_name == "origin":
            analysis += "### By Origin\n"
            analysis += "- **Synthetic**: Bach10Synth, MDBStemSynth, SpeechSynth, NSynth\n"
            analysis += "- **Real**: MIR1K, PTDB, PTDBNoisy, Vocadito\n\n"
        elif category_name == "domain":
            analysis += "### By Domain\n"
            analysis += "- **Speech**: PTDB, PTDBNoisy, SpeechSynth\n"
            analysis += "- **Music**: Bach10Synth, MDBStemSynth, NSynth, Vocadito, MIR1K\n\n"
        elif category_name == "cross_dimension":
            analysis += "### By Cross-Dimension\n"
            analysis += "- **Synthetic + Speech**: SpeechSynth\n"
            analysis += (
                "- **Synthetic + Music**: Bach10Synth, MDBStemSynth, NSynth\n"
            )
            analysis += "- **Real + Speech**: PTDB, PTDBNoisy\n"
            analysis += "- **Real + Music**: Vocadito, MIR1K\n\n"

        # Create table header
        header = (
            "| **Algorithm** | "
            + " | ".join([f"**{subcat}**" for subcat in subcategories.keys()])
            + " |\n"
        )
        separator = "|" + "---|" * (len(subcategories) + 1) + "\n"

        # Calculate averages for each algorithm-subcategory combination
        table_data = []
        for algo_name in sorted(aggregated_results.keys()):
            row_data = {"algorithm": algo_name, "scores": []}

            for subcat_name, datasets in subcategories.items():
                subcat_scores = []
                for dataset in datasets:
                    # Map common dataset name variations
                    dataset_variants = [
                        dataset,
                        dataset.replace("Synth", ""),
                        dataset.replace("Synth", "-synth"),
                        dataset.replace("Synth", "_synth"),
                        dataset.lower(),
                        dataset.upper(),
                    ]

                    for variant in dataset_variants:
                        if variant in aggregated_results[algo_name]:
                            scores = aggregated_results[algo_name][variant]
                            if scores:
                                subcat_scores.extend(scores)
                            break

                if subcat_scores:
                    avg_score = np.mean(subcat_scores) * 100
                    row_data["scores"].append(avg_score)
                else:
                    row_data["scores"].append(None)

            table_data.append(row_data)

        # Find best score in each column for highlighting
        best_scores = []
        for col_idx in range(len(subcategories)):
            col_scores = [
                row["scores"][col_idx]
                for row in table_data
                if row["scores"][col_idx] is not None
            ]
            best_scores.append(max(col_scores) if col_scores else None)

        # Generate table rows
        analysis += header + separator
        for row in table_data:
            algo_name = row["algorithm"]
            formatted_scores = []

            for i, score in enumerate(row["scores"]):
                if score is not None:
                    score_str = f"{score:.1f}%"
                    # Highlight best scores
                    if (
                        best_scores[i] is not None
                        and abs(score - best_scores[i]) < 0.1
                    ):
                        score_str = f"**{score_str}**"
                        if (
                            i == 0
                        ):  # If best in first column, also bold algorithm name
                            algo_name = (
                                f"**{algo_name}**"
                                if algo_name[0] != "*"
                                else algo_name
                            )
                    formatted_scores.append(score_str)
                else:
                    formatted_scores.append("N/A")

            scores_str = " | ".join(formatted_scores)
            analysis += f"| {algo_name} | {scores_str} |\n"

        analysis += "\n"

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive analysis report from benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_report.md",
        help="Output markdown file path",
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist.")
        return 1

    print(f"Loading results from {args.results_dir}...")
    pitch_results, speed_results = load_all_results(args.results_dir)

    print(
        f"Found {len(pitch_results)} pitch benchmark results and {len(speed_results)} speed benchmark results."
    )

    # Process data
    aggregated_pitch = aggregate_pitch_results(pitch_results)
    aggregated_speed = aggregate_speed_results(speed_results)
    detailed_metrics = collect_detailed_metrics(pitch_results)

    # Generate report sections
    print("Generating analysis report...")

    report = "# Pitch Detection Algorithm Benchmark Report\n\n"

    # Add methodology section
    report += generate_methodology_section()

    # Add dataset descriptions
    report += generate_dataset_descriptions()

    # Add main performance table
    report += generate_combined_score_table(aggregated_pitch)

    # Add speed table
    report += generate_speed_table(aggregated_speed)

    # Add detailed analysis
    report += generate_detailed_analysis(detailed_metrics)

    # Add subset analysis
    report += generate_subset_analysis(aggregated_pitch)

    # Write report
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report generated: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
