import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from algorithms import get_algorithm, list_algorithms


def assign_colors_and_styles(num_algorithms):
    """Automatically assign colors and linestyles."""
    colors = plt.get_cmap("tab10", num_algorithms).colors
    linestyles = ["-", "--", "-.", ":"] * (num_algorithms // 4 + 1)
    return colors, linestyles


def calculate_spectrogram_params(fmin: float, fmax: float, sr: int) -> dict:
    """Calculate optimal spectrogram parameters based on frequency range and sampling rate."""
    n_mels = int(np.ceil(128 * (fmax - fmin) / 4000))
    hop_length = int(sr / 100)
    win_length = 4 * hop_length

    return {
        "n_mels": max(32, min(256, n_mels)),
        "hop_length": hop_length,
        "win_length": win_length,
        "n_fft": 2 ** int(np.ceil(np.log2(win_length))),
    }


def compare_pitch_algorithms(
    audio_file: str,
    selected_algorithms: List[str],
    sr: int = 22050,
    hop_size: int = 256,
    fmin: float = 65,
    fmax: float = 300,
    output_file: str = "output.jpg",
):
    """Compare different pitch detection algorithms."""
    try:
        audio, _ = librosa.load(audio_file, sr=sr)
        audio_duration = librosa.get_duration(y=audio, sr=sr)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")

    # Filter algorithms based on selection
    filtered_algorithms = [get_algorithm(algo) for algo in selected_algorithms]

    if not filtered_algorithms:
        raise ValueError("No valid algorithms selected.")

    # Assign colors and linestyles
    colors, linestyles = assign_colors_and_styles(len(filtered_algorithms))

    # Process audio with each algorithm
    results = []
    for algo_class, color, linestyle in zip(filtered_algorithms, colors, linestyles):
        algo_name = algo_class.__name__.replace("PitchAlgorithm", "")
        print(f"Running: {algo_name}")

        try:
            algo_instance = algo_class(
                sample_rate=sr, hop_size=hop_size, fmin=fmin, fmax=fmax
            )

            pitch, periodicity = algo_instance(audio)

            results.append((algo_name, pitch, periodicity, color, linestyle))
        except Exception as e:
            print(f"Error processing {algo_name}: {e}")
            continue

    if not results:
        raise ValueError("No algorithms produced results.")

    ref_length = results[0][2].shape[-1]
    times = np.linspace(0, audio_duration, ref_length)

    # Set up the plot
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Generate and plot spectrogram
    spec_params = calculate_spectrogram_params(fmin, fmax, sr)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, **spec_params)
    spectrogram = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(
        spectrogram,
        sr=sr,
        hop_length=spec_params["hop_length"],
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        ax=ax1,
        cmap="gray_r",
        alpha=0.3,
    )

    # Plot pitch and periodicity
    for algo_name, pitch, periodicity, color, linestyle in results:
        # pitch[periodicity < threshold] = np.nan
        ax1.plot(
            times, pitch, label=algo_name, alpha=0.8, color=color, linestyle=linestyle
        )
        ax2.plot(
            times,
            periodicity,
            label=f"{algo_name}",  # (threshold={threshold:.2f})",
            alpha=0.8,
            color=color,
            linestyle=linestyle,
        )

    # Configure pitch plot
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Pitch Analysis with Spectrogram")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax1.set_xlim(0, audio_duration)

    yticks = np.linspace(fmin, fmax, 10)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([f"{y:.1f}" for y in yticks])
    ax1.set_ylim(fmin, fmax)

    # Configure periodicity plot
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Periodicity")
    ax2.set_title("Periodicity")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    ax2.set_xlim(0, audio_duration)

    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_file}")
    except Exception as e:
        raise RuntimeError(f"Error saving visualization: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare Pitch Detection Algorithms")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file")
    parser.add_argument(
        "--selected_algorithms",
        nargs="+",
        type=str,
        default=list_algorithms(),
        help="List of algorithms to visualize. Separate names by spaces, e.g., 'Praat SWIPE'.",
    )
    parser.add_argument("--sr", type=int, default=22050, help="Sampling rate")
    parser.add_argument("--fmin", type=float, default=65, help="Minimum frequency")
    parser.add_argument("--fmax", type=float, default=300, help="Maximum frequency")
    parser.add_argument("--hop-size", type=int, default=256, help="Hop size in samples")
    parser.add_argument(
        "--output_file", type=str, default="output.jpg", help="Output file name"
    )

    args = parser.parse_args()

    compare_pitch_algorithms(
        audio_file=args.audio_file,
        selected_algorithms=args.selected_algorithms,
        sr=args.sr,
        fmin=args.fmin,
        fmax=args.fmax,
        hop_size=args.hop_size,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
