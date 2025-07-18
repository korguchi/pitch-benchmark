# Pitch Detection Benchmark

A comprehensive benchmark suite for evaluating and comparing different pitch detection algorithms across multiple datasets and metrics.

## 📊 Key Findings

SwiftF0 achieves the highest average harmonic-mean accuracy (79.6%) across six datasets while also delivering near real-time performance (≈42× faster than the TorchCREPE baseline on CPU). pYIN follows with the second-highest average accuracy (68.1%). TorchCREPE ranks third (67.8%) and remains the slowest algorithm, taking ≈5.5 s to process 5 s of audio on CPU. Praat delivers an excellent speed–accuracy balance: it processes 5 s of audio in just 7 ms on CPU (≈809× faster than TorchCREPE) while maintaining a strong overall accuracy of 65.9%. For a detailed breakdown of results, see [Benchmark Results](benchmark_results.md).

| **Algorithm**  | **NSynth** | **PTDB** | **SpeechSynth** | **MIR‑1K** | **MDB‑STEM‑Synth** | **Vocadito** | **Average** |
| -------------- | ---------- | -------- | --------------- | ---------- | ------------------ | ------------ | ----------- |
| BasicPitch     | 11.9%      | 12.8%    | 55.9%           | 25.7%      | 8.1%               | 13.1%        | 21.2%       |
| pYIN           | 17.8%      | 72.3%    | 55.8%           | 89.4%      | **83.6%**          | 89.8%        | 68.1%       |
| Praat          | 22.5%      | 80.4%    | 77.0%           | 74.1%      | 59.1%              | 82.2%        | 65.9%       |
| PENN           | 2.0%       | 82.5%    | 77.0%           | 80.4%      | 61.4%              | 57.2%        | 60.1%       |
| RAPT           | 13.2%      | 70.7%    | 67.3%           | 76.5%      | 70.3%              | 78.0%        | 62.7%       |
| SWIPE          | 13.4%      | 50.8%    | 66.8%           | 73.6%      | 58.6%              | 72.7%        | 56.0%       |
| TorchCREPE     | **73.4%**  | 66.0%    | 82.4%           | 71.4%      | 49.6%              | 64.2%        | 67.8%       |
| YAAPT          | 2.3%       | 67.9%    | 78.7%           | 70.0%      | 24.9%              | 86.0%        | 55.0%       |
| SwiftF0        | 33.6%      | **87.0%**| **88.7%**       | **93.3%**  | 82.6%              | **92.1%**    | **79.6%**   |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

Visualize algorithm comparisons:
```bash
python visualize_algorithms.py audio_file.wav
```

Run speed benchmark:
```bash
python speed_benchmark.py
```

Run pitch detection benchmark:
```bash
python pitch_benchmark.py --dataset DATASET_NAME --data-dir DATA_PATH
```

## 🛠️ Features

- Comprehensive evaluation across various datasets:
  - [PTDB](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html)
  - [NSynth](https://magenta.tensorflow.org/datasets/nsynth)
  - [MDB-stem-synth](https://zenodo.org/records/1481172)
  - [MIR-1K](https://zenodo.org/records/3532216)
  - [Vocadito](https://zenodo.org/records/5578807)
  - A novel synthetic speech dataset: SpeechSynth
- Performance benchmarking for CPU and GPU execution
- Testing under noisy conditions: [CHiME-Home dataset](https://archive.org/details/chime-home)
- Visualization tools for algorithm comparison
- Implementation of popular pitch detection algorithms:
  - [YAAPT](https://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html) (pYAAPT implementation)
  - [Praat](https://github.com/YannickJadoul/Parselmouth) (Parselmouth implementation)
  - [TorchCREPE](https://github.com/maxrmorrison/torchcrepe) (PyTorch implementation of CREPE) and [CREPE](https://github.com/marl/crepe) (original implementation)
  - [Pitch-Estimating Neural Networks (PENN)](https://github.com/interactiveaudiolab/penn)
  - [SWIPE](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.swipe.html) (SPTK implementation)
  - [RAPT](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.rapt.html) (SPTK implementation)
  - [pYIN](https://librosa.org/doc/main/generated/librosa.pyin.html) (librosa implementation)
  - [BasicPitch](https://github.com/spotify/basic-pitch)
  - [SwiftF0](https://github.com/lars76/swift-f0)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{pitch_detection_benchmark,
  title = {Pitch Detection Benchmark},
  author = {Lars Nieradzik},
  year = {2025},
  url = {https://github.com/lars76/pitch-detection-benchmark}
}
```