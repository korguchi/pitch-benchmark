# Pitch Detection Benchmark

A comprehensive benchmark suite for evaluating and comparing different pitch detection algorithms across multiple datasets and metrics.

## 📊 Key Findings

Praat demonstrates the best overall performance, achieving the highest average harmonic-mean accuracy (79.3%) across four datasets and maintaining a 201-cent average error while processing audio faster than all the other algorithms. While TorchCREPE achieves lower pitch errors on individual datasets, it is much slower and has lower recall. For a detailed breakdown of results, see [Benchmark Results](benchmark_results.md).

In the tables below, any entry marked with “†” indicates that the algorithm was trained (or partially trained) on that same dataset, so its performance on that split may be inflated by data leakage and should not be directly compared to the other methods.

### Pitch Accuracy (↑ higher is better)

| Algorithm  | NSynth    | PTDB      | MDB‑STEM‑Synth | SpeechSynth | Average   |
| ---------- | --------- | --------- | -------------- | ----------- | --------- |
| YAAPT      | 64.0%     | **77.6%** | 77.7%          | **82.4%**   | 75.4%     |
| Praat      | **79.0%** | 74.4%     | 81.8%          | 82.2%       | **79.3%** |
| SWIPE      | 63.9%     | 71.4%     | 78.8%          | 79.2%       | 73.3%     |
| RAPT       | 71.9%     | 67.9%     | **85.9%**      | 78.0%       | 75.9%     |
| pYIN       | 73.2%     | 62.8%     | 79.3%          | 75.9%       | 72.8%     |
| TorchCREPE | 76.7%     | 73.2%     | 78.7%†         | 80.8%       | 77.4%     |
| PENN       | 55.5%     | 86.8%†    | 86.8%†         | 72.1%       | 75.3%     |
| BasicPitch | 78.6%     | 71.1%     | 54.2%          | 74.8%       | 69.7%     |

### Cents Error (↓ lower is better)

| Algorithm  | NSynth     | PTDB      | MDB‑STEM‑Synth | SpeechSynth | Average    |
| ---------- | ---------- | --------- | -------------- | ----------- | ---------- |
| YAAPT      | 1045.09    | 105.95    | 366.13         | 63.18       | 395.09     |
| Praat      | 580.77     | 75.12     | **80.45**      | 69.11       | **201.36** |
| SWIPE      | 916.74     | 131.11    | 119.05         | 86.28       | 313.29     |
| RAPT       | 720.48     | 118.70    | 97.72          | 75.65       | 253.14     |
| pYIN       | 648.25     | 80.08     | 127.57         | 47.83       | 225.93     |
| TorchCREPE | **101.93** | **71.23** | 14.34†         | **36.24**   | 55.94      |
| PENN       | 1364.99    | 55.55†    | 146.70†        | 74.56       | 410.45     |
| BasicPitch | 660.44     | 101.42    | 266.40         | 53.84       | 270.52     |

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