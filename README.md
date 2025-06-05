# Pitch Detection Benchmark

A comprehensive benchmark suite for evaluating and comparing different pitch detection algorithms across multiple datasets and metrics.

## 📊 Key Findings

Praat demonstrates the best overall performance, achieving the highest average harmonic-mean accuracy (78.4%) across three datasets and maintaining a 245-cent average error while processing audio faster than all the other algorithms. While TorchCREPE achieves lower pitch errors on individual datasets, it is much slower and has lower recall. For a detailed breakdown of results, see [Benchmark Results](benchmark_results.md).

In the tables below, any entry marked with “†” indicates that the algorithm was trained (or partially trained) on that same dataset, so its performance on that split is inflated by data leakage and should not be directly compared to the other methods.

### Pitch accuracy (↑ higher is better)

| Algorithm  | NSynth    | PTDB      | MDB‑STEM‑Synth | Average   |
| ---------- | --------- | --------- | -------------- | --------- |
| YAAPT      | 64.0%     | 77.6%     | 77.7%          | 73.1%     |
| Praat      | **79.0%** | 74.4%     | 81.8%          | **78.4%** |
| SWIPE      | 63.9%     | 71.4%     | 78.8%          | 71.4%     |
| RAPT       | 71.9%     | 67.9%     | **85.9%**      | 75.2%     |
| pYIN       | 73.2%     | 62.8%     | 79.3%          | 71.8%     |
| TorchCREPE | 76.7%     | 73.2%     | 78.7%†         | 76.2%     |
| PENN       | 55.5%     | 86.8%†    | 86.8%†         | 76.4%     |

### Cents Error (↓ lower is better)

| Algorithm   | NSynth   | PTDB    | MDB-STEM-Synth | Average  |
|-------------|----------|---------|----------------|----------|
| YAAPT       | 1045.09  | 105.95  | 366.13         | 505.72   |
| Praat       | 580.77   | 75.12   | **80.45**   | **245.45**   |
| SWIPE       | 916.74   | 131.11  | 119.05         | 388.97   |
| RAPT        | 720.48   | 118.70  | 97.72          | 312.30   |
| pYIN        | 648.25   | 80.08   | 127.57         | 285.30   |
| TorchCREPE  | **101.93** | **71.23**  | 14.34†        | 62.50    |
| PENN        | 1364.99  | 55.55† | 146.70†          | 522.41   |

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

- Comprehensive evaluation across standard datasets:
  - [PTDB](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html)
  - [NSynth](https://magenta.tensorflow.org/datasets/nsynth)
  - [MDB-stem-synth](https://zenodo.org/records/1481172)
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