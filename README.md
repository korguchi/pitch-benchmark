# Pitch Detection Benchmark

A comprehensive benchmark suite for evaluating and comparing different pitch detection algorithms across multiple datasets and metrics.

## 📊 Key Findings

Praat emerges as the best choice for pitch detection, offering high pitch accuracy with the lowest cents error across all datasets while also providing unmatched computational efficiency (449x faster than baseline). While pYIN shows higher harmonic mean scores in some datasets, Praat's significantly lower pitch error and dramatic speed advantage make it the clear winner for both accuracy and practical applications. For a detailed breakdown of results, see [Benchmark Results](benchmark_results.md).

### Pitch accuracy (↑ higher is better)

| Algorithm | NSynth | PTDB | MDB-STEM-Synth | Average |
|-----------|---------|---------|----------------|----------|
| YAAPT | 56.8% | 79.3% | 82.8% | 73.0% |
| Praat | 72.8% | 78.5% | **91.7%** | 81.0% |
| SWIPE | 65.3% | 79.6% | 90.3% | 78.4% |
| RAPT | 67.7% | 67.5% | **91.7%** | 75.6% |
| pYIN | **77.8%** | **81.7%** | 87.6% | **82.4%** |
| TorchCREPE | N/A* | 78.2% | N/A* | N/A* |
| PENN | 64.7% | N/A* | N/A* | N/A* |

*N/A indicates the algorithm was excluded from testing on this dataset as it was used in training

### Cents Error (↓ lower is better)

| Algorithm | NSynth | PTDB | MDB-STEM-Synth | Average |
|-----------|---------|---------|----------------|----------|
| YAAPT | 1621.25 | 103.48 | 392.33 | 705.69 |
| Praat | **499.38** | **86.76** | **105.75** | **230.63** |
| SWIPE | 953.01 | 99.67 | 181.01 | 411.23 |
| RAPT | 1166.45 | 109.97 | 152.39 | 476.27 |
| pYIN | 685.57 | 131.40 | 168.22 | 328.40 |
| TorchCREPE | N/A* | 89.17 | N/A* | N/A* |
| PENN | 1221.15 | N/A* | N/A* | N/A* |

*N/A indicates the algorithm was excluded from testing on this dataset as it was used in training

## 🚀 Quick Start

### Installation

```bash
pip install torchcrepe crepe praat-parselmouth pysptk librosa penn==0.0.14 AMFM-decompy pandas
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
- Visualization tools for algorithm comparison
- Implementation of popular pitch detection algorithms:
  - [YAAPT](https://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html) (pYAAPT implementation)
  - [Praat](https://github.com/YannickJadoul/Parselmouth) (Parselmouth implementation)
  - [TorchCREPE](https://github.com/maxrmorrison/torchcrepe) (PyTorch implementation of CREPE) and [CREPE](https://github.com/marl/crepe) (original implementation)
  - [Pitch-Estimating Neural Networks (PENN)](https://github.com/interactiveaudiolab/penn)
  - [SWIPE](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.swipe.html) (SPTK implementation)
  - [RAPT](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.rapt.html) (SPTK implementation)
  - [pYIN](https://librosa.org/doc/main/generated/librosa.pyin.html) (librosa implementation)

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