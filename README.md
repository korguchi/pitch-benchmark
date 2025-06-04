# Pitch Detection Benchmark

A comprehensive benchmark suite for evaluating and comparing different pitch detection algorithms across multiple datasets and metrics.

## 📊 Key Findings

Praat emerges as the best choice for pitch detection, offering high pitch accuracy with the lowest cents error across all datasets while also providing unmatched computational efficiency (449x faster than baseline). While pYIN shows higher harmonic mean scores in some datasets, Praat's significantly lower pitch error and dramatic speed advantage make it the clear winner for both accuracy and practical applications. For a detailed breakdown of results, see [Benchmark Results](benchmark_results.md).

### Pitch accuracy (↑ higher is better)

| Algorithm | NSynth | PTDB | MDB-STEM-Synth | Average |
|-----------|---------|---------|----------------|----------|
| YAAPT | 60.1% | 79.3% | 82.8% | 74.1% |
| Praat | **81.2%** | 78.5% | **91.7%** | **83.8%** |
| SWIPE | 65.7% | 79.6% | 90.3% | 78.5% |
| RAPT | 73.3% | 67.5% | **91.7%** | 77.5% |
| pYIN | 81.0% | **81.7%** | 87.6% | 83.4% |
| TorchCREPE | N/A* | 78.2% | N/A* | N/A* |
| PENN | 70.5% | N/A* | N/A* | N/A* |

*N/A indicates the algorithm was excluded from testing on this dataset as it was used in training

### Cents Error (↓ lower is better)

| Algorithm | NSynth | PTDB | MDB-STEM-Synth | Average |
|-----------|---------|---------|----------------|----------|
| YAAPT | 1607.24 | 103.48 | 392.33 | 701.02 |
| Praat | **499.35** | **86.76** | **105.75** | **230.62** |
| SWIPE | 993.88 | 99.67 | 181.01 | 424.85 |
| RAPT | 1150.29 | 109.97 | 152.39 | 470.88 |
| pYIN | 564.86 | 131.40 | 168.22 | 288.16 |
| TorchCREPE | N/A* | 89.17 | N/A* | N/A* |
| PENN | 976.37 | N/A* | N/A* | N/A* |

*N/A indicates the algorithm was excluded from testing on this dataset as it was used in training

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