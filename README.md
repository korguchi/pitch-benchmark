# Pitch Detection Benchmark

A comprehensive benchmark suite for evaluating and comparing different pitch detection algorithms across multiple datasets and metrics.

## 📊 Key Findings

TorchCREPE achieves the highest average harmonic-mean accuracy (79.9%) across four datasets, but suffers from lower voicing recall and being the slowest algorithm. Praat delivers the best speed-accuracy balance. It processes audio faster than all competitors while maintaining the second-best overall accuracy (59.2%). For a detailed breakdown of results, see [Benchmark Results](benchmark_results.md).

In the tables below, any entry marked with “†” indicates that the algorithm was trained (or partially trained) on that same dataset, so its performance on that split may be inflated by data leakage and should not be directly compared to the other methods.

| Algorithm  | NSynth    | PTDB      | MDB‑STEM‑Synth | SpeechSynth | Average   |
| ---------- | --------- | --------- | -------------- | ----------- | --------- |
| YAAPT      | 2.2%      | 73.0%     | 24.1%          | 84.3%       | 45.9%     |
| Praat      | 22.9%     | 73.1%     | 61.1%          | 79.5%       | 59.2%     |
| SWIPE      | 11.0%     | 65.3%     | 69.0%          | 81.4%       | 56.7%     |
| RAPT       | 13.1%     | 66.8%     | 75.2%          | 79.3%       | 58.6%     |
| pYIN       | 17.5%     | 67.2%     | 67.3%          | 81.5%       | 58.4%     |
| TorchCREPE | **73.1%†**     | 76.0%     | **83.6%†**         | **87.0%**       | **79.9%**     |
| PENN       | 2.0%      | **84.4%†**    | 63.0%†         | 76.4%       | 56.5%     |
| BasicPitch | 18.6%     | 71.9%     | 30.4%          | 83.0%       | 51.0%     |

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