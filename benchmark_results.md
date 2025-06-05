# Evaluation Metrics

## Voicing Detection Metrics

- Precision: Accuracy of voiced frame detection
- Recall: Proportion of actual voiced frames detected
- F1: Harmonic mean of precision and recall

## Pitch Accuracy Metrics

- RMSE: Root Mean Square Error in Hz (lower is better)
- Cents Error (Δ¢): Pitch deviation in musical cents (lower is better)
- RPA: Raw Pitch Accuracy (higher is better)
- RCA: Raw Chroma Accuracy (higher is better)

## Combined

The overall performance of each algorithm is evaluated using a combined metric based on the harmonic mean of multiple measurements. This approach was chosen specifically because:

1. **Balanced Performance Requirement**: The harmonic mean requires good performance across ALL metrics: an algorithm must perform well in both voicing detection and pitch accuracy to achieve a high score.

2. **Penalty for Poor Performance**: Poor performance in any single metric significantly impacts the final score, preventing algorithms from achieving high overall scores through excellence in just one area.

The combined metric incorporates four key measurements, all normalized to the range [0,1] where 1 is best:
- Voicing Detection Precision
- Voicing Detection Recall
- Raw Pitch Accuracy (RPA)
- Raw Chroma Accuracy (RCA)

# NSynth (test split)

```bash
python pitch_benchmark.py --dataset NSynth --data-dir nsynth-test --fmin 65 --fmax 2093 --snr 10 --noise-dir "chime_home"
```

## Voicing Detection

| Algorithm    | Precision ↑ | Recall ↑   | F1 ↑      |
|--------------|-------------|------------|-----------|
| BasicPitch   | 71.7%       | **100.0%** | 83.5%     |
| PENN         | 71.7%       | **100.0%** | 83.5%     |
| Praat        | 87.7%       | 74.3%      | 80.5%     |
| pYIN         | 74.0%       | 98.9%      | **84.7%** |
| RAPT         | 80.3%       | 83.3%      | 81.8%     |
| SWIPE        | 71.4%       | 88.6%      | 79.1%     |
| TorchCREPE   | **93.4%**   | 48.7%      | 64.0%     |
| YAAPT        | 91.8%       | 78.2%      | 84.4%     |

## Pitch Accuracy

| Algorithm    | RMSE (Hz) ↓ | Cents Error (Δ¢) ↓ | RPA ↑     | RCA ↑     |
|--------------|-------------|--------------------|-----------|-----------|
| BasicPitch   | 278.20      | 660.44             | 68.2%     | 81.5%     |
| PENN         | 483.76      | 1364.99            | 35.7%     | 49.8%     |
| Praat        | 364.75      | 580.77             | 71.9%     | 84.4%     |
| pYIN         | 295.70      | 648.25             | 59.6%     | 70.1%     |
| RAPT         | 373.19      | 720.48             | 58.6%     | 70.8%     |
| SWIPE        | 398.53      | 916.74             | 48.9%     | 59.5%     |
| TorchCREPE   | **192.30**  | **101.93**         | **93.5%** | **97.7%** |
| YAAPT        | 440.75      | 1045.09            | 40.9%     | 69.3%     |

## Overall

| Algorithm    | Harmonic Mean ↑ |
|--------------|-----------------|
| BasicPitch   | 78.6%           |
| PENN         | 55.5%           |
| Praat        | **79.0%**       |
| pYIN         | 73.2%           |
| RAPT         | 71.9%           |
| SWIPE        | 63.9%           |
| TorchCREPE   | 76.7%           |
| YAAPT        | 64.0%           |

# Pitch Tracking Database (PTDB)

```bash
python pitch_benchmark.py --dataset PTDB --data-dir "SPEECH DATA" --fmin 65 --fmax 300 --snr 10 --noise-dir "chime_home"
```

Note: PENN was trained on the PTDB dataset, but here we evaluate it on the full PTDB dataset without excluding its training samples. This causes data leakage and inflates PENN’s results. The other algorithms were not trained on PTDB, so their results are unaffected.

## Voicing Detection

| Algorithm    | Precision ↑ | Recall ↑ | F1 ↑     |
|--------------|-------------|----------|----------|
| BasicPitch   | 67.9%       | 72.2%    | 70.0%    |
| PENN         | 81.2%       | 85.7%    | 83.4%    |
| Praat        | 64.6%       | 79.4%    | 71.2%    |
| pYIN         | 39.8%       | 78.6%    | 52.9%    |
| RAPT         | **74.6%**   | 72.9%    | 73.7%    |
| SWIPE        | 67.8%       | 71.0%    | 69.3%    |
| TorchCREPE   | 59.6%       | 80.7%    | 68.6%    |
| YAAPT        | 69.4%       | **93.6%**| **79.7%**|

## Pitch Accuracy

| Algorithm    | RMSE (Hz) ↓ | Cents Err (Δ¢) ↓ | RPA ↑     | RCA ↑     |
|--------------|-------------|------------------|-----------|-----------|
| BasicPitch   | 22.19       | 101.42           | 71.7%     | 72.9%     |
| PENN         | 16.51       | 55.55            | 89.7%     | 91.1%     |
| Praat        | 20.60       | 75.12            | 76.9%     | 78.7%     |
| pYIN         | 20.04       | 80.08            | 76.8%     | 77.6%     |
| RAPT         | 24.61       | 118.70           | 62.2%     | 63.5%     |
| SWIPE        | 28.29       | 131.11           | 73.0%     | 74.0%     |
| TorchCREPE   | **18.54**   | **71.23**        | **77.7%** | **79.1%** |
| YAAPT        | 21.65       | 105.95           | 74.8%     | 76.4%     |

## Overall

| Algorithm    | Harmonic Mean ↑ |
|--------------|-----------------|
| BasicPitch   | 71.1%           |
| PENN         | 86.8%           |
| Praat        | 74.4%           |
| pYIN         | 62.8%           |
| RAPT         | 67.9%           |
| SWIPE        | 71.4%           |
| TorchCREPE   | 73.2%           |
| YAAPT        | **77.6%**       |

# MDB-STEM-Synth

```bash
python pitch_benchmark.py --dataset MDBStemSynth --data-dir MDB-stem-synth --fmin 65 --fmax 2093 --snr 10 --noise-dir "chime_home"
```

Note: As with the PTDB dataset, there may be some leakage affecting PENN's results here.

## Voicing Detection

| Algorithm    | Precision ↑ | Recall ↑  | F1 ↑      |
|--------------|-------------|-----------|-----------|
| BasicPitch   | 70.0%       | 66.6%     | 68.3%     |
| PENN         | 97.9%       | 74.7%     | 84.8%     |
| Praat        | 94.7%       | 57.9%     | 71.9%     |
| pYIN         | 65.8%       | 93.2%     | 77.1%     |
| RAPT         | 93.2%       | 72.4%     | 81.5%     |
| SWIPE        | 79.4%       | 64.2%     | 71.0%     |
| TorchCREPE   | **98.8%**   | 48.6%     | 65.1%     |
| YAAPT        | 80.0%       | **91.1%** | **85.2%** |

## Pitch Accuracy

| Algorithm    | RMSE (Hz) ↓ | Cents Error (Δ¢) ↓ | RPA ↑     | RCA ↑     |
|--------------|-------------|--------------------|-----------|-----------|
| BasicPitch   | 86.38       | 266.40             | 43.9%     | 46.2%     |
| PENN         | 28.36       | 146.70             | 84.1%     | 94.3%     |
| Praat        | 134.66      | 80.45              | 94.1%     | 96.0%     |
| pYIN         | 70.73       | 127.57             | 79.5%     | 83.6%     |
| RAPT         | 95.40       | 97.72              | 89.3%     | 92.5%     |
| SWIPE        | 110.08      | 119.05             | 88.0%     | 89.2%     |
| TorchCREPE   | **24.88**   | **14.34**          | **99.1%** | **99.3%** |
| YAAPT        | 164.92      | 366.13             | 64.5%     | 80.2%     |

## Overall

| Algorithm    | Harmonic Mean ↑ |
|--------------|-----------------|
| BasicPitch   | 54.2%           |
| PENN         | 86.8%           |
| Praat        | 81.8%           |
| pYIN         | 79.3%           |
| RAPT         | **85.9%**       |
| SWIPE        | 78.8%           |
| TorchCREPE   | 78.7%           |
| YAAPT        | 77.7%           |

# SpeechSynth

```bash
python pitch_benchmark.py --dataset SpeechSynth --data-dir lightspeech_new.pt --fmin 65 --fmax 300 --snr 10 --noise-dir "chime_home"
```

## Voicing Detection

| Algorithm    | Precision ↑ | Recall ↑  | F1 ↑      |
|--------------|-------------|-----------|-----------|
| BasicPitch   | 73.0%       | 93.8%     | 82.1%     |
| PENN         | **82.7%**   | 61.7%     | 70.7%     |
| Praat        | 78.6%       | 88.6%     | 83.3%     |
| pYIN         | 67.1%       | 87.7%     | 76.1%     |
| RAPT         | **82.7%**   | 86.2%     | 84.4%     |
| SWIPE        | 80.6%       | 85.8%     | 83.1%     |
| TorchCREPE   | 82.4%       | 83.8%     | 83.1%     |
| YAAPT        | 79.7%       | **95.4%** | **86.8%** |

## Pitch Accuracy

| Algorithm    | RMSE (Hz) ↓ | Cents Error (Δ¢) ↓ | RPA ↑     | RCA ↑     |
|--------------|-------------|--------------------|-----------|-----------|
| BasicPitch   | 13.17       | 53.84              | 68.6%     | 68.8%     |
| PENN         | 21.58       | 74.56              | 72.7%     | 74.4%     |
| Praat        | 24.57       | 69.11              | **80.0%** | **82.2%** |
| pYIN         | 14.00       | 47.83              | 75.4%     | 75.9%     |
| RAPT         | 26.36       | 75.65              | 71.4%     | 73.5%     |
| SWIPE        | 33.30       | 86.28              | 75.6%     | 75.8%     |
| TorchCREPE   | **7.93**    | **36.24**          | 78.6%     | 78.6%     |
| YAAPT        | 18.96       | 63.18              | 78.1%     | 78.6%     |

## Combined Score

| Algorithm    | Harmonic Mean ↑ |
|--------------|-----------------|
| BasicPitch   | 74.8%           |
| PENN         | 72.1%           |
| Praat        | 82.2%           |
| pYIN         | 75.9%           |
| RAPT         | 78.0%           |
| SWIPE        | 79.2%           |
| TorchCREPE   | 80.8%           |
| YAAPT        | **82.4%**       |

# Speed Benchmark

```bash
python speed_benchmark.py
```

Processing time for 5-second audio samples (averaged over 20 runs, how much faster than baseline TorchCREPE):

## Runtime Performance

| Algorithm    | CPU               | CUDA              |
|--------------|-------------------|-------------------|
| BasicPitch   | 7.18x (320.8ms)   | CPU only          |
| PENN         | 4.59x (501.6ms)   | **5.65x (57.0ms)**|
| Praat        | **469.16x (4.9ms)**| CPU only          |
| pYIN         | 1.28x (1796.6ms)  | CPU only          |
| RAPT         | 147.61x (15.6ms)  | CPU only          |
| SWIPE        | 14.61x (157.6ms)  | CPU only          |
| TorchCREPE   | 1.00x (2088.6ms)  | 1.00x (242.9ms)   |
| YAAPT        | 7.10x (324.5ms)   | CPU only          |