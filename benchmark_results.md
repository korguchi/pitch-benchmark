# Evaluation Metrics

## Voicing Detection

How well the algorithm identifies voiced vs unvoiced speech segments.

**Precision:** `TP/(TP+FP)` - Percentage of predicted voiced frames that are actually voiced  
**Recall:** `TP/(TP+FN)` - Percentage of true voiced frames that are detected  
**F1 Score:** `2×(Precision×Recall)/(Precision+Recall)` - Balanced voicing performance

## Pitch Accuracy

How accurate the fundamental frequency estimates are for voiced segments.

**RMSE (Hz):** `√(mean((f_pred - f_true)²))` - Average pitch error in Hz  
**Cents Error:** `1200×log₂(f_pred/f_true)` - Pitch error in cents (perceptually meaningful)  
**Cents Accuracy:** `exp(-|cents_error|/500)` - Soft accuracy score (0-1)  
**RPA (Raw Pitch Accuracy):** Percentage of frames within 50 cents of ground truth  
**RCA (Raw Chroma Accuracy):** Like RPA but ignores octave errors (pitch class only)

## Error Robustness

Measures how often the algorithm makes catastrophic pitch errors.

**Octave Error Rate:** Percentage of frames with >600 cent errors (octave mistakes)  
**Octave Accuracy:** `exp(-10×octave_error_rate)` - Heavily penalizes octave jumps  
**Gross Error Rate:** Percentage of frames with >200 cent errors (general instability)  
**Gross Accuracy:** `exp(-5×gross_error_rate)` - Moderately penalizes large errors

## Combined Score

Single metric that balances all aspects using harmonic mean of 6 components: RPA, Cents Accuracy, Precision, Recall, Octave Accuracy, and Gross Accuracy.

**Formula:** `6/Σ(1/component)` if all components > 0, otherwise 0

The harmonic mean heavily penalizes algorithms that perform poorly in any single area. Error robustness metrics prevent high pitch accuracy scores when the algorithm frequently makes octave jumps, as a single octave error is far more disruptive than several small deviations.

# Benchmark results

## A note on data leakage

Non-neural network algorithms don't use training data, so data leakage isn't an issue for them. However, for neural networks, a validation dataset is always required, which can lead to leakage. Since there are no standardized dataset splits, we evaluated all algorithms using the full datasets. Despite this, we believe our results remain comparable for a few reasons:

- We added significant noise to all audio. This transforms the data enough so it's not identical to any training examples.
- Neural networks are usually trained with early stopping and audio chunks (not the entire audio file). This approach helps prevent them from simply memorizing the entire training set.
- We also did a qualitative analysis on real microphone audio clips from outside any training distribution. Models that performed better on the benchmark demonstrated greater accuracy on the real data.

Although not perfect, this is the best we can do without retraining the networks. Here's a breakdown of the datasets each algorithm was trained on:
- TorchCREPE: NSynth, MDB-STEM-Synth, MIR-1K
- PENN: PTDB, MDB-STEM-Synth
- SwiftF0: NSynth, PTDB, MDB-STEM-Synth, MIR-1K

## NSynth (test split)

```bash
python pitch_benchmark.py --dataset NSynth --data-dir nsynth-test --fmin 65 --fmax 2093 --snr 10 --noise-dir "chime_home"
```

### Voicing Detection

| Algorithm  | Precision ↑ | Recall ↑ | F1 ↑  |
| ---------- | ----------- | -------- | ----- |
| PENN       | 71.6%       | **99.6%**    | 83.3% |
| Praat      | 84.6%       | 78.5%    | 81.4% |
| RAPT       | 80.2%       | 82.5%    | 81.3% |
| SWIPE      | 71.6%       | 88.4%    | 79.1% |
| TorchCREPE | **96.0%**       | 47.8%    | 63.8% |
| YAAPT      | 89.5%       | 79.8%    | 84.4% |
| pYIN       | 73.7%       | 99.0%    | 84.5% |
| BasicPitch | 81.7%       | 98.1%    | **89.2%** |
| SwiftF0    | 78.1%       | 96.6%    | 86.4% |

### Pitch Accuracy

| Algorithm  | RMSE (Hz) ↓ | Cents Err (Δ¢) ↓ | RPA ↑ | RCA ↑ |
| ---------- | ----------- | ---------------- | ----- | ----- |
| PENN       | 481.72      | 1347.03          | 34.5% | 47.5% |
| Praat      | 376.19      | 670.86           | 68.1% | 80.7% |
| RAPT       | 372.32      | 761.98           | 58.6% | 71.0% |
| SWIPE      | 366.93      | 878.48           | 52.8% | 63.2% |
| TorchCREPE | **103.22**      | **66.16**            | **95.0%** | **99.2%** |
| YAAPT      | 446.94      | 1070.40          | 39.9% | 68.6% |
| pYIN       | 295.20      | 643.81           | 60.0% | 70.2% |
| BasicPitch | 313.41      | 802.61           | 60.2% | 72.6% |
| SwiftF0    | 283.38      | 477.03           | 69.6% | 78.0% |

### Pitch Robustness

| Algorithm  | Octave Err % ↓ | Gross Err % ↓ |
| ---------- | -------------- | ------------- |
| PENN       | 55.6%          | 62.0%         |
| Praat      | 26.9%          | 28.7%         |
| RAPT       | 34.2%          | 36.1%         |
| SWIPE      | 33.3%          | 37.6%         |
| TorchCREPE | **4.6%**           | **4.7%**          |
| YAAPT      | 54.4%          | 57.4%         |
| pYIN       | 30.2%          | 34.5%         |
| BasicPitch | 35.5%          | 38.6%         |
| SwiftF0    | 20.6%          | 26.1%         |

### Overall

| Algorithm  | Harmonic Mean ↑ |
| ---------- | --------------- |
| PENN       | 2.0%            |
| Praat      | 22.5%           |
| RAPT       | 13.2%           |
| SWIPE      | 13.4%           |
| TorchCREPE | **73.4%**           |
| YAAPT      | 2.3%            |
| pYIN       | 17.8%           |
| BasicPitch | 11.9%           |
| SwiftF0    | 33.6%           |

## Pitch Tracking Database (PTDB)

```bash
python pitch_benchmark.py --dataset PTDB --data-dir "SPEECH DATA" --fmin 65 --fmax 300 --snr 10 --noise-dir "chime_home"
```

### Voicing Detection

| Algorithm  | Precision ↑ | Recall ↑ | F1 ↑  |
| ---------- | ----------- | -------- | ----- |
| PENN       | 80.3%       | 88.2%    | **84.1%** |
| Praat      | 73.7%       | 75.2%    | 74.4% |
| RAPT       | 75.9%       | 86.0%    | 80.6% |
| SWIPE      | 70.0%       | 73.9%    | 71.9% |
| TorchCREPE | 62.2%       | 84.4%    | 71.6% |
| YAAPT      | 70.1%       | 94.5%    | 80.5% |
| pYIN       | **81.2%**       | 45.2%    | 58.1% |
| BasicPitch | 27.1%       | **99.8%**    | 42.7% |
| SwiftF0    | 77.7%       | 86.5%    | 81.9% |

### Pitch Accuracy

| Algorithm  | RMSE (Hz) ↓ | Cents Err (Δ¢) ↓ | RPA ↑ | RCA ↑ |
| ---------- | ----------- | ---------------- | ----- | ----- |
| PENN       | 17.89       | 65.85            | 88.4% | 89.5% |
| Praat      | 18.95       | 58.56            | 87.4% | 89.2% |
| RAPT       | 31.74       | 231.20           | 76.7% | 79.2% |
| SWIPE      | 33.56       | 196.77           | 80.3% | 90.5% |
| TorchCREPE | 24.71       | 131.36           | 55.5% | 56.9% |
| YAAPT      | 23.76       | 128.44           | 75.4% | 78.9% |
| pYIN       | 20.22       | 67.09            | 86.8% | 87.6% |
| BasicPitch | 56.87       | 472.66           | 10.7% | 12.7% |
| SwiftF0    | **12.12**       | **40.24**            | **90.6%** | **91.0%** |

### Pitch Robustness

| Algorithm  | Octave Err % ↓ | Gross Err % ↓ |
| ---------- | -------------- | ------------- |
| PENN       | 3.2%           | 4.5%          |
| Praat      | 2.8%           | 3.4%          |
| RAPT       | 5.3%           | 6.9%          |
| SWIPE      | 14.3%          | 15.1%         |
| TorchCREPE | 5.0%           | 8.8%          |
| YAAPT      | 7.5%           | 9.6%          |
| pYIN       | 2.8%           | 4.1%          |
| BasicPitch | 21.6%          | 61.5%         |
| SwiftF0    | **1.5%**           | **2.0%**          |

### Overall

| Algorithm  | Harmonic Mean ↑ |
| ---------- | --------------- |
| PENN       | 82.5%           |
| Praat      | 80.4%           |
| RAPT       | 70.7%           |
| SWIPE      | 50.8%           |
| TorchCREPE | 66.0%           |
| YAAPT      | 67.9%           |
| pYIN       | 72.3%           |
| BasicPitch | 12.8%           |
| SwiftF0    | **87.0%**           |

## MDB-STEM-Synth

```bash
python pitch_benchmark.py --dataset MDBStemSynth --data-dir MDB-stem-synth --fmin 65 --fmax 2093 --snr 10 --noise-dir "chime_home"
```

## Voicing Detection

| Algorithm     | Precision ↑ | Recall ↑ | F1 ↑   |
|---------------|-------------|-----------|--------|
| PENN          | **97.7%**       | 75.5%     | 85.2%  |
| Praat         | 78.3%       | 85.4%     | 81.7%  |
| RAPT          | 91.8%       | 72.5%     | 81.0%  |
| SWIPE         | 78.4%       | 57.9%     | 66.6%  |
| TorchCREPE    | 90.6%       | 42.5%     | 57.8%  |
| YAAPT         | 71.8%       | **91.9%**     | 80.6%  |
| pYIN          | 93.7%       | 56.7%     | 70.7%  |
| BasicPitch    | 59.4%       | 54.7%     | 57.0%  |
| SwiftF0       | 91.0%       | 80.3%     | **85.3%**  |

## Pitch Accuracy

| Algorithm     | RMSE (Hz) ↓ | Cents Err (Δ¢) ↓ | RPA ↑  | RCA ↑  |
|---------------|-------------|------------------|--------|--------|
| PENN          | **28.87**       | 150.80           | 82.7%  | 92.4%  |
| Praat         | 244.27      | 276.96           | 83.5%  | 88.9%  |
| RAPT          | 100.65      | 199.31           | 80.1%  | 83.7%  |
| SWIPE         | 87.53       | 157.28           | 83.7%  | 91.5%  |
| TorchCREPE    | 67.73       | 162.05           | 53.7%  | 54.3%  |
| YAAPT         | 170.43      | 397.82           | 62.4%  | 78.9%  |
| pYIN          | 64.36       | **32.56**            | **95.8%**  | **96.7%**  |
| BasicPitch    | 153.36      | 603.34           | 14.1%  | 18.2%  |
| SwiftF0       | 44.00       | 60.20            | 84.4%  | 85.4%  |

## Pitch Robustness

| Algorithm     | Octave Err % ↓ | Gross Err % ↓ |
|---------------|----------------|----------------|
| PENN          | 11.1%          | 12.0%          |
| Praat         | 10.8%          | 11.9%          |
| RAPT          | 6.1%           | 8.0%           |
| SWIPE         | 10.7%          | 11.3%          |
| TorchCREPE    | 5.0%           | 26.4%          |
| YAAPT         | 25.9%          | 29.9%          |
| pYIN          | **1.4%**           | **1.5%**           |
| BasicPitch    | 32.0%          | 71.4%          |
| SwiftF0       | **1.4%**           | 7.5%           |

## Overall

| Algorithm     | Harmonic Mean ↑ |
|---------------|------------------|
| PENN          | 61.4%            |
| Praat         | 59.1%            |
| RAPT          | 70.3%            |
| SWIPE         | 58.6%            |
| TorchCREPE    | 49.6%            |
| YAAPT         | 24.9%            |
| pYIN          | **83.6%**            |
| BasicPitch    | 8.1%             |
| SwiftF0       | 82.6%            |

## SpeechSynth

```bash
python pitch_benchmark.py --dataset SpeechSynth --data-dir datasets/lightspeech_new.pt --fmin 65 --fmax 300 --snr 10 --noise-dir "chime_home"
```

### Voicing Detection

| Algorithm  | Precision ↑ | Recall ↑ | F1 ↑  |
| ---------- | ----------- | -------- | ----- |
| PENN       | 82.1%       | 63.8%    | 71.8% |
| Praat      | 72.1%       | **93.2%**    | 81.3% |
| RAPT       | **84.6%**       | 85.8%    | **85.2%** |
| SWIPE      | 78.2%       | 88.6%    | 83.1% |
| TorchCREPE | 74.5%       | 90.7%    | 81.8% |
| YAAPT      | 79.5%       | 94.7%    | 86.5% |
| pYIN       | 50.3%       | 99.3%    | 66.8% |
| BasicPitch | 70.3%       | 78.1%    | 74.0% |
| SwiftF0    | 75.0%       | 92.7%    | 82.9% |

### Pitch Accuracy

| Algorithm  | RMSE (Hz) ↓ | Cents Err (Δ¢) ↓ | RPA ↑ | RCA ↑ |
| ---------- | ----------- | ---------------- | ----- | ----- |
| PENN       | 18.78       | 69.48            | 71.1% | 72.0% |
| Praat      | 26.38       | 85.06            | 71.9% | 74.1% |
| RAPT       | 47.06       | 292.58           | 63.5% | 66.0% |
| SWIPE      | 41.43       | 158.17           | 70.2% | 73.4% |
| TorchCREPE | 10.63       | 53.11            | 62.6% | 62.7% |
| YAAPT      | 23.46       | 85.12            | 73.7% | 75.9% |
| pYIN       | 39.96       | 205.30           | 68.0% | 69.8% |
| BasicPitch | 24.51       | 134.58           | 33.8% | 34.1% |
| SwiftF0    | **8.35**        | **34.20**            | **81.3%** | **81.4%** |

### Pitch Robustness

| Algorithm  | Octave Err % ↓ | Gross Err % ↓ |
| ---------- | -------------- | ------------- |
| PENN       | 1.8%           | 4.4%          |
| Praat      | 3.7%           | 5.4%          |
| RAPT       | 5.7%           | 7.1%          |
| SWIPE      | 8.0%           | 8.8%          |
| TorchCREPE | 0.5%           | 1.6%          |
| YAAPT      | 3.9%           | 5.1%          |
| pYIN       | 10.7%          | 12.3%         |
| BasicPitch | 2.0%           | 18.7%         |
| SwiftF0    | **0.2%**           | **0.5%**          |

### Overall

| Algorithm  | Harmonic Mean ↑ |
| ---------- | --------------- |
| PENN       | 77.0%           |
| Praat      | 77.0%           |
| RAPT       | 67.3%           |
| SWIPE      | 66.8%           |
| TorchCREPE | 82.4%           |
| YAAPT      | 78.7%           |
| pYIN       | 55.8%           |
| BasicPitch | 55.9%           |
| SwiftF0    | **88.7%**           |

## MIR-1K

### Voicing Detection

| Algorithm  | Precision ↑ | Recall ↑ | F1 ↑  |
| ---------- | ----------- | -------- | ----- |
| PENN       | **97.1%**       | 58.8%    | 73.2% |
| Praat      | 83.8%       | 96.0%    | 89.5% |
| RAPT       | 93.1%       | 95.5%    | 94.3% |
| SWIPE      | 93.7%       | 88.3%    | 90.9% |
| TorchCREPE | 90.2%       | 39.5%    | 54.9% |
| YAAPT      | 92.7%       | **97.0%**    | **94.8%** |
| pYIN       | **97.1%**       | 77.0%    | 85.9% |
| BasicPitch | 72.4%       | 88.5%    | 79.6% |
| SwiftF0    | 95.8%       | 89.2%    | 92.4% |

### Pitch Accuracy

| Algorithm  | RMSE (Hz) ↓ | Cents Err (Δ¢) ↓ | RPA ↑ | RCA ↑ |
| ---------- | ----------- | ---------------- | ----- | ----- |
| PENN       | 23.67       | 59.88            | 90.0% | 91.1% |
| Praat      | 115.27      | 130.27           | 86.2% | 89.7% |
| RAPT       | 42.87       | 200.91           | 79.4% | 81.2% |
| SWIPE      | 49.83       | 111.05           | 86.3% | 91.0% |
| TorchCREPE | 24.82       | 58.48            | 77.5% | 77.6% |
| YAAPT      | 49.35       | 141.17           | 80.1% | 85.0% |
| pYIN       | **19.07**       | 35.59            | **94.2%** | **94.6%** |
| BasicPitch | 125.29      | 368.71           | 22.8% | 23.9% |
| SwiftF0    | 20.13       | **27.95**            | **94.2%** | 94.3% |

### Pitch Robustness

| Algorithm  | Octave Err % ↓ | Gross Err % ↓ |
| ---------- | -------------- | ------------- |
| PENN       | 2.4%           | 3.9%          |
| Praat      | 6.4%           | 8.2%          |
| RAPT       | 4.6%           | 6.4%          |
| SWIPE      | 7.2%           | 8.0%          |
| TorchCREPE | 1.5%           | 3.3%          |
| YAAPT      | 8.1%           | 10.0%         |
| pYIN       | 1.4%           | 1.9%          |
| BasicPitch | 14.3%          | 46.5%         |
| SwiftF0    | **0.8%**           | **1.3%**          |

### Overall

| Algorithm  | Harmonic Mean ↑ |
| ---------- | --------------- |
| PENN       | 80.4%           |
| Praat      | 74.1%           |
| RAPT       | 76.5%           |
| SWIPE      | 73.6%           |
| TorchCREPE | 71.4%           |
| YAAPT      | 70.0%           |
| pYIN       | 89.4%           |
| BasicPitch | 25.7%           |
| SwiftF0    | **93.3%**           |

## Speed Benchmark

```bash
python speed_benchmark.py
```

Processing time for 5-second audio samples (averaged over 20 runs, how much faster than baseline TorchCREPE):

### Runtime Performance

| Algorithm  | CPU               | CUDA             |
| ---------- | ----------------- | ---------------- |
| PENN       | 6.12× (919.0 ms)  | **4.65× (69.4 ms)**  |
| Praat      | **808.99× (7.0 ms)**  | CPU only         |
| RAPT       | 414.66× (13.6 ms) | CPU only         |
| SWIPE      | 42.21× (133.3 ms) | CPU only         |
| TorchCREPE | 1.00× (5508.3 ms) | 1.00× (194.1 ms) |
| YAAPT      | 17.66× (318.5 ms) | CPU only         |
| pYIN       | 3.96× (1420.6 ms) | CPU only         |
| BasicPitch | 20.07× (280.3 ms) | CPU only         |
| SwiftF0    | 42.42× (132.6 ms) | CPU only         |