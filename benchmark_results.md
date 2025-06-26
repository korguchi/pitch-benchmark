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

## NSynth (test split)

```bash
python pitch_benchmark.py --dataset NSynth --data-dir nsynth-test --fmin 65 --fmax 2093 --snr 10 --noise-dir "chime_home"
```

### Voicing Detection

| Algorithm  | Precision ↑ | Recall ↑ | F1 ↑  |
| ---------- | ----------- | -------- | ----- |
| PENN       | 71.7%       | **100.0%**   | **83.5%** |
| Praat      | 85.0%       | 79.8%    | 82.3% |
| RAPT       | 80.8%       | 83.0%    | 81.9% |
| SWIPE      | 71.4%       | 88.7%    | 79.1% |
| TorchCREPE | **96.1%**       | 47.3%    | 63.4% |
| YAAPT      | 89.7%       | 80.2%    | 84.7% |
| pYIN       | 74.1%       | 98.9%    | 84.7% |
| BasicPitch | 71.7%       | **100.0%**   | **83.5%** |

### Pitch Accuracy

| Algorithm  | RMSE (Hz) ↓ | Cents Err (Δ¢) ↓ | RPA ↑ | RCA ↑ |
| ---------- | ----------- | ---------------- | ----- | ----- |
| PENN       | 485.04      | 1366.60          | 35.8% | 49.9% |
| Praat      | 392.14      | 676.52           | 68.5% | 81.3% |
| RAPT       | 374.29      | 712.21           | 58.4% | 71.0% |
| SWIPE      | 402.02      | 928.38           | 48.5% | 59.2% |
| TorchCREPE | **104.58**      | **66.19**            | **94.7%** | **98.9%** |
| YAAPT      | 441.04      | 1070.03          | 39.4% | 68.9% |
| pYIN       | 298.70      | 650.47           | 59.7% | 70.0% |
| BasicPitch | 277.38      | 657.18           | 68.2% | 81.6% |

### Pitch Robustness

| Algorithm  | Octave Err % ↓ | Gross Err % ↓ |
| ---------- | -------------- | ------------- |
| PENN       | 55.4%          | 59.7%         |
| Praat      | 26.6%          | 28.0%         |
| RAPT       | 34.4%          | 36.8%         |
| SWIPE      | 35.8%          | 40.4%         |
| TorchCREPE | **4.5%**           | **4.8%**          |
| YAAPT      | 54.9%          | 57.9%         |
| pYIN       | 30.3%          | 34.9%         |
| BasicPitch | 29.9%          | 31.2%         |

### Overall

| Algorithm  | Harmonic Mean ↑ |
| ---------- | --------------- |
| PENN       | 2.0%            |
| Praat      | 22.9%           |
| RAPT       | 13.1%           |
| SWIPE      | 11.0%           |
| TorchCREPE | **73.1%**           |
| YAAPT      | 2.2%            |
| pYIN       | 17.5%           |
| BasicPitch | 18.6%           |

## Pitch Tracking Database (PTDB)

```bash
python pitch_benchmark.py --dataset PTDB --data-dir "SPEECH DATA" --fmin 65 --fmax 300 --snr 10 --noise-dir "chime_home"
```

Note: PENN was trained on the PTDB dataset, but here we evaluate it on the full PTDB dataset without excluding its training samples. This causes data leakage and inflates PENN’s results. The other algorithms were not trained on PTDB, so their results are unaffected.

### Voicing Detection

| Algorithm  | Precision ↑ | Recall ↑ | F1 ↑  |
| ---------- | ----------- | -------- | ----- |
| PENN       | **80.7%**       | 85.7%    | **83.1%** |
| Praat      | 56.0%       | 85.5%    | 67.6% |
| RAPT       | 68.8%       | 78.3%    | 73.3% |
| SWIPE      | 67.4%       | 71.0%    | 69.2% |
| TorchCREPE | 59.4%       | 80.7%    | 68.4% |
| YAAPT      | 69.4%       | **93.6%**    | 79.7% |
| pYIN       | 39.6%       | 78.6%    | 52.6% |
| BasicPitch | 67.4%       | 72.1%    | 69.7% |

### Pitch Accuracy

| Algorithm  | RMSE (Hz) ↓ | Cents Err (Δ¢) ↓ | RPA ↑ | RCA ↑ |
| ---------- | ----------- | ---------------- | ----- | ----- |
| PENN       | **16.36**       | **54.91**            | **89.8%** | **91.2%** |
| Praat      | 22.35       | 84.93            | 74.4% | 76.2% |
| RAPT       | 24.86       | 123.36           | 60.5% | 61.8% |
| SWIPE      | 28.35       | 131.57           | 72.9% | 73.9% |
| TorchCREPE | 18.63       | 71.34            | 77.7% | 79.1% |
| YAAPT      | 21.62       | 105.78           | 74.7% | 76.3% |
| pYIN       | 20.12       | 80.61            | 76.7% | 77.5% |
| BasicPitch | 22.11       | 100.98           | 71.8% | 73.0% |

### Pitch Robustness

| Algorithm  | Octave Err % ↓ | Gross Err % ↓ |
| ---------- | -------------- | ------------- |
| PENN       | **2.6%**           | **3.2%**          |
| Praat      | 3.6%           | 5.1%          |
| RAPT       | 4.4%           | 11.4%         |
| SWIPE      | 4.5%           | 14.4%         |
| TorchCREPE | 2.9%           | 3.7%          |
| YAAPT      | 4.7%           | 8.6%          |
| pYIN       | 3.0%           | 5.2%          |
| BasicPitch | 4.0%           | 6.2%          |

### Overall

| Algorithm  | Harmonic Mean ↑ |
| ---------- | --------------- |
| PENN       | **84.4%**           |
| Praat      | 73.1%           |
| RAPT       | 66.8%           |
| SWIPE      | 65.3%           |
| TorchCREPE | 76.0%           |
| YAAPT      | 73.0%           |
| pYIN       | 67.2%           |
| BasicPitch | 71.9%           |

## MDB-STEM-Synth

```bash
python pitch_benchmark.py --dataset MDBStemSynth --data-dir MDB-stem-synth --fmin 65 --fmax 2093 --snr 10 --noise-dir "chime_home"
```

Note: As with the PTDB dataset, there may be some leakage affecting PENN's (and also TorchCREPE's) results here.

### Voicing Detection

| Algorithm  | Precision ↑ | Recall ↑ | F1 ↑  |
| ---------- | ----------- | -------- | ----- |
| PENN       | 97.7%       | 75.4%    | **85.1%** |
| Praat      | 74.6%       | 85.9%    | 79.9% |
| RAPT       | 93.6%       | 72.4%    | 81.6% |
| SWIPE      | 81.5%       | 49.9%    | 61.9% |
| TorchCREPE | **98.5%**       | 48.8%    | 65.2% |
| YAAPT      | 71.4%       | 91.5%    | 80.2% |
| pYIN       | 67.4%       | **92.0%**    | 77.8% |
| BasicPitch | 69.3%       | 75.2%    | 72.1% |

### Pitch Accuracy

| Algorithm  | RMSE (Hz) ↓ | Cents Err (Δ¢) ↓ | RPA ↑ | RCA ↑ |
| ---------- | ----------- | ---------------- | ----- | ----- |
| PENN       | 28.21       | 143.47           | 84.2% | 94.2% |
| Praat      | 251.98      | 272.85           | 82.9% | 87.4% |
| RAPT       | 96.63       | 103.54           | 89.0% | 92.4% |
| SWIPE      | 110.43      | 113.72           | 87.3% | 88.1% |
| TorchCREPE | **29.53**       | **15.65**            | **98.8%** | **99.1%** |
| YAAPT      | 172.71      | 408.38           | 62.4% | 78.9% |
| pYIN       | 69.46       | 126.36           | 79.7% | 83.8% |
| BasicPitch | 101.65      | 337.27           | 41.0% | 44.2% |

### Pitch Robustness

| Algorithm  | Octave Err % ↓ | Gross Err % ↓ |
| ---------- | -------------- | ------------- |
| PENN       | 10.7%          | 11.0%         |
| Praat      | 9.8%           | 10.9%         |
| RAPT       | 5.7%           | 6.9%          |
| SWIPE      | 4.4%           | 8.2%          |
| TorchCREPE | **0.4%**           | **0.5%**          |
| YAAPT      | 26.3%          | 30.5%         |
| pYIN       | 7.3%           | 11.0%         |
| BasicPitch | 15.8%          | 41.0%         |

### Overall

| Algorithm  | Harmonic Mean ↑ |
| ---------- | --------------- |
| PENN       | 63.0%           |
| Praat      | 61.1%           |
| RAPT       | 75.2%           |
| SWIPE      | 69.0%           |
| TorchCREPE | **83.6%**           |
| YAAPT      | 24.1%           |
| pYIN       | 67.3%           |
| BasicPitch | 30.4%           |

## SpeechSynth

```bash
python pitch_benchmark.py --dataset SpeechSynth --data-dir lightspeech_new.pt --fmin 65 --fmax 300 --snr 10 --noise-dir "chime_home"
```

### Voicing Detection

| Algorithm  | Precision ↑ | Recall ↑ | F1 ↑  |
| ---------- | ----------- | -------- | ----- |
| PENN       | 82.8%       | 62.7%    | 71.3% |
| Praat      | 73.8%       | 92.0%    | 81.9% |
| RAPT       | **82.9%**       | 86.9%    | 84.8% |
| SWIPE      | 78.1%       | 88.9%    | 83.1% |
| TorchCREPE | 75.5%       | 91.4%    | 82.7% |
| YAAPT      | 78.9%       | **95.8%**    | **86.5%** |
| pYIN       | 67.3%       | 89.5%    | 76.8% |
| BasicPitch | 72.8%       | 93.7%    | 81.9% |

### Pitch Accuracy

| Algorithm  | RMSE (Hz) ↓ | Cents Err (Δ¢) ↓ | RPA ↑ | RCA ↑ |
| ---------- | ----------- | ---------------- | ----- | ----- |
| PENN       | 21.16       | 73.46            | 72.4% | 74.0% |
| Praat      | 25.62       | 75.19            | 78.2% | **80.5%** |
| RAPT       | 26.97       | 77.54            | 71.0% | 73.2% |
| SWIPE      | 34.15       | 90.93            | 74.6% | 74.9% |
| TorchCREPE | **8.68**        | **39.43**            | 76.8% | 76.8% |
| YAAPT      | 19.07       | 61.42            | **78.6%** | 79.1% |
| pYIN       | 16.75       | 56.47            | 74.9% | 75.5% |
| BasicPitch | 13.24       | 54.02            | 68.6% | 68.8% |

### Pitch Robustness

| Algorithm  | Octave Err % ↓ | Gross Err % ↓ |
| ---------- | -------------- | ------------- |
| PENN       | 2.8%           | 3.5%          |
| Praat      | 3.3%           | 4.8%          |
| RAPT       | 3.2%           | 4.6%          |
| SWIPE      | 2.4%           | 3.0%          |
| TorchCREPE | **0.5%**           | **0.9%**          |
| YAAPT      | 1.7%           | 4.1%          |
| pYIN       | 1.4%           | 2.9%          |
| BasicPitch | 0.9%           | 2.3%          |

### Overall

| Algorithm  | Harmonic Mean ↑ |
| ---------- | --------------- |
| PENN       | 76.4%           |
| Praat      | 79.5%           |
| RAPT       | 79.3%           |
| SWIPE      | 81.4%           |
| TorchCREPE | **87.0%**           |
| YAAPT      | 84.3%           |
| pYIN       | 81.5%           |
| BasicPitch | 83.0%           |

## Speed Benchmark

```bash
python speed_benchmark.py
```

Processing time for 5-second audio samples (averaged over 20 runs, how much faster than baseline TorchCREPE):

### Runtime Performance

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