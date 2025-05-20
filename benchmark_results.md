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

The overall performance of each algorithm is evaluated using a combined metric based on the harmonic mean of multiple key measurements. This approach was chosen specifically because:

1. **Balanced Performance Requirement**: The harmonic mean requires good performance across ALL metrics - an algorithm must perform well in both voicing detection and pitch accuracy to achieve a high score.

2. **Penalty for Poor Performance**: Poor performance in any single metric significantly impacts the final score, preventing algorithms from achieving high overall scores through excellence in just one area.

The combined metric incorporates four key measurements, all normalized to the range [0,1] where 1 is best:
- Voicing Detection Precision
- Voicing Detection Recall
- Raw Pitch Accuracy (RPA)
- Raw Chroma Accuracy (RCA)

# NSynth (test split)

```bash
python pitch_benchmark.py --dataset NSynth --data-dir nsynth-test --fmin 65 --fmax 2093
```

## Voicing Detection

| Algorithm  | Precision ↑  | Recall ↑          | F1 ↑         |
| ---------- | ------------ | ----------------- | ------------ |
| YAAPT      | 92.1 ± 14.8% | 86.3 ± 26.5%      | 84.9 ± 23.2% |
| Praat      | **99.9 ± 0.6 %** | 71.4 ± 33.5%      | 77.2 ± 31.8% |
| SWIPE      | 78.2 ± 26.7% | 76.8 ± 37.2%      | 72.6 ± 34.0% |
| RAPT       | 94.6 ± 9.0 % | 84.7 ± 31.5%      | 84.0 ± 28.7% |
| pYIN       | 83.6 ± 18.6% | 98.0 ± 11.2%      | **88.3 ± 16.2%** |
| TorchCREPE | 73.2 ± 35.3% | 66.0 ± 45.9%      | 62.2 ± 43.0% |
| PENN       | 70.2 ± 24.0% | **100.0 ± 0.0 %** | 79.4 ± 22.0% |

## Pitch Accuracy

| Algorithm  | RMSE (Hz) ↓         | Cents Error (Δ¢) ↓   | RPA ↑            | RCA ↑            |
| ---------- | ------------------- | -------------------- | ---------------- | ---------------- |
| YAAPT      | 481.97 ± 843.07     | 1607.24 ± 1903.65    | 35.8 ± 45.7%     | 61.7 ± 43.5%     |
| Praat      | **205.85 ± 525.38** | **499.35 ± 1030.27** | **71.4 ± 43.3%** | **88.7 ± 27.7%** |
| SWIPE      | 331.30 ± 656.28     | 993.88 ± 1379.31     | 50.4 ± 47.2%     | 65.6 ± 42.9%     |
| RAPT       | 405.91 ± 835.13     | 1150.29 ± 1749.96    | 57.3 ± 46.8%     | 67.9 ± 41.0%     |
| pYIN       | 219.08 ± 583.29     | 564.86 ± 1150.01     | 67.0 ± 44.6%     | 81.2 ± 34.7%     |
| TorchCREPE | 165.49 ± 457.58     | 310.54 ± 1074.70     | 83.8 ± 36.3%     | 89.3 ± 30.3%     |
| PENN       | 366.39 ± 747.75     | 976.37 ± 1395.49     | 55.6 ± 45.5%     | 69.0 ± 39.4%     |

Note: TorchCREPE was trained on the NSynth dataset.

## Overall

| Algorithm  | Harmonic Mean ↑  |
| ---------- | ---------------- |
| YAAPT      | 60.1 ± 31.9%     |
| Praat      | **81.2 ± 26.4%** |
| SWIPE      | 65.7 ± 29.5%     |
| RAPT       | 73.3 ± 30.7%     |
| pYIN       | 81.0 ± 26.9%     |
| TorchCREPE | 77.0 ± 27.7%     |
| PENN       | 70.5 ± 29.8%     |

# Pitch Tracking Database (PTDB)

```bash
python pitch_benchmark.py --dataset PTDB --data-dir SPEECH\ DATA/ --fmin 65 --fmax 300
```

## Voicing Detection

| Algorithm | Precision ↑ | Recall ↑ | F1 ↑ |
|-----------|------------|-----------|-------|
| YAAPT | **81.3 ± 11.3%** | **89.4 ± 8.0%** | **84.6 ± 8.6%** |
| Praat | 81.1 ± 10.3% | 86.0 ± 8.8% | 82.9 ± 8.1% |
| SWIPE | 80.9 ± 11.0% | 82.4 ± 12.8% | 80.6 ± 9.2% |
| RAPT | 78.2 ± 10.4% | 80.2 ± 10.5% | 78.6 ± 9.0% |
| pYIN | 80.4 ± 12.4% | 87.6 ± 13.3% | 82.7 ± 10.8% |
| TorchCREPE | 78.4 ± 11.4% | 82.4 ± 11.7% | 79.4 ± 9.0% |

## Pitch Accuracy

| Algorithm | RMSE (Hz) ↓ | Cents Error (Δ¢) ↓ | RPA ↑ | RCA ↑ |
|-----------|------------|-------------------|--------|--------|
| YAAPT | 16.93 ± 13.21 | 103.48 ± 94.63 | 73.3 ± 13.3% | 75.1 ± 13.0% |
| Praat | **15.60 ± 14.48** | **86.76 ± 87.91** | 73.2 ± 10.9% | 74.9 ± 10.5% |
| SWIPE | 19.84 ± 18.22 | 99.67 ± 104.20 | 77.0 ± 11.4% | 78.2 ± 10.8% |
| RAPT | 17.57 ± 14.53 | 109.97 ± 86.87 | 58.2 ± 13.4% | 59.5 ± 13.4% |
| pYIN | 23.52 ± 14.24 | 131.40 ± 97.35 | **79.1 ± 9.5%** | **80.3 ± 8.9%** |
| TorchCREPE | 15.93 ± 13.96 | 89.17 ± 92.68 | 75.5 ± 11.2% | 76.9 ± 10.9% |

## Overall

| Algorithm | Harmonic Mean ↑ |
|-----------|----------------|
| YAAPT | 79.3 ± 6.4% |
| Praat | 78.5 ± 5.3% |
| SWIPE | 79.6 ± 5.8% |
| RAPT | 67.5 ± 7.3% |
| **pYIN** | **81.7 ± 5.5%** |
| TorchCREPE | 78.2 ± 5.8% |

# MDB-STEM-Synth

```bash
python pitch_benchmark.py --dataset MDBStemSynth --data-dir MDB-stem-synth --fmin 65 --fmax 2093
```

## Voicing Detection

| Algorithm | Precision ↑ | Recall ↑ | F1 ↑ |
|-----------|------------|-----------|-------|
| YAAPT | 97.5 ± 2.9% | **96.0 ± 8.0%** | **96.5 ± 4.8%** |
| Praat | 99.0 ± 1.5% | 88.3 ± 14.7% | 92.5 ± 10.2% |
| SWIPE | **99.4 ± 0.7%** | 84.9 ± 19.5% | 90.2 ± 13.5% |
| RAPT | 99.1 ± 1.1% | 92.7 ± 13.5% | 95.1 ± 9.3% |
| pYIN | 97.0 ± 3.3% | 95.3 ± 7.6% | 96.0 ± 5.3% |

## Pitch Accuracy

| Algorithm | RMSE (Hz) ↓ | Cents Error (Δ¢) ↓ | RPA ↑ | RCA ↑ |
|-----------|------------|-------------------|--------|--------|
| YAAPT | 107.28 ± 153.99 | 392.33 ± 477.18 | 65.4 ± 26.6% | 81.0 ± 13.3% |
| Praat | 60.14 ± 136.81 | 105.75 ± 248.92 | **88.8 ± 10.8%** | **91.7 ± 7.3%** |
| SWIPE | 145.04 ± 185.68 | 181.01 ± 401.97 | 88.2 ± 17.7% | 89.9 ± 15.0% |
| RAPT | **45.87 ± 84.72** | **152.39 ± 257.53** | 86.2 ± 19.6% | 89.7 ± 14.5% |
| pYIN | 66.44 ± 169.06 | 168.22 ± 337.44 | 78.4 ± 17.7% | 82.7 ± 13.0% |

## Overall

| Algorithm | Harmonic Mean ↑ |
|-----------|----------------|
| YAAPT | 82.8 ± 15.5% |
| Praat | **91.7 ± 4.9%** |
| SWIPE | 90.3 ± 7.8% |
| RAPT | **91.7 ± 7.0%** |
| pYIN | 87.6 ± 7.1% |

# Speed Benchmark

```bash
python speed_benchmark.py
```

Processing time for 5-second audio samples (averaged over 20 runs, how much faster than baseline TorchCREPE):

| Algorithm   | CPU     | CUDA   |
|------------|---------|--------|
| YAAPT       | 12.31x | -  |
| Praat      | **449.43x** | -      |
| TorchCREPE | 1.00x   | 1.00x  |
| PENN       | 6.17x   | **5.49x**  |
| SWIPE      | 29.08x  | -      |
| RAPT       | 274.51x | -      |
| pYIN       | 3.55x   | -      |