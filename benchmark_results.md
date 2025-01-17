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

| Algorithm | Precision ↑ | Recall ↑ | F1 ↑ |
|-----------|------------|-----------|-------|
| YAAPT | 100.0 ± 0.0 % | 64.0 ± 27.0% | 73.9 ± 25.7% |
| Praat | 100.0 ± 0.0 % | 51.0 ± 32.4% | 60.4 ± 33.3% |
| SWIPE | 100.0 ± 0.0 % | 70.3 ± 29.0% | 78.1 ± 26.0% |
| RAPT | 100.0 ± 0.0 % | 60.9 ± 31.3% | 69.8 ± 31.0% |
| pYIN | 100.0 ± 0.0 % | 79.1 ± 22.5% | 86.0 ± 18.9% |
| PENN | **100.0 ± 0.0 %** | **100.0 ± 0.0 %** | **100.0 ± 0.0 %** |

## Pitch Accuracy

| Algorithm | RMSE (Hz) ↓ | Cents Error (Δ¢) ↓ | RPA ↑ | RCA ↑ |
|-----------|------------|-------------------|--------|--------|
| YAAPT | 484.61 ± 842.05 | 1621.25 ± 1899.23 | 35.2 ± 45.1% | 61.0 ± 43.2% |
| Praat | **205.90 ± 525.36** | **499.38 ± 1030.22** | **71.4 ± 43.3%** | **88.6 ± 27.7%** |
| SWIPE | 395.19 ± 558.86 | 953.01 ± 1392.07 | 47.5 ± 44.2% | 62.8 ± 39.5% |
| RAPT | 410.89 ± 834.45 | 1166.45 ± 1751.65 | 56.4 ± 46.2% | 67.1 ± 40.6% |
| pYIN | 267.18 ± 606.86 | 685.57 ± 1212.59 | 63.3 ± 43.7% | 76.9 ± 35.0% |
| PENN | 435.75 ± 751.75 | 1221.15 ± 1331.19 | 42.6 ± 37.6% | 54.6 ± 34.4% |

## Overall

| Algorithm | Harmonic Mean ↑ |
|-----------|----------------|
| YAAPT | 56.8 ± 30.4% |
| Praat | 72.8 ± 28.0% |
| SWIPE | 65.3 ± 30.1% |
| RAPT | 67.7 ± 29.9% |
| pYIN | **77.8 ± 27.6%** |
| PENN | 64.7 ± 31.2% |

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