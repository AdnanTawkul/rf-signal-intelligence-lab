# Baseline CNN v1 Results

## Overview

Baseline CNN v1 is a compact one-dimensional convolutional neural network trained directly on synthetic raw IQ sequences.

The model input shape is:

```text
[batch, 2, 2048]
```

Channel 0 contains the in-phase component. Channel 1 contains the quadrature component.

This report preserves the complete experimental progression:

1. Original BatchNorm baseline
2. Five-seed reproducibility study
3. RMS-normalization ablation
4. GroupNorm ablation and promotion
5. BatchNorm-versus-GroupNorm class-by-SNR diagnosis
6. Frozen-embedding and classifier-head diagnosis
7. Validation-selected frozen linear-head refit
8. Final five-seed supervised baseline promotion

## Dataset

The dataset contains four modulation classes:

- BPSK
- QPSK
- 8PSK
- 16QAM

Dataset size:

| Split | Examples |
|---|---:|
| Training | 5,600 |
| Validation | 1,400 |
| Test | 1,400 |
| Total | 8,400 |

Each split is balanced by modulation class and SNR.

Evaluated SNR levels:

```text
-4, 0, 4, 8, 12, 16, 20 dB
```

Synthetic examples include:

- Root-raised-cosine pulse shaping
- AWGN
- Carrier frequency offset
- Carrier phase offset
- Amplitude variation
- Integer time shift
- Limited flat Rayleigh fading

## Original Model

The original baseline classifier contains three one-dimensional convolutional blocks:

```text
2 → 32 → 64 → 128 channels
```

Each original block contains:

- One-dimensional convolution
- Batch normalization
- GELU activation
- Max pooling

Global average pooling and a linear layer produce four class logits.

Trainable parameters:

```text
73,092
```

The later selected encoder retains the same topology and parameter count but replaces BatchNorm with GroupNorm using eight groups.

## Original Training Configuration

| Setting | Value |
|---|---:|
| Epochs | 30 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Optimizer | AdamW |
| Best epoch, seed 2026 | 24 |
| Best validation accuracy, seed 2026 | 94.3% |

Validation performance fluctuated substantially during original BatchNorm training. This indicated sensitivity to optimization or BatchNorm statistics and motivated the later normalization-layer study.

## Original Single-Seed Held-Out Test Results

Overall test accuracy:

```text
94.14%
```

Per-class accuracy:

| Modulation | Accuracy |
|---|---:|
| BPSK | 100.00% |
| QPSK | 85.71% |
| 8PSK | 94.86% |
| 16QAM | 96.00% |

Accuracy by SNR:

| SNR | Accuracy |
|---:|---:|
| -4 dB | 70.50% |
| 0 dB | 93.50% |
| 4 dB | 99.00% |
| 8 dB | 99.00% |
| 12 dB | 99.00% |
| 16 dB | 98.00% |
| 20 dB | 100.00% |

## Original Baseline Figures

### Confusion Matrix

![Baseline CNN v1 confusion matrix](figures/baseline_cnn_v1_confusion_matrix.png)

### Accuracy by SNR

![Baseline CNN v1 accuracy by SNR](figures/baseline_cnn_v1_accuracy_by_snr.png)

### Class-by-SNR Error Analysis

![Baseline CNN v1 class-by-SNR accuracy](figures/baseline_cnn_v1_class_snr_accuracy.png)

The original class-by-SNR breakdown identified the dominant failure mode:

| Modulation | Accuracy at -4 dB |
|---|---:|
| BPSK | 100.0% |
| QPSK | 40.0% |
| 8PSK | 66.0% |
| 16QAM | 76.0% |

QPSK at -4 dB was the worst original class-SNR group.

## Five-Seed BatchNorm Reproducibility Study

The original architecture was trained independently using five random seeds:

```text
2026, 2027, 2028, 2029, 2030
```

For every run, the checkpoint with the highest validation accuracy was evaluated on the same untouched test split.

### Aggregate Held-Out Test Accuracy

| Metric | Result |
|---|---:|
| Mean test accuracy | 94.24% |
| Standard deviation | 0.29 percentage points |
| Minimum | 93.93% |
| Maximum | 94.71% |

Individual test accuracies:

| Seed | Test accuracy |
|---:|---:|
| 2026 | 94.14% |
| 2027 | 94.71% |
| 2028 | 94.00% |
| 2029 | 94.43% |
| 2030 | 93.93% |

### Mean Per-Class Accuracy

| Modulation | Mean accuracy | Standard deviation |
|---|---:|---:|
| BPSK | 99.77% | 0.33 percentage points |
| QPSK | 87.77% | 2.32 percentage points |
| 8PSK | 93.09% | 3.15 percentage points |
| 16QAM | 96.34% | 1.72 percentage points |

### Mean Accuracy by SNR

| SNR | Mean accuracy | Standard deviation |
|---:|---:|---:|
| -4 dB | 70.50% | 1.10 percentage points |
| 0 dB | 94.60% | 1.07 percentage points |
| 4 dB | 99.10% | 0.37 percentage points |
| 8 dB | 98.50% | 0.45 percentage points |
| 12 dB | 99.30% | 0.24 percentage points |
| 16 dB | 97.90% | 0.20 percentage points |
| 20 dB | 99.80% | 0.24 percentage points |

Best validation accuracy was stable across seeds:

```text
94.17% ± 0.31 percentage points
```

Final-epoch validation accuracy was substantially less stable:

```text
88.39% ± 4.13 percentage points
```

The defensible original BatchNorm claim became:

```text
94.24% ± 0.29 percentage points held-out test accuracy across five independent training seeds
```

## RMS Normalization Ablation

A controlled ablation tested per-example complex RMS normalization while keeping the dataset, architecture, optimizer, batch size, epoch count, and training seeds unchanged.

The transform scales every IQ example to unit average complex power while preserving relative constellation geometry.

### Single-Seed Result

| Metric | Original | RMS-normalized | Change |
|---|---:|---:|---:|
| Overall test accuracy | 94.14% | 94.79% | +0.65 percentage points |
| QPSK accuracy | 85.71% | 86.57% | +0.86 percentage points |
| 8PSK accuracy | 94.86% | 94.00% | -0.86 percentage points |
| 16QAM accuracy | 96.00% | 98.57% | +2.57 percentage points |
| Accuracy at -4 dB | 70.50% | 73.50% | +3.00 percentage points |
| Accuracy at 0 dB | 93.50% | 90.50% | -3.00 percentage points |

The single-run improvement was not sufficient evidence for promotion.

### Five-Seed Validation Comparison

| Metric | Original | RMS-normalized | Change |
|---|---:|---:|---:|
| Mean best validation accuracy | 94.17% | 93.76% | -0.41 percentage points |
| Best-validation standard deviation | 0.31 pp | 0.49 pp | Worse |
| Minimum best validation accuracy | 93.57% | 93.07% | Worse |
| Mean final validation accuracy | 88.39% | 84.94% | -3.45 percentage points |
| Final-validation standard deviation | 4.13 pp | 8.36 pp | Worse |

### Five-Seed Held-Out Test Comparison

| Metric | Original | RMS-normalized | Change |
|---|---:|---:|---:|
| Mean test accuracy | 94.24% | 93.93% | -0.31 percentage points |
| Test standard deviation | 0.29 pp | 0.80 pp | Worse |
| Minimum test accuracy | 93.93% | 92.43% | -1.50 percentage points |
| QPSK accuracy | 87.77% | 86.23% | -1.54 percentage points |
| 8PSK accuracy | 93.09% | 90.86% | -2.23 percentage points |
| 16QAM accuracy | 96.34% | 98.69% | +2.35 percentage points |
| Accuracy at -4 dB | 70.50% | 68.90% | -1.60 percentage points |
| Accuracy at 0 dB | 94.60% | 89.80% | -4.80 percentage points |

### RMS Ablation Decision

Per-example RMS normalization is rejected as the default preprocessing method. It reduced mean held-out accuracy, increased variance, lowered the worst-seed result, degraded QPSK and 8PSK, and made training instability worse.

## GroupNorm Ablation and Promotion

A controlled normalization-layer ablation replaced BatchNorm with GroupNorm while preserving the dataset, CNN topology, optimizer, learning rate, batch size, 30-epoch budget, five random seeds, and disabled RMS input normalization.

The GroupNorm configuration used eight groups in every convolutional block.

### Five-Seed Validation Comparison

| Metric | BatchNorm | GroupNorm | Change |
|---|---:|---:|---:|
| Mean best validation accuracy | 94.17% | 95.64% | +1.47 percentage points |
| Best-validation standard deviation | 0.31 pp | 0.40 pp | +0.09 pp |
| Minimum best validation accuracy | 93.57% | 95.29% | +1.72 percentage points |
| Maximum best validation accuracy | 94.50% | 96.43% | +1.93 percentage points |
| Mean final validation accuracy | 88.39% | 94.10% | +5.71 percentage points |
| Final-validation standard deviation | 4.13 pp | 2.34 pp | -1.79 pp |

Every GroupNorm run exceeded the maximum BatchNorm best-validation result. GroupNorm also improved final-epoch validation accuracy and reduced its variation substantially.

### Five-Seed Held-Out Test Comparison

| Metric | BatchNorm | GroupNorm | Change |
|---|---:|---:|---:|
| Mean test accuracy | 94.24% | 95.29% | +1.05 percentage points |
| Test standard deviation | 0.29 pp | 0.45 pp | +0.16 pp |
| Minimum test accuracy | 93.93% | 94.71% | +0.78 percentage points |
| Maximum test accuracy | 94.71% | 96.07% | +1.36 percentage points |
| BPSK accuracy | 99.77% | 99.89% | +0.12 percentage points |
| QPSK accuracy | 87.77% | 92.23% | +4.46 percentage points |
| 8PSK accuracy | 93.09% | 89.94% | -3.15 percentage points |
| 16QAM accuracy | 96.34% | 99.09% | +2.75 percentage points |
| Accuracy at -4 dB | 70.50% | 74.90% | +4.40 percentage points |
| Accuracy at 0 dB | 94.60% | 93.40% | -1.20 percentage points |

### Mean GroupNorm Accuracy by SNR

| SNR | Mean accuracy | Standard deviation |
|---:|---:|---:|
| -4 dB | 74.90% | 2.58 percentage points |
| 0 dB | 93.40% | 1.07 percentage points |
| 4 dB | 99.40% | 0.37 percentage points |
| 8 dB | 99.80% | 0.24 percentage points |
| 12 dB | 99.90% | 0.20 percentage points |
| 16 dB | 99.90% | 0.20 percentage points |
| 20 dB | 99.70% | 0.60 percentage points |

### Five-Seed GroupNorm Test Figure

![GroupNorm five-seed held-out evaluation](figures/baseline_cnn_groupnorm_seed_sweep_v1_test.png)

### Historical GroupNorm Decision

GroupNorm replaced BatchNorm as the selected supervised baseline because it improved mean and worst-seed held-out accuracy, QPSK, 16QAM, aggregate -4 dB performance, validation accuracy, and late-training stability.

Its pre-refit baseline claim was:

```text
95.29% ± 0.45 percentage points held-out test accuracy across five independent training seeds
```

## BatchNorm versus GroupNorm Class-by-SNR Diagnosis

All five BatchNorm and five GroupNorm checkpoints were reevaluated on the same held-out test split. Accuracy was aggregated jointly by true modulation class and SNR.

![Five-seed BatchNorm versus GroupNorm class-by-SNR comparison](figures/batchnorm_vs_groupnorm_class_snr_v1.png)

### 8PSK Accuracy by SNR

| SNR | BatchNorm | GroupNorm | Change |
|---:|---:|---:|---:|
| -4 dB | 62.0% | 45.2% | -16.8 percentage points |
| 0 dB | 96.8% | 86.8% | -10.0 percentage points |
| 4 dB | 98.8% | 98.4% | -0.4 percentage points |
| 8 dB | 96.0% | 99.6% | +3.6 percentage points |
| 12 dB | 100.0% | 100.0% | 0.0 percentage points |
| 16 dB | 98.8% | 99.6% | +0.8 percentage points |
| 20 dB | 99.2% | 100.0% | +0.8 percentage points |

The GroupNorm 8PSK regression was concentrated almost entirely at -4 dB and 0 dB. From 4 dB upward, GroupNorm was effectively equivalent to or slightly better than BatchNorm.

### Largest Class-by-SNR Changes

| Comparison | Class and SNR | Change |
|---|---|---:|
| Largest GroupNorm gain | QPSK at -4 dB | +20.4 percentage points |
| Largest GroupNorm loss | 8PSK at -4 dB | -16.8 percentage points |

The GroupNorm model shifted low-SNR performance toward QPSK and away from 8PSK. This suggested a changed phase-class decision boundary rather than a general loss of 8PSK representation quality.

## Frozen-Embedding Diagnosis

The GroupNorm encoder's pooled representation has shape:

```text
[batch, 128]
```

A logistic-regression probe trained on frozen low-SNR QPSK and 8PSK embeddings improved seed-2026 performance on the corresponding held-out subset:

| Metric | Original four-class head | Frozen binary probe | Change |
|---|---:|---:|---:|
| Overall low-SNR QPSK/8PSK accuracy | 74.5% | 81.0% | +6.5 pp |
| Accuracy at -4 dB | 59.0% | 67.0% | +8.0 pp |
| Accuracy at 0 dB | 90.0% | 95.0% | +5.0 pp |
| QPSK at -4 dB | 44.0% | 68.0% | +24.0 pp |
| 8PSK at -4 dB | 74.0% | 66.0% | -8.0 pp |

Across all five GroupNorm seeds, the low-SNR frozen binary probe improved validation in four of five seeds and test in all five seeds.

A non-oracle routing rule based on confidence margins was not robust enough for promotion. Helpful and harmful specialist overrides occupied overlapping confidence ranges, and the direction-aware router had approximately neutral mean validation change.

This evidence indicated that the encoder contained useful separability that the original learned four-class head did not fully exploit.

## Frozen Four-Class Linear Probe

A standardized four-class logistic-regression head was then fitted on the frozen 128-dimensional training embeddings for each seed.

### Diagnostic Five-Seed Results

| Split | Original GroupNorm head | Frozen linear probe | Change |
|---|---:|---:|---:|
| Validation mean accuracy | 95.64% | 96.16% | +0.51 pp |
| Test mean accuracy | 95.29% | 96.33% | +1.04 pp |

The frozen four-class probe improved validation and test accuracy for all five seeds. It particularly improved 8PSK, showing that the representation was stronger than the end-to-end classifier head suggested.

This result justified implementing a reproducible, validation-selected, native PyTorch head-refit pipeline.

## Deployable Frozen Linear-Head Refit

### Method

For each GroupNorm checkpoint:

1. Freeze the trained CNN encoder.
2. Extract 128-dimensional embeddings for the training and validation splits.
3. Fit standardized multinomial logistic-regression classifiers on training embeddings.
4. Evaluate candidate regularization values only on validation embeddings.
5. Select the best validation candidate, preferring the smaller `C` on ties.
6. Convert the standardized classifier into equivalent raw-embedding parameters.
7. Replace the model's existing `Linear(128, 4)` weight and bias.
8. Save a normal PyTorch checkpoint.
9. Evaluate the resulting checkpoint once on the untouched test split.

Candidate regularization values:

```text
0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0
```

For standardized features

```text
z = (x - mean) / scale
```

the conversion is:

```text
W_raw = W_standardized / scale
b_raw = b_standardized - W_raw @ mean
```

The deployed checkpoint therefore requires no scikit-learn dependency at inference time.

### Selected Regularization by Seed

| Seed | Selected C |
|---:|---:|
| 2026 | 1.0 |
| 2027 | 10.0 |
| 2028 | 0.1 |
| 2029 | 3.0 |
| 2030 | 1.0 |

### Five-Seed Validation Results

| Seed | Original GroupNorm | Refitted head | Change |
|---:|---:|---:|---:|
| 2026 | 96.43% | 96.64% | +0.21 pp |
| 2027 | 95.50% | 96.29% | +0.79 pp |
| 2028 | 95.43% | 96.14% | +0.71 pp |
| 2029 | 95.29% | 96.14% | +0.86 pp |
| 2030 | 95.57% | 96.07% | +0.50 pp |

Aggregate validation comparison:

| Metric | Original GroupNorm | Refitted head | Change |
|---|---:|---:|---:|
| Mean accuracy | 95.64% | 96.26% | +0.61 pp |
| Standard deviation | 0.40 pp | 0.21 pp | -0.19 pp |
| Seeds improved | — | 5/5 | — |

Regularization selection used validation data only. The test split was not accessed during fitting or model selection.

### Five-Seed Held-Out Test Results

Individual native-checkpoint results:

| Seed | Test accuracy |
|---:|---:|
| 2026 | 96.93% |
| 2027 | 96.43% |
| 2028 | 96.43% |
| 2029 | 96.57% |
| 2030 | 96.14% |

Aggregate comparison:

| Metric | Original GroupNorm | Refitted head | Change |
|---|---:|---:|---:|
| Mean test accuracy | 95.29% | 96.50% | +1.21 pp |
| Test standard deviation | 0.45 pp | 0.26 pp | -0.19 pp |
| Minimum test accuracy | 94.71% | 96.14% | +1.43 pp |
| Maximum test accuracy | 96.07% | 96.93% | +0.86 pp |
| BPSK accuracy | 99.89% | 99.94% | +0.05 pp |
| QPSK accuracy | 92.23% | 93.14% | +0.91 pp |
| 8PSK accuracy | 89.94% | 93.94% | +4.00 pp |
| 16QAM accuracy | 99.09% | 98.97% | -0.12 pp |
| Accuracy at -4 dB | 74.90% | 78.40% | +3.50 pp |
| Accuracy at 0 dB | 93.40% | 97.10% | +3.70 pp |

Every refitted checkpoint improved on its original GroupNorm checkpoint. The worst refitted test result, 96.14%, exceeded the best original GroupNorm result, 96.07%.

### Mean Refitted Accuracy by Class

| Modulation | Mean accuracy | Standard deviation |
|---|---:|---:|
| BPSK | 99.94% | 0.11 percentage points |
| QPSK | 93.14% | 1.02 percentage points |
| 8PSK | 93.94% | 0.42 percentage points |
| 16QAM | 98.97% | 0.29 percentage points |

### Mean Refitted Accuracy by SNR

| SNR | Mean accuracy | Standard deviation |
|---:|---:|---:|
| -4 dB | 78.40% | 1.39 percentage points |
| 0 dB | 97.10% | 0.73 percentage points |
| 4 dB | 100.00% | 0.00 percentage points |
| 8 dB | 100.00% | 0.00 percentage points |
| 12 dB | 100.00% | 0.00 percentage points |
| 16 dB | 100.00% | 0.00 percentage points |
| 20 dB | 100.00% | 0.00 percentage points |

### Five-Seed Refit Figure

![GroupNorm frozen-head-refit five-seed held-out evaluation](figures/baseline_cnn_groupnorm_head_refit_seed_sweep_v1_test.png)

### Final Promotion Decision

The GroupNorm CNN with a validation-selected frozen linear-head refit replaces the original end-to-end GroupNorm checkpoint as the selected supervised baseline.

The promotion is supported by:

1. Higher validation accuracy for all five seeds.
2. Higher held-out test accuracy for all five seeds.
3. Lower validation and test variation.
4. A higher worst-seed test result.
5. Stronger QPSK and substantially stronger 8PSK accuracy.
6. Better performance at both -4 dB and 0 dB.
7. Native PyTorch deployment with unchanged architecture and parameter count.
8. No use of test labels or test metrics during fitting or selection.

### Final Selected Supervised Baseline Claim

```text
96.50% ± 0.26 percentage points held-out test accuracy across five independent training seeds
```

This supersedes the pre-refit GroupNorm result:

```text
95.29% ± 0.45 percentage points
```

and the original BatchNorm result:

```text
94.24% ± 0.29 percentage points
```

## Current Limitations

1. The benchmark remains entirely synthetic.
2. Training, validation, and test examples use the same signal-generator family.
3. Real receiver and hardware-specific distortions are not represented fully.
4. No public real-world RF dataset has been evaluated yet.
5. Severe-noise phase-modulation discrimination remains the dominant error regime.
6. Confidence calibration and uncertainty have not been evaluated.
7. The selected head is linear; nonlinear head alternatives have not yet been compared under the same protocol.
8. No self-supervised representation-learning result is available yet.
9. Deployment latency and ONNX compatibility have not yet been benchmarked.

## Next Research Targets

The next experiments should remain evidence-driven:

1. Add low-SNR-aware sampling or curriculum experiments.
2. Evaluate confidence calibration and uncertainty.
3. Train a self-supervised encoder and apply the same linear-evaluation protocol.
4. Validate on a public RF modulation dataset.
5. Export the selected native checkpoint to ONNX.
6. Benchmark local CPU and GPU latency.
7. Build a local interactive demonstration.
