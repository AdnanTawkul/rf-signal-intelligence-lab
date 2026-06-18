# Channel Robustness Evaluation v1

## Summary

This report evaluates the selected supervised RF modulation-classification baseline under paired frequency-selective multipath conditions.

The evaluated model family is the selected GroupNorm CNN with the frozen linear-head refit workflow. Five independently trained checkpoints were tested on four paired channel conditions:

- clean baseline
- mild multipath
- moderate multipath
- severe multipath

All conditions use the same labels, SNR values, nuisance parameters, example seeds, class balance, and training-seed checkpoints. Only the multipath condition changes.

The main result is that the selected baseline is highly accurate on the clean test split but is not robust to moderate or severe frequency-selective multipath.

## Evaluation protocol

### Model checkpoints

Five selected supervised checkpoints were evaluated:

- seed 2026
- seed 2027
- seed 2028
- seed 2029
- seed 2030

### Test conditions

| Condition | Multipath profile | Description |
|---|---|---|
| Clean | None | Original held-out baseline test split |
| Mild | Mild tapped-delay profile | Short delay spread and relatively weak delayed paths |
| Moderate | Moderate tapped-delay profile | Larger delay spread and stronger delayed paths |
| Severe | Severe tapped-delay profile | Long delay spread and strong frequency-selective distortion |

### Pairing controls

The test datasets were generated with matched metadata. Across all four conditions, the following values remain identical:

- class labels
- SNR values
- carrier-frequency offsets
- carrier-phase offsets
- amplitude scales
- time shifts
- flat-Rayleigh flags
- example seeds

This makes the clean-to-multipath comparison paired at the example level.

## Overall accuracy

| Condition | Mean accuracy | Standard deviation | Paired drop from clean |
|---|---:|---:|---:|
| Clean | 0.9650 | 0.0026 | 0.0000 |
| Mild | 0.8804 | 0.0063 | 0.0846 |
| Moderate | 0.6701 | 0.0136 | 0.2949 |
| Severe | 0.4096 | 0.0085 | 0.5554 |

The degradation is consistent across all five checkpoints.

Mild multipath reduces mean accuracy by about 8.5 percentage points. Moderate multipath reduces it by about 29.5 percentage points. Severe multipath reduces it by about 55.5 percentage points.

## Per-class accuracy

| Class | Clean | Mild | Moderate | Severe |
|---|---:|---:|---:|---:|
| BPSK | 0.9994 | 0.9389 | 0.7977 | 0.4371 |
| QPSK | 0.9314 | 0.7354 | 0.4303 | 0.0680 |
| 8PSK | 0.9394 | 0.8611 | 0.4571 | 0.1349 |
| 16QAM | 0.9897 | 0.9863 | 0.9954 | 0.9983 |

The failure is strongly class-dependent.

- BPSK degrades gradually and remains partially recognizable.
- QPSK collapses under severe multipath.
- 8PSK also collapses under severe multipath.
- 16QAM remains almost perfectly classified under every tested multipath condition.

## Confusion analysis

The pooled five-seed confusion matrices reveal that moderate and severe multipath cause PSK examples to move toward the 16QAM decision region.

### Dominant moderate-multipath errors

| True class | Dominant predicted class | Error rate |
|---|---|---:|
| BPSK | 8PSK | 0.0994 |
| QPSK | 16QAM | 0.4451 |
| 8PSK | 16QAM | 0.5086 |
| 16QAM | QPSK | 0.0040 |

### Dominant severe-multipath errors

| True class | Dominant predicted class | Error rate |
|---|---|---:|
| BPSK | 16QAM | 0.4640 |
| QPSK | 16QAM | 0.8771 |
| 8PSK | 16QAM | 0.8491 |
| 16QAM | QPSK | 0.0011 |

The failure is therefore not limited to confusion between neighboring PSK modulation orders. Under strong frequency-selective multipath, the model increasingly interprets distorted constant-envelope PSK waveforms as 16QAM.

## Accuracy by SNR

The clean baseline reaches perfect mean accuracy from 4 dB upward.

Mild multipath still benefits from higher SNR, but accuracy saturates below the clean result.

Under moderate and severe multipath, the accuracy curve becomes comparatively flat across SNR. This indicates that additive noise is no longer the dominant limitation. Increasing SNR cannot remove inter-symbol interference and waveform distortion introduced by delayed paths.

## Interpretation

The selected CNN appears to rely on waveform characteristics that are stable under the original synthetic channel but are not invariant to frequency-selective distortion.

The asymmetric confusion pattern suggests that multipath creates amplitude-envelope and dispersion patterns that resemble features the classifier associates with 16QAM. Because 16QAM itself remains highly recognizable, the model has not lost all discriminative capability. Instead, its learned representation becomes biased toward the 16QAM region for distorted PSK signals.

## Engineering conclusion

The selected supervised baseline is reliable for the original held-out channel distribution but is not deployable as a general multipath-robust RF classifier without additional mitigation.

The next development phase should evaluate at least one of the following:

1. supervised training with randomized multipath augmentation;
2. explicit channel equalization before classification;
3. an architecture or representation designed to reduce sensitivity to channel-induced amplitude and phase distortion;
4. paired robustness evaluation after mitigation using the same clean, mild, moderate, and severe protocol.

The paired multipath datasets, per-seed prediction artifacts, consolidated robustness metrics, and pooled confusion matrices now provide a reproducible benchmark for measuring improvement.

## Reproducible artifacts

### Dataset configurations

- `configs/dataset_multipath_mild_v1.yaml`
- `configs/dataset_multipath_moderate_v1.yaml`
- `configs/dataset_multipath_severe_v1.yaml`

### Evaluation configurations

- `configs/evaluate_groupnorm_head_refit_multipath_mild_seed_sweep_v1.yaml`
- `configs/evaluate_groupnorm_head_refit_multipath_moderate_seed_sweep_v1.yaml`
- `configs/evaluate_groupnorm_head_refit_multipath_severe_seed_sweep_v1.yaml`
- `configs/compare_channel_robustness_v1.yaml`
- `configs/compare_channel_confusions_v1.yaml`

### Main outputs

- `results/channel_robustness_v1/summary.json`
- `results/channel_confusion_robustness_v1/summary.json`
- `reports/figures/channel_robustness_v1.png`
- `reports/figures/channel_confusion_robustness_v1.png`

## Final result

The channel-robustness milestone establishes a clear baseline:

- clean mean accuracy: **96.50%**
- mild multipath mean accuracy: **88.04%**
- moderate multipath mean accuracy: **67.01%**
- severe multipath mean accuracy: **40.96%**

The most important observed failure mode is the collapse of QPSK and 8PSK predictions toward 16QAM under moderate and severe multipath.
