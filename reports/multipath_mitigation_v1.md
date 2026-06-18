# Multipath Mitigation Evaluation v1

## Summary

This report evaluates supervised mixed-multipath training as a mitigation strategy for frequency-selective channel distortion in the RF modulation-classification baseline.

The mitigation uses the same GroupNorm CNN architecture and five training seeds as the original clean-trained baseline. The only controlled change is the training distribution:

- 25% clean examples
- 25% mild multipath examples
- 25% moderate multipath examples
- 25% severe multipath examples

Validation remains clean. Final evaluation uses the same paired clean, mild, moderate, and severe held-out test datasets used in the channel-robustness baseline.

The mitigation successfully improves multipath robustness, especially under moderate and severe channels, while causing a modest reduction in clean-channel accuracy.

## Experimental protocol

### Model architecture

- Baseline CNN with GroupNorm
- 73,092 trainable parameters
- Random initialization
- 30 supervised epochs
- Five training seeds: 2026, 2027, 2028, 2029, 2030

### Mixed training distribution

Each training example deterministically receives one of the following channel conditions:

| Training condition | Probability |
|---|---:|
| Clean | 0.25 |
| Mild multipath | 0.25 |
| Moderate multipath | 0.25 |
| Severe multipath | 0.25 |

The profile selection is derived from each example seed. Labels, SNR values, frequency offsets, phase offsets, amplitude scales, time shifts, Rayleigh flags, and example seeds remain paired with the original clean training dataset.

### Evaluation conditions

The original clean-trained and mixed-multipath-trained model families are both evaluated on:

- clean held-out test split
- mild multipath held-out test split
- moderate multipath held-out test split
- severe multipath held-out test split

Each condition contains 1,400 examples and is evaluated across the same five training seeds.

## Training outcome

The mixed-multipath model retained strong clean validation performance.

| Statistic | Value |
|---|---:|
| Mean best validation accuracy | 0.9260 |
| Best-validation standard deviation | 0.0070 |
| Best-validation range | 0.9121–0.9314 |
| Mean final validation accuracy | 0.9130 |
| Final-validation standard deviation | 0.0227 |
| Best seed | 2029 |

The lower training accuracy relative to clean validation is expected because the training split contains moderate and severe multipath examples, while validation remains clean.

## Overall held-out test results

| Test condition | Original clean-trained model | Mixed-multipath-trained model | Paired change |
|---|---:|---:|---:|
| Clean | 0.9650 ± 0.0026 | 0.9334 ± 0.0071 | -0.0316 ± 0.0059 |
| Mild | 0.8804 ± 0.0063 | 0.9010 ± 0.0102 | +0.0206 ± 0.0127 |
| Moderate | 0.6701 ± 0.0136 | 0.7893 ± 0.0190 | +0.1191 ± 0.0288 |
| Severe | 0.4096 ± 0.0085 | 0.5664 ± 0.0298 | +0.1569 ± 0.0303 |

The mitigation trades 3.16 percentage points of clean accuracy for:

- 2.06 percentage points improvement under mild multipath
- 11.91 percentage points improvement under moderate multipath
- 15.69 percentage points improvement under severe multipath

All five paired seeds improve under mild, moderate, and severe conditions.

## Per-class effects

### Clean condition

| Class | Original | Mitigated | Change |
|---|---:|---:|---:|
| BPSK | 0.9994 | 0.9983 | -0.0011 |
| QPSK | 0.9314 | 0.8549 | -0.0766 |
| 8PSK | 0.9394 | 0.8966 | -0.0429 |
| 16QAM | 0.9897 | 0.9840 | -0.0057 |

Most of the clean-channel tradeoff comes from reduced QPSK and 8PSK accuracy.

### Mild multipath

| Class | Original | Mitigated | Change |
|---|---:|---:|---:|
| BPSK | 0.9389 | 0.9926 | +0.0537 |
| QPSK | 0.7354 | 0.7880 | +0.0526 |
| 8PSK | 0.8611 | 0.8463 | -0.0149 |
| 16QAM | 0.9863 | 0.9771 | -0.0091 |

### Moderate multipath

| Class | Original | Mitigated | Change |
|---|---:|---:|---:|
| BPSK | 0.7977 | 0.9811 | +0.1834 |
| QPSK | 0.4303 | 0.6143 | +0.1840 |
| 8PSK | 0.4571 | 0.5989 | +0.1417 |
| 16QAM | 0.9954 | 0.9629 | -0.0326 |

### Severe multipath

| Class | Original | Mitigated | Change |
|---|---:|---:|---:|
| BPSK | 0.4371 | 0.8297 | +0.3926 |
| QPSK | 0.0680 | 0.2440 | +0.1760 |
| 8PSK | 0.1349 | 0.2257 | +0.0909 |
| 16QAM | 0.9983 | 0.9663 | -0.0320 |

The strongest gain is BPSK under severe multipath, improving by 39.26 percentage points.

QPSK and 8PSK also improve, but they remain the principal unresolved failure classes under severe multipath.

## Reduction of the PSK-to-16QAM collapse

The original robustness evaluation showed that frequency-selective multipath caused PSK examples to collapse toward the 16QAM decision region.

Mixed-multipath training reduces this behavior substantially.

### Moderate multipath

| Error path | Original rate | Mitigated rate | Absolute reduction |
|---|---:|---:|---:|
| BPSK → 16QAM | 0.0840 | 0.0011 | 0.0829 |
| QPSK → 16QAM | 0.4451 | 0.1960 | 0.2491 |
| 8PSK → 16QAM | 0.5086 | 0.2211 | 0.2874 |

### Severe multipath

| Error path | Original rate | Mitigated rate | Absolute reduction |
|---|---:|---:|---:|
| BPSK → 16QAM | 0.4640 | 0.1023 | 0.3617 |
| QPSK → 16QAM | 0.8771 | 0.6274 | 0.2497 |
| 8PSK → 16QAM | 0.8491 | 0.6406 | 0.2086 |

The mitigation almost eliminates the moderate BPSK-to-16QAM error and strongly reduces the severe BPSK failure.

However, severe QPSK and 8PSK examples are still predicted as 16QAM in 62.74% and 64.06% of cases. This remains the main unresolved weakness.

## Accuracy by SNR

Under severe multipath, the mitigated model improves accuracy at every evaluated SNR.

| SNR | Original severe | Mitigated severe | Change |
|---:|---:|---:|---:|
| -4 dB | 0.3790 | 0.4660 | +0.0870 |
| 0 dB | 0.4350 | 0.5270 | +0.0920 |
| 4 dB | 0.4210 | 0.5790 | +0.1580 |
| 8 dB | 0.4680 | 0.6260 | +0.1580 |
| 12 dB | 0.3830 | 0.5930 | +0.2100 |
| 16 dB | 0.3780 | 0.5940 | +0.2160 |
| 20 dB | 0.4030 | 0.5800 | +0.1770 |

The largest gains occur from 12 to 20 dB. This supports the interpretation that the mitigation learned partial invariance to channel distortion rather than merely improving low-SNR noise handling.

## Interpretation

Mixed-multipath training changes the model’s decision boundaries in a useful direction.

The model becomes far less likely to interpret distorted BPSK as 16QAM, and it recovers substantial QPSK and 8PSK accuracy under moderate multipath. The reduction in clean QPSK and 8PSK performance indicates a robustness-specialization tradeoff.

The remaining severe-channel error is still dominated by QPSK and 8PSK collapsing toward 16QAM. A more targeted strategy is therefore needed to separate channel-induced amplitude dispersion from true QAM amplitude structure.

## Engineering conclusion

The 25% clean / 25% mild / 25% moderate / 25% severe mixed-training distribution is a successful first mitigation baseline.

It provides a strong robustness improvement while preserving more than 93% clean accuracy:

- clean: 93.34%
- mild: 90.10%
- moderate: 78.93%
- severe: 56.64%

This model is substantially more robust than the original clean-trained baseline, but severe multipath performance remains insufficient for a deployment-oriented classifier.

## Recommended next experiment

The next experiment should test a second controlled mitigation strategy focused on the remaining PSK-to-16QAM failure. Good candidates are:

1. reweight the mixed distribution toward moderate and severe multipath;
2. add a short learnable or deterministic equalization front end;
3. add a magnitude-normalized or phase-difference input branch;
4. use channel-aware auxiliary supervision;
5. compare mixed training with curriculum training from clean to severe channels.

The next experiment should reuse the same five seeds and paired four-condition evaluation protocol.

## Reproducible artifacts

### Dataset generation

- `configs/dataset_multipath_mixed_train_v1.yaml`
- `scripts/generate_dataset.py`
- `src/rfsil/data/dataset.py`

### Training

- `configs/train_baseline_groupnorm_multipath_mixed_v1.yaml`
- `configs/baseline_groupnorm_multipath_mixed_seed_sweep_v1.yaml`

### Evaluation

- `configs/evaluate_groupnorm_multipath_mixed_seed_sweep_v1.yaml`
- `configs/evaluate_groupnorm_multipath_mixed_mild_seed_sweep_v1.yaml`
- `configs/evaluate_groupnorm_multipath_mixed_moderate_seed_sweep_v1.yaml`
- `configs/evaluate_groupnorm_multipath_mixed_severe_seed_sweep_v1.yaml`

### Comparison

- `configs/compare_multipath_mitigation_v1.yaml`
- `scripts/compare_multipath_mitigation.py`
- `src/rfsil/evaluation/mitigation_comparison.py`

### Main generated outputs

- `results/multipath_mitigation_comparison_v1/summary.json`
- `reports/figures/multipath_mitigation_comparison_v1.png`
- `reports/figures/multipath_mitigation_confusions_v1.png`

## Final result

Mixed-multipath supervised training is validated as an effective robustness intervention.

It reduces the dominant PSK-to-16QAM failure pathway and improves moderate and severe held-out accuracy substantially, at the cost of a limited clean-channel performance reduction.
