# Channel-Aware Input Experiments v1

## Summary

This report evaluates handcrafted channel-aware input representations for the RF modulation-classification baseline.

The experiments were motivated by the remaining severe-multipath failure mode after mixed-multipath supervised training, especially the tendency of QPSK and 8PSK examples to be confused with 16QAM.

The tested approaches were:

1. Early concatenation of raw I/Q, normalized magnitude, and differential phase.
2. Early concatenation of raw I/Q and normalized magnitude only.
3. Early concatenation of raw I/Q and differential phase only.
4. Late fusion with separate raw-I/Q and differential-phase convolutional branches.

None of the channel-aware variants outperformed the established mixed-I/Q baseline. The early four-channel model underperformed across all four test conditions over five seeds. The one-seed magnitude, differential-phase, and late-fusion screens also failed to improve overall accuracy.

The current mixed-I/Q model remains the preferred robustness baseline.

## Reference baseline

The reference model is the GroupNorm CNN trained on the balanced mixed-multipath dataset:

- 25% clean
- 25% mild multipath
- 25% moderate multipath
- 25% severe multipath

Reference five-seed performance:

| Condition | Accuracy |
|---|---:|
| Clean | 0.9334 ± 0.0071 |
| Mild | 0.9010 ± 0.0102 |
| Moderate | 0.7893 ± 0.0190 |
| Severe | 0.5664 ± 0.0298 |

## Shared training protocol

Unless stated otherwise:

- Training examples: 5,600
- Validation examples: 1,400 clean examples
- Epochs: 30
- Optimizer and training settings: identical to the mixed-I/Q baseline
- Model normalization: GroupNorm
- Raw stored input: two-channel I/Q
- Evaluation conditions: clean, mild, moderate, severe paired held-out test sets

## Experiment 1: Early I/Q + magnitude + differential phase

### Representation

The stored two-channel I/Q tensor was expanded inside the model to:

1. I
2. Q
3. RMS-normalized magnitude
4. wrapped sample-to-sample differential phase divided by π

Parameter count: 73,540.

The existing two-channel behavior remained the default so old checkpoints continued to load unchanged.

### Five-seed validation outcome

| Statistic | Value |
|---|---:|
| Mean best clean-validation accuracy | 0.8509 |
| Standard deviation | 0.0227 |
| Range | 0.8293–0.8907 |
| Mean final validation accuracy | 0.7884 |
| Best seed | 2028 |

This was substantially below the mixed-I/Q model’s 0.9260 mean best validation accuracy.

### Five-seed held-out results

| Condition | Mixed-I/Q baseline | Four-channel model | Change |
|---|---:|---:|---:|
| Clean | 0.9334 | 0.8557 | -0.0777 |
| Mild | 0.9010 | 0.7700 | -0.1310 |
| Moderate | 0.7893 | 0.6839 | -0.1054 |
| Severe | 0.5664 | 0.5213 | -0.0451 |

The four-channel representation underperformed in every condition and improved only one of five seeds under severe multipath.

### Per-class degradation

The largest failure was 8PSK:

| Condition | 8PSK change |
|---|---:|
| Clean | -0.2303 |
| Mild | -0.2966 |
| Moderate | -0.2057 |
| Severe | -0.0503 |

QPSK also degraded strongly, while 16QAM remained stable or improved slightly under moderate and severe conditions.

The model therefore lost separation among PSK classes rather than merely shifting all predictions toward 16QAM.

## Experiment 2: Magnitude-only early concatenation

### Representation

Input channels:

1. I
2. Q
3. RMS-normalized magnitude

Parameter count: 73,316.

The screen used seed 2026.

### Seed-2026 performance

| Condition | Mixed I/Q | I/Q + magnitude | Change |
|---|---:|---:|---:|
| Clean | 0.9464 | 0.8657 | -0.0807 |
| Mild | 0.9207 | 0.8443 | -0.0764 |
| Moderate | 0.8179 | 0.7343 | -0.0836 |
| Severe | 0.5921 | 0.5285 | -0.0636 |

Magnitude-only concatenation underperformed in every condition.

The main losses were QPSK:

| Condition | QPSK change |
|---|---:|
| Clean | -0.1629 |
| Mild | -0.2114 |
| Moderate | -0.3114 |
| Severe | -0.2057 |

Magnitude-only concatenation was rejected.

## Experiment 3: Differential-phase-only early concatenation

### Representation

Input channels:

1. I
2. Q
3. wrapped sample-to-sample differential phase divided by π

Parameter count: 73,316.

The screen used seed 2026.

### Seed-2026 performance

| Condition | Mixed I/Q | I/Q + differential phase | Change |
|---|---:|---:|---:|
| Clean | 0.9464 | 0.9286 | -0.0178 |
| Mild | 0.9207 | 0.8993 | -0.0214 |
| Moderate | 0.8179 | 0.7986 | -0.0193 |
| Severe | 0.5921 | 0.5821 | -0.0100 |

This was the strongest handcrafted early-concatenation result, but it still underperformed in every condition.

Differential phase improved 8PSK:

| Condition | 8PSK change |
|---|---:|
| Clean | +0.0171 |
| Mild | +0.0257 |
| Moderate | +0.0714 |
| Severe | +0.0457 |

However, QPSK losses outweighed those gains:

| Condition | QPSK change |
|---|---:|
| Clean | -0.0886 |
| Mild | -0.0971 |
| Moderate | -0.1457 |
| Severe | -0.0771 |

Differential-phase early concatenation did not advance to five seeds.

## Experiment 4: Late fusion of I/Q and differential phase

### Architecture

The model processed raw I/Q and differential phase in separate branches:

```text
Raw I/Q ───────────── CNN branch ─────┐
                                      ├─ concatenation ─ fusion layer ─ classifier
Differential phase ─ CNN branch ──────┘
```

Architecture:

- Raw-I/Q branch: 2 → 16 → 32 → 64
- Differential-phase branch: 1 → 16 → 32 → 64
- Concatenated embedding: 128
- Fusion hidden layer: 256
- Output classes: 4
- Total parameters: 70,676

This was capacity-matched to the 73,092-parameter baseline.

### Seed-2026 validation outcome

| Statistic | Value |
|---|---:|
| Best clean-validation accuracy | 0.8786 |
| Best epoch | 27 |
| Final clean-validation accuracy | 0.8729 |

### Seed-2026 held-out results

| Condition | Mixed I/Q | Late fusion | Change |
|---|---:|---:|---:|
| Clean | 0.9464 | 0.8900 | -0.0564 |
| Mild | 0.9207 | 0.8293 | -0.0914 |
| Moderate | 0.8179 | 0.7329 | -0.0850 |
| Severe | 0.5921 | 0.5729 | -0.0193 |

Late fusion improved severe BPSK by 0.0343 and severe 16QAM by 0.0114, but QPSK fell by 0.1114 and 8PSK fell by 0.0114.

Separating differential phase into its own branch therefore did not solve the class-separation problem.

## Engineering interpretation

The results reject the hypothesis that handcrafted magnitude and raw sample-to-sample differential-phase channels will improve robustness when added directly to this classifier.

Several factors may explain the failure:

- Differential phase is unstable when the instantaneous envelope is small.
- Pulse shaping means adjacent samples do not correspond directly to independent symbol transitions.
- Frequency offset adds a persistent phase increment that can dominate the differential-phase channel.
- Frequency-selective multipath distorts local phase progression.
- The derived features may simplify 8PSK discrimination while harming QPSK discrimination.
- Magnitude-based features may encourage amplitude-sensitive boundaries that favor 16QAM.

The late-fusion result shows that the problem is not only interference in the first convolution. Even independent branch processing did not produce an overall gain.

## Decision

The following models are rejected as replacements for the mixed-I/Q baseline:

- I/Q + magnitude + differential phase
- I/Q + magnitude
- I/Q + differential phase
- late-fusion I/Q and differential phase

The mixed-I/Q model from the multipath-mitigation milestone remains the selected model.

## Reusable engineering contributions

Although the experiment was negative, it added reusable infrastructure:

- checkpoint-compatible input representations;
- legacy checkpoint reconstruction;
- supervised model factory;
- late-fusion model implementation;
- magnitude-only and differential-phase-only ablations;
- reproducible screening and evaluation configurations;
- expanded automated test coverage.

All previous baseline and SSL checkpoint behavior remains backward compatible.

## Recommended next experiment

The next experiment should stop adding handcrafted channels and instead test a learnable channel-correction front end.

Recommended initial design:

```text
Raw I/Q
   │
2 → 16 Conv1D, kernel 9
   │
16 → 16 Conv1D, kernel 9
   │
16 → 2 Conv1D, kernel 9
   │
Residual corrected I/Q = raw I/Q + learned correction
   │
Existing GroupNorm CNN
```

The equalizer should initially be trained end to end on the existing mixed-multipath dataset using seed 2026. It should advance to five seeds only if it improves moderate or severe held-out accuracy without a large clean or mild penalty.

## Main reproducibility artifacts

### Early channel-aware model

- `src/rfsil/data/transforms.py`
- `src/rfsil/models/baseline_cnn.py`
- `tests/test_channel_aware_input.py`

### Late-fusion infrastructure

- `src/rfsil/models/late_fusion_cnn.py`
- `src/rfsil/models/model_factory.py`
- `tests/test_late_fusion_cnn.py`

### Training configurations

- `configs/train_baseline_groupnorm_channel_aware_v1.yaml`
- `configs/train_baseline_groupnorm_channel_aware_magnitude_screen_v1.yaml`
- `configs/train_baseline_groupnorm_channel_aware_dphase_screen_v1.yaml`
- `configs/train_baseline_groupnorm_late_fusion_screen_v1.yaml`

### Evaluation configurations

- `configs/evaluate_groupnorm_channel_aware_seed_sweep_v1.yaml`
- `configs/evaluate_groupnorm_channel_aware_mild_seed_sweep_v1.yaml`
- `configs/evaluate_groupnorm_channel_aware_moderate_seed_sweep_v1.yaml`
- `configs/evaluate_groupnorm_channel_aware_severe_seed_sweep_v1.yaml`
- magnitude and differential-phase screening evaluation configs
- late-fusion screening evaluation configs

## Final conclusion

Handcrafted channel-aware input representations did not improve the RF classifier under the current synthetic channel protocol.

The negative result is useful: it narrows the next design space toward learned channel correction rather than direct feature concatenation.
