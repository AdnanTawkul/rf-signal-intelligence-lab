# Learnable Residual Signal Front End v1

## Summary

This milestone evaluates a learnable residual front end for improving RF modulation classification under frequency-selective multipath.

The front end receives raw two-channel I/Q and predicts a length-preserving residual transformation before the existing GroupNorm CNN classifier:

```text
Raw I/Q
   │
Conv1D 2 → 16, kernel 9
   │
GroupNorm + GELU
   │
Conv1D 16 → 16, kernel 9
   │
GroupNorm + GELU
   │
Conv1D 16 → 2, kernel 9
   │
Transformed I/Q = input I/Q + learned residual
   │
Existing GroupNorm CNN
```

The output convolution is initialized to zero, so the complete model starts as an exact identity transformation.

Two training strategies were evaluated:

1. **Joint training:** residual front end and classifier trained together.
2. **Frozen-backbone adaptation:** a pretrained mixed-multipath classifier is loaded and frozen; only the 2,944-parameter residual front end is trained.

Both strategies improve robustness. Joint training provides the highest severe-multipath accuracy, while frozen-backbone adaptation preserves clean accuracy better, improves every channel condition on every paired seed, and uses a substantially smaller learned signal transformation.

## Reference mixed-I/Q baseline

The reference model is the GroupNorm CNN trained on a balanced mixed-multipath dataset:

- 25% clean
- 25% mild multipath
- 25% moderate multipath
- 25% severe multipath

Five-seed held-out performance:

| Condition | Mean accuracy | Standard deviation |
|---|---:|---:|
| Clean | 0.9334 | 0.0071 |
| Mild | 0.9010 | 0.0102 |
| Moderate | 0.7893 | 0.0190 |
| Severe | 0.5664 | 0.0298 |

## Model size

| Component | Parameters |
|---|---:|
| Residual front end | 2,944 |
| GroupNorm CNN backbone | 73,092 |
| Complete model | 76,036 |

For frozen-backbone adaptation, only the 2,944 front-end parameters are optimized.

## Jointly trained residual front end

### Five-seed performance

| Condition | Baseline | Joint model | Paired change | Improved seeds |
|---|---:|---:|---:|---:|
| Clean | 0.9334 | 0.9323 | -0.0011 | 2/5 |
| Mild | 0.9010 | 0.9169 | +0.0159 | 4/5 |
| Moderate | 0.7893 | 0.8497 | +0.0604 | 5/5 |
| Severe | 0.5664 | 0.6559 | +0.0894 | 5/5 |

Joint training produces the strongest severe-channel mean result.

### Class-level effects

Under moderate multipath:

- BPSK: +0.0080
- QPSK: +0.0623
- 8PSK: +0.1880
- 16QAM: -0.0166

Under severe multipath:

- BPSK: +0.0520
- QPSK: +0.0989
- 8PSK: +0.2051
- 16QAM: +0.0017

### Correction magnitude

| Condition | Relative correction RMS |
|---|---:|
| Clean | 3.2860 |
| Mild | 3.2694 |
| Moderate | 3.2152 |
| Severe | 3.1883 |

The jointly trained transformation is much larger than the input RMS. It should therefore be described as a learned residual signal representation rather than a physically interpretable channel inverse.

## Frozen-backbone residual front end

### Controlled training protocol

For every seed:

1. Load the corresponding mixed-I/Q baseline checkpoint.
2. Copy it into the residual model backbone.
3. Freeze all 73,092 backbone parameters.
4. Keep the frozen backbone in evaluation mode during training.
5. Train only the 2,944-parameter residual front end.
6. Verify that the frozen backbone remains bit-for-bit unchanged.

Paired checkpoint mapping:

- seed 2026 → baseline seed 2026
- seed 2027 → baseline seed 2027
- seed 2028 → baseline seed 2028
- seed 2029 → baseline seed 2029
- seed 2030 → baseline seed 2030

### Five-seed validation

| Statistic | Value |
|---|---:|
| Mean best validation accuracy | 0.9369 |
| Standard deviation | 0.0061 |
| Range | 0.9271–0.9450 |
| Mean final validation accuracy | 0.9316 |
| Best seed | 2029 |

### Five-seed held-out performance

| Condition | Baseline | Frozen front end | Paired change | Improved seeds |
|---|---:|---:|---:|---:|
| Clean | 0.9334 | 0.9421 | +0.0087 ± 0.0045 | 5/5 |
| Mild | 0.9010 | 0.9171 | +0.0161 ± 0.0083 | 5/5 |
| Moderate | 0.7893 | 0.8467 | +0.0574 ± 0.0139 | 5/5 |
| Severe | 0.5664 | 0.6384 | +0.0720 ± 0.0223 | 5/5 |

The frozen front end improves every channel condition on every paired seed.

### Mean class accuracies

| Condition | BPSK | QPSK | 8PSK | 16QAM |
|---|---:|---:|---:|---:|
| Clean | 0.9983 | 0.8606 | 0.9349 | 0.9749 |
| Mild | 0.9977 | 0.7983 | 0.9057 | 0.9669 |
| Moderate | 0.9943 | 0.6731 | 0.7583 | 0.9611 |
| Severe | 0.8766 | 0.3217 | 0.3806 | 0.9749 |

### Reduction of errors toward 16QAM

| Condition | Error path | Baseline | Frozen front end | Reduction |
|---|---|---:|---:|---:|
| Moderate | QPSK → 16QAM | 0.1960 | 0.1006 | 0.0954 |
| Moderate | 8PSK → 16QAM | 0.2211 | 0.1263 | 0.0949 |
| Severe | BPSK → 16QAM | 0.1023 | 0.0760 | 0.0263 |
| Severe | QPSK → 16QAM | 0.6274 | 0.4771 | 0.1503 |
| Severe | 8PSK → 16QAM | 0.6406 | 0.5303 | 0.1103 |

The front end substantially reduces the dominant PSK-to-16QAM collapse.

### Accuracy by severe-channel SNR

| SNR (dB) | Frozen front end accuracy |
|---:|---:|
| -4 | 0.520 |
| 0 | 0.578 |
| 4 | 0.641 |
| 8 | 0.728 |
| 12 | 0.673 |
| 16 | 0.667 |
| 20 | 0.662 |

The front end improves severe-channel accuracy across every evaluated SNR.

### Correction magnitude

| Condition | Mean absolute correction | Relative correction RMS |
|---|---:|---:|
| Clean | 0.1584 ± 0.0388 | 0.6049 ± 0.1506 |
| Mild | 0.1564 ± 0.0381 | 0.6003 ± 0.1486 |
| Moderate | 0.1523 ± 0.0367 | 0.5859 ± 0.1435 |
| Severe | 0.1485 ± 0.0354 | 0.5752 ± 0.1404 |

The frozen front end uses a far smaller transformation than the jointly trained model. Its correction RMS is approximately 0.58–0.60 times the input RMS instead of approximately 3.19–3.29 times the input RMS.

The transformation is still not a proven physical channel inverse, but it is a more constrained and interpretable adaptation than the jointly trained representation.

## Joint versus frozen training

| Condition | Joint change | Frozen change | Preferred result |
|---|---:|---:|---|
| Clean | -0.0011 | +0.0087 | Frozen |
| Mild | +0.0159 | +0.0161 | Approximately equal |
| Moderate | +0.0604 | +0.0574 | Joint by 0.0030 |
| Severe | +0.0894 | +0.0720 | Joint by 0.0174 |

### Interpretation

- **Joint training** is the highest-accuracy robustness model, especially under severe multipath.
- **Frozen-backbone training** is the strongest parameter-efficient adaptation result.
- The frozen control proves that robustness gains do not require classifier co-adaptation.
- The frozen model preserves clean accuracy better and produces much smaller signal corrections.
- Both models reduce the dominant PSK-to-16QAM failure pathway.

## Model-selection decision

Two models should be retained for different purposes:

### Selected maximum-robustness model

The jointly trained residual front end remains the selected model when maximum moderate/severe accuracy is the primary objective.

### Selected parameter-efficient adaptation model

The frozen-backbone residual front end is selected when the objective is:

- adapting an existing deployed classifier;
- training only a small number of parameters;
- preserving clean-channel behavior;
- maintaining a more constrained signal transformation;
- demonstrating a controlled scientific comparison.

The frozen model is not merely an ablation. It is a successful adaptation method that improves all four conditions on all five paired seeds.

## Reusable engineering contributions

This milestone adds:

- checkpoint-compatible residual front-end architecture;
- exact identity initialization;
- supervised backbone checkpoint loading;
- frozen-backbone training mode;
- optimizer filtering to train only parameters with gradients enabled;
- paired seed-template expansion;
- generated per-seed training configurations;
- bit-for-bit frozen-backbone verification;
- correction-magnitude analysis;
- consolidated accuracy, SNR, and confusion comparison figures;
- backward compatibility with prior supervised and SSL workflows.

## Main reproducibility artifacts

### Model and training infrastructure

- `src/rfsil/models/residual_equalizer_cnn.py`
- `src/rfsil/training/backbone_initialization.py`
- `src/rfsil/training/seed_sweep_config.py`
- `scripts/train_baseline.py`
- `scripts/run_baseline_seed_sweep.py`

### Joint-training configs

- `configs/train_baseline_groupnorm_residual_equalizer_v1.yaml`
- `configs/baseline_groupnorm_residual_equalizer_seed_sweep_v1.yaml`
- residual-equalizer clean, mild, moderate, and severe evaluation configs
- `configs/compare_residual_equalizer_v1.yaml`

### Frozen-backbone configs

- `configs/train_baseline_groupnorm_frozen_backbone_equalizer_v1.yaml`
- `configs/baseline_groupnorm_frozen_backbone_equalizer_seed_sweep_v1.yaml`
- frozen-backbone clean, mild, moderate, and severe evaluation configs
- `configs/compare_frozen_backbone_equalizer_v1.yaml`

### Analysis artifacts

- `results/residual_equalizer_comparison_v1/summary.json`
- `results/frozen_backbone_equalizer_comparison_v1/summary.json`
- `reports/figures/residual_equalizer_comparison_v1.png`
- `reports/figures/residual_equalizer_confusions_v1.png`
- `reports/figures/frozen_backbone_equalizer_comparison_v1.png`
- `reports/figures/frozen_backbone_equalizer_confusions_v1.png`

## Conclusion

A small learnable residual signal front end significantly improves RF modulation classification under multipath.

Joint end-to-end training achieves the highest severe-channel accuracy, while frozen-backbone adaptation demonstrates that only 2,944 trainable parameters are sufficient to improve a fixed pretrained classifier across clean, mild, moderate, and severe conditions on every paired seed.

This is the strongest robustness result in the project so far and provides both a high-performance model and a parameter-efficient adaptation method.
