# SSL Label-Efficiency Evaluation v2

## Executive summary

This study evaluates whether self-supervised pretraining improves RF modulation classification when labeled training data are limited.

Three initialization strategies are compared:

- Random initialization
- SimCLR pretraining
- VICReg pretraining

The downstream classifier is evaluated at five labeled-data fractions:

- 1%: 56 labeled examples
- 5%: 280 labeled examples
- 10%: 560 labeled examples
- 25%: 1,400 labeled examples
- 100%: 5,600 labeled examples

Every downstream run uses an exact budget of 1,320 optimizer updates. Results are aggregated across five paired downstream seeds: `2026`, `2027`, `2028`, `2029`, and `2030`.

The complete experiment contains:

- 75 supervised downstream training runs
- 300 held-out evaluations
- Four held-out conditions: clean, mild, moderate, and severe multipath
- SimCLR and VICReg comparisons against paired random initialization

The main conclusion is that self-supervised initialization is useful only in targeted low-label regimes. SimCLR provides the strongest clean-channel result with 1% labels, while VICReg at 10% labels provides the strongest robustness-oriented SSL result. Random initialization remains preferable when the full labeled dataset is available.

## Experimental protocol

### Modulation classes

The study covers four modulation classes:

- BPSK
- QPSK
- 8PSK
- 16QAM

### Label fractions

| Fraction | Examples per class/SNR | Total labeled examples |
|---:|---:|---:|
| 1% | 2 | 56 |
| 5% | 10 | 280 |
| 10% | 20 | 560 |
| 25% | 50 | 1,400 |
| 100% | 200 | 5,600 |

The labeled subsets are selected using class/SNR-stratified sampling.

### Exact optimizer-step matching

All runs use:

- Batch size: `128`
- `drop_last: false`
- Target optimizer steps: `1,320`
- Exact budget matching: enabled

| Fraction | Steps per epoch | Epochs | Total updates |
|---:|---:|---:|---:|
| 1% | 1 | 1,320 | 1,320 |
| 5% | 3 | 440 | 1,320 |
| 10% | 5 | 264 | 1,320 |
| 25% | 11 | 120 | 1,320 |
| 100% | 44 | 30 | 1,320 |

### Seed protocol

For every fraction and method:

- The top-level supervised training seed equals the subset-selection seed.
- Seeds `2026` through `2030` are used.
- Random, SimCLR, and VICReg therefore receive paired labeled subsets and downstream seeds.

The same fixed SimCLR checkpoint is reused across all SimCLR downstream runs. The same fixed VICReg checkpoint is reused across all VICReg downstream runs. The reported variance therefore includes labeled-subset and downstream-training variation, but does not include SSL-pretraining variation.

## Validation results

The downstream checkpoint for each run is selected using best validation accuracy.

| Fraction | Random | SimCLR | VICReg |
|---:|---:|---:|---:|
| 1% | 74.73% ± 1.89% | **76.66% ± 0.85%** | 75.71% ± 1.64% |
| 5% | 92.34% ± 1.34% | 92.00% ± 0.97% | **93.04% ± 1.01%** |
| 10% | 93.96% ± 0.49% | 93.70% ± 0.58% | **94.24% ± 0.80%** |
| 25% | 95.01% ± 0.79% | **95.41% ± 0.63%** | 94.66% ± 0.57% |
| 100% | **95.64% ± 0.40%** | 94.57% ± 0.72% | 95.17% ± 0.71% |

![Validation accuracy versus label fraction](figures/ssl_label_efficiency_validation_v2.png)

### Paired validation changes

| Fraction | SimCLR vs random | Improved seeds | VICReg vs random | Improved seeds |
|---:|---:|---:|---:|---:|
| 1% | +1.93 ± 1.56 pp | 4/5 | +0.99 ± 1.00 pp | 4/5 |
| 5% | −0.34 ± 1.22 pp | 3/5 | +0.70 ± 0.48 pp | 4/5 |
| 10% | −0.26 ± 0.99 pp | 2/5 | +0.29 ± 1.01 pp | 3/5 |
| 25% | +0.40 ± 1.10 pp | 4/5 | −0.36 ± 0.74 pp | 2/5 |
| 100% | −1.07 ± 1.01 pp | 1/5 | −0.47 ± 1.07 pp | 1/5 |

Validation suggests a label-dependent SSL effect. SimCLR is strongest at 1%, VICReg is strongest at 5% and 10%, and random initialization is strongest at 100%.

## Held-out channel evaluation

Every selected checkpoint is evaluated on four held-out datasets:

- Clean
- Mild multipath
- Moderate multipath
- Severe multipath

This produces 300 completed evaluations.

![Held-out accuracy across channel conditions](figures/ssl_label_efficiency_held_out_v2.png)

![Paired accuracy change versus random initialization](figures/ssl_label_efficiency_paired_changes_v2.png)

## Selected systems

### Clean low-label specialist: 1% SimCLR

| Metric | Random | SimCLR | Change |
|---|---:|---:|---:|
| Validation | 74.73% | 76.66% | +1.93 pp |
| Clean test | 74.09% | 75.76% | +1.67 pp |
| Mild test | 67.94% | 68.10% | +0.16 pp |
| Moderate test | 53.43% | 51.39% | −2.04 pp |
| Severe test | 37.21% | 34.80% | −2.41 pp |
| Four-condition macro | 58.17% | 57.51% | −0.66 pp |

SimCLR provides a meaningful clean-channel gain with only 56 labels. However, the benefit does not transfer to moderate or severe multipath. This model is therefore selected only as a clean low-label specialist.

### Label-efficient compromise: 5% VICReg

| Condition | Random | VICReg | Change |
|---|---:|---:|---:|
| Clean | 91.86% | 92.24% | +0.39 pp |
| Mild | 82.56% | 83.43% | +0.87 pp |
| Moderate | 62.43% | 63.03% | +0.60 pp |
| Severe | 38.99% | 38.39% | −0.60 pp |
| Four-condition macro | 68.96% | 69.27% | +0.31 pp |

The 5% VICReg model uses only 280 labeled examples. It improves the clean, mild, and moderate conditions and improves the four-condition macro average on four of five paired seeds.

### Robust low-label model: 10% VICReg

| Condition | Random | VICReg | Change |
|---|---:|---:|---:|
| Clean | 93.87% | 93.64% | −0.23 pp |
| Mild | 83.64% | 84.50% | +0.86 pp |
| Moderate | 60.69% | 62.41% | +1.73 pp |
| Severe | 36.83% | 37.51% | +0.69 pp |
| Four-condition macro | 68.76% | 69.52% | +0.76 pp |

VICReg at 10% labels provides the strongest robustness-oriented SSL result. It sacrifices only 0.23 percentage points on clean data while improving all three multipath conditions. The four-condition macro average improves on four of five paired seeds.

### Full-label model: random initialization

| Condition | Random | SimCLR | VICReg |
|---|---:|---:|---:|
| Clean | **95.29%** | 94.49% | 94.74% |
| Mild | **86.81%** | 84.14% | 85.93% |
| Moderate | **65.26%** | 62.11% | 64.89% |
| Severe | **39.21%** | 38.47% | 39.01% |
| Four-condition macro | **71.64%** | 69.80% | 71.14% |

With all 5,600 labels available, random initialization is selected. SimCLR decreases the macro result on all five paired seeds, and neither SSL method provides a consistent full-label advantage.

## Confusion-matrix analysis

![Selected pooled confusion-matrix comparisons](figures/ssl_label_efficiency_selected_confusions_v2.png)

### 1% SimCLR on clean data

Overall accuracy increases from 74.09% to 75.76%.

Per-class recall changes:

| Class | Recall change |
|---|---:|
| BPSK | −2.63 pp |
| QPSK | +13.89 pp |
| 8PSK | −7.94 pp |
| 16QAM | +3.37 pp |

The clean gain is primarily explained by improved QPSK recall and reduced QPSK-to-8PSK confusion. This improvement is partly offset by lower 8PSK recall.

### 10% VICReg under moderate multipath

Overall accuracy increases from 60.69% to 62.41%.

Per-class recall changes:

| Class | Recall change |
|---|---:|
| BPSK | +4.63 pp |
| QPSK | +5.83 pp |
| 8PSK | −3.66 pp |
| 16QAM | +0.11 pp |

The robustness gain comes mainly from stronger BPSK and QPSK recall. The dominant unresolved error remains 8PSK being classified as 16QAM.

## Model-selection summary

The selected systems are:

```yaml
clean_low_label_model:
  fraction: 1%
  method: SimCLR
  intended_use: clean-channel classification with extremely limited labels

label_efficient_compromise:
  fraction: 5%
  method: VICReg
  intended_use: balanced low-label performance

robust_low_label_model:
  fraction: 10%
  method: VICReg
  intended_use: improved multipath robustness with limited labels

full_label_model:
  fraction: 100%
  method: random initialization
  intended_use: maximum performance when all labels are available
```

The 25% SimCLR system is not selected even though its mean macro change is positive. Only one of five paired seeds improves on the four-condition macro average, indicating that its result is not sufficiently consistent.

## Limitations

1. Only five paired downstream seeds are evaluated.
2. SimCLR and VICReg each use one fixed pretrained checkpoint.
3. SSL-pretraining variance is therefore not measured.
4. Model selection uses validation accuracy from the clean validation distribution.
5. The held-out datasets are synthetic channel distributions from the project.
6. The study does not claim that the selected SSL methods are statistically superior outside the evaluated protocol.
7. The RadioML external-transfer experiment is a separate evaluation and is not included in the model-selection results reported here.

## Reproduction

Prepare the exact-budget run matrix:

```powershell
python scripts\run_ssl_label_efficiency_seed_sweep.py `
  --config configs\ssl_label_efficiency_seed_sweep_v2.yaml `
  --dry-run
```

Execute or resume downstream training:

```powershell
python scripts\execute_ssl_label_efficiency_seed_sweep.py `
  --manifest results\ssl_label_efficiency_seed_sweep_v2\dry_run_manifest.json `
  --resume
```

Evaluate all checkpoints:

```powershell
python scripts\evaluate_ssl_label_efficiency.py `
  --config configs\evaluate_ssl_label_efficiency_v2.yaml `
  --resume
```

Generate the analysis package:

```powershell
python scripts\analyze_ssl_label_efficiency.py `
  --config configs\analyze_ssl_label_efficiency_v2.yaml
```

Run quality checks:

```powershell
python -m ruff check src tests scripts
python -m pytest -W error
```

At the completion of this milestone, the repository contains 620 passing automated tests.

## Conclusion

Self-supervised pretraining does not provide a universal improvement for RF modulation classification.

Its value depends on both the labeled-data regime and the expected channel conditions:

- SimCLR is useful as a clean-channel specialist at 1% labels.
- VICReg provides a balanced result at 5% labels.
- VICReg at 10% labels provides the strongest robustness-oriented SSL result.
- Random initialization remains best when the complete labeled dataset is available.

The results demonstrate that SSL initialization should be selected according to the deployment regime rather than treated as a universally superior replacement for supervised learning.
