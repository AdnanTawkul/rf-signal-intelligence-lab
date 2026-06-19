# RadioML 2016.10A External Transfer Evaluation v1

## Executive summary

This report evaluates how the existing RF modulation-classification systems transfer from the project’s synthetic training distribution to the external RadioML 2016.10A generator.

The evaluation covers four shared modulation classes:

- BPSK
- QPSK
- 8PSK
- 16QAM

All results are aggregated across five trained seeds: `2026`, `2027`, `2028`, `2029`, and `2030`.

The main findings are:

1. The plain mixed-I/Q baseline is the strongest zero-shot external-transfer model.
2. Reducing the input window from 2,048 to 128 samples causes a large accuracy loss for every model.
3. The jointly trained residual front end is especially sensitive to short windows.
4. The frozen-backbone residual front end is highly sensitive to input amplitude.
5. A validation-selected input scale of `×112` restores the frozen model to baseline-level RadioML performance.
6. The scaled frozen model is statistically tied with the plain baseline rather than clearly superior.
7. The jointly trained residual model remains the preferred model for maximum robustness on the project’s original synthetic multipath benchmark, but not for short-window external transfer.

## Evaluation status

The unscaled RadioML experiments are the initial zero-shot evaluations.

The scaled RadioML experiments are post-hoc diagnostics. Their multipliers were selected using the RadioML validation split after the original test results had already been inspected. They must not be presented as untouched confirmatory test results.

## Dataset and conversion

The selected external benchmark is RadioML 2016.10A.

The conversion pipeline selects the four classes shared with the project and maps the source labels as follows:

| RadioML source label | Project label | Class index |
|---|---|---:|
| BPSK | BPSK | 0 |
| QPSK | QPSK | 1 |
| 8PSK | 8PSK | 2 |
| QAM16 | 16QAM | 3 |

The converted canonical tensor format is:

```text
iq:     [examples, 2, 128], float32
labels: [examples], int64
snr_db: [examples], float32
```

The deterministic split policy allocates examples independently inside every class/SNR group:

| Split | Examples per class/SNR group | Total examples |
|---|---:|---:|
| Train | 700 | 56,000 |
| Validation | 150 | 12,000 |
| Test | 150 | 12,000 |

The test set contains all four classes and all 20 RadioML SNR levels from `−20 dB` through `18 dB` in `2 dB` increments.

## Evaluated model families

### Mixed-I/Q baseline

The GroupNorm CNN trained on the balanced synthetic mixed-multipath dataset.

This model receives raw I/Q directly and does not use a learned residual signal front end.

### Joint residual front end

A residual I/Q correction network trained jointly with the CNN backbone.

This model was previously selected for maximum robustness on the project’s original synthetic multipath benchmark.

### Frozen-backbone residual front end

A parameter-efficient adaptation model that preserves the supervised backbone and trains only the residual front end.

The adaptation trains 2,944 parameters while retaining the original backbone and classifier.

## Primary external-transfer results

Two aggregate views are reported:

- **All SNRs:** all 20 RadioML SNR levels.
- **Shared grid:** `−4, 0, 4, 8, 12, 16 dB`, matching the common evaluation grid used for the synthetic controls.

| Model | All SNRs | Shared grid |
|---|---:|---:|
| Mixed-I/Q baseline | **52.54% ± 3.82%** | **67.03% ± 5.65%** |
| Joint residual, unscaled | 44.77% ± 2.59% | 54.86% ± 4.26% |
| Joint residual, ×40 | 46.40% ± 2.16% | 57.68% ± 3.64% |
| Frozen residual, unscaled | 40.37% ± 3.08% | 49.34% ± 4.74% |
| Frozen residual, ×112 | **52.89% ± 3.95%** | **67.38% ± 5.93%** |

![Overall RadioML transfer comparison](figures/radioml2016_external_transfer_overall_v1.png)

## Initial zero-shot interpretation

Without dataset-specific scaling, the mixed-I/Q baseline transfers best.

Relative to the baseline:

| Model | All-SNR change |
|---|---:|
| Joint residual, unscaled | −7.77 percentage points |
| Frozen residual, unscaled | −12.17 percentage points |

The unscaled result initially suggested that the residual front ends had specialized to the project’s synthetic distribution. Further controls showed that this interpretation was incomplete because the RadioML benchmark differs in both input length and amplitude scale.

## Short-window synthetic controls

The project models were trained using 2,048-sample windows, while RadioML provides 128-sample windows.

To isolate this factor, clean, mild, moderate, and severe synthetic datasets were regenerated with exactly 128 samples while preserving the original class balance, SNR grid, impairment settings, and example counts.

### 2,048-to-128-sample accuracy changes

| Condition | Model | 2,048 samples | 128 samples | Change |
|---|---|---:|---:|---:|
| Clean | Mixed-I/Q baseline | 93.34% | 66.97% | −26.37 pp |
| Clean | Joint residual | 93.23% | 58.30% | −34.93 pp |
| Clean | Frozen residual | 94.21% | 65.86% | −28.36 pp |
| Mild | Mixed-I/Q baseline | 90.10% | 64.16% | −25.94 pp |
| Mild | Joint residual | 91.69% | 55.61% | −36.07 pp |
| Mild | Frozen residual | 91.71% | 63.71% | −28.00 pp |
| Moderate | Mixed-I/Q baseline | 78.93% | 55.49% | −23.44 pp |
| Moderate | Joint residual | 84.97% | 50.96% | −34.01 pp |
| Moderate | Frozen residual | 84.67% | 56.91% | −27.76 pp |
| Severe | Mixed-I/Q baseline | 56.64% | 42.67% | −13.97 pp |
| Severe | Joint residual | 65.59% | 41.64% | −23.94 pp |
| Severe | Frozen residual | 63.84% | 43.59% | −20.26 pp |

Every model degrades substantially on 128-sample windows.

The jointly trained residual model is the most short-window-sensitive system, losing approximately 24–36 percentage points depending on the channel condition.

The frozen model remains close to the baseline on the short-window synthetic controls. Its much larger unscaled RadioML degradation therefore cannot be explained by sequence length alone.

## Exact length- and SNR-matched comparison

The shared six-SNR grid provides the cleanest comparison between synthetic 128-sample controls and RadioML 128-sample inputs.

| Model | Synthetic 128 | RadioML 128, unscaled | External minus synthetic |
|---|---:|---:|---:|
| Mixed-I/Q baseline | 65.57% ± 2.07% | 67.03% ± 5.65% | +1.47 pp |
| Joint residual | 56.97% ± 1.73% | 54.86% ± 4.26% | −2.11 pp |
| Frozen residual | 64.53% ± 2.05% | 49.34% ± 4.74% | −15.19 pp |

After matching window length and SNR:

- The baseline shows no evidence of an additional generator-domain penalty.
- The joint residual model has a small remaining external-transfer penalty.
- The frozen model still shows a large deficit, indicating a separate input-scale problem.

## Input-amplitude analysis

The selected models were trained without input RMS normalization.

The RadioML examples have a much smaller amplitude scale than the project’s synthetic examples:

```text
Synthetic median RMS: approximately 0.39
RadioML median RMS:   approximately 0.0086
Approximate ratio:    46×
```

This mismatch is especially important for the residual models because a learned correction can become large relative to a very small raw input.

## Validation-only input-scale selection

Global input multipliers were screened using the RadioML validation split only.

The final selection rule chose the smallest multiplier within 0.2 percentage points of the maximum validation accuracy.

| Model | Maximum validation point | Selected multiplier |
|---|---:|---:|
| Joint residual | ×52 | **×40** |
| Frozen residual | ×256 | **×112** |

The selected multipliers were then implemented through the reproducible configuration field:

```yaml
evaluation:
  input_scale: 112.0
```

Native evaluator scaling was verified to be prediction-equivalent to manually scaled NPZ archives.

## Post-hoc scaling effects

| Paired comparison | All-SNR change | Shared-grid change |
|---|---:|---:|
| Joint residual ×40 minus unscaled | +1.63 ± 0.56 pp, 5/5 seeds | +2.83 ± 1.39 pp, 5/5 seeds |
| Frozen residual ×112 minus unscaled | +12.52 ± 3.35 pp, 5/5 seeds | +18.04 ± 5.12 pp, 5/5 seeds |
| Frozen residual ×112 minus baseline | +0.35 ± 0.71 pp, 3/5 seeds | +0.34 ± 1.34 pp, 3/5 seeds |

The joint model receives a modest, consistent improvement from scaling but remains below the baseline.

The frozen model receives a large improvement on every seed and recovers to baseline-level performance.

The scaled frozen model is not clearly superior to the baseline. The mean paired advantage is only `0.35` percentage points and the direction changes across seeds.

## Accuracy by SNR

![RadioML accuracy by SNR](figures/radioml2016_external_transfer_snr_v1.png)

The SNR curves show three regimes:

1. **Very low SNR:** Below approximately `−12 dB`, all models remain near the four-class chance level.
2. **Transition region:** Accuracy rises rapidly between approximately `−6 dB` and `0 dB`.
3. **Positive SNR plateau:** The mixed-I/Q baseline and frozen residual `×112` remain close from `0 dB` through `18 dB`, while the joint residual `×40` stays consistently lower.

The scaled frozen model therefore recovers not only overall accuracy but also the full positive-SNR performance profile of the baseline.

## Per-class results

| Model | BPSK | QPSK | 8PSK | 16QAM |
|---|---:|---:|---:|---:|
| Mixed-I/Q baseline | 67.63% | 32.35% | 33.79% | **76.39%** |
| Joint residual, ×40 | 65.51% | 37.23% | **40.82%** | 42.05% |
| Frozen residual, ×112 | **70.05%** | **40.61%** | 34.84% | 66.06% |

![RadioML class accuracy](figures/radioml2016_external_transfer_classes_v1.png)

The systems reach similar aggregate performance through different class-level trade-offs:

- The frozen model is strongest on BPSK and QPSK.
- The joint model is strongest on 8PSK.
- The baseline is clearly strongest on 16QAM.
- The joint model’s weak 16QAM performance prevents it from matching the baseline overall.

## Model-selection decision

### Selected zero-shot external-transfer model

**Mixed-I/Q baseline**

Reason:

- Highest performance without dataset-specific calibration.
- Matches or exceeds the scaled residual systems within seed variability.
- Simpler inference path.
- Less sensitive to external amplitude scale.

### Selected amplitude-calibrated diagnostic

**Frozen-backbone residual front end with `×112` input scaling**

Reason:

- Scaling recovers 12.52 percentage points across all SNR levels.
- Recovery occurs on all five seeds.
- Final performance is effectively tied with the baseline.
- Demonstrates that the frozen residual adaptation can transfer when operated in a compatible amplitude range.

This selection is diagnostic rather than confirmatory because the multiplier was selected after the initial test results had already been examined.

### Selected maximum synthetic-robustness model

**Joint residual front end, unscaled**

Reason:

- Best previously established performance under the project’s original synthetic mixed-multipath evaluation.
- The external experiment does not invalidate that result.
- It does show that the model is not the preferred option for short-window external transfer.

## Engineering contributions

This milestone adds:

- A restricted RadioML 2016.10A pickle loader.
- Deterministic four-class conversion into the canonical NPZ format.
- Source archive and converted-data integrity checks.
- Balanced train, validation, and test splitting by class and SNR.
- Native 128-sample model evaluation.
- Five-seed zero-shot external evaluation protocols.
- Exact 128-sample synthetic control datasets.
- Configurable evaluator-level input scaling.
- Validation-only amplitude-scale screening.
- Prediction-equivalence checks for native input scaling.
- Reusable external-transfer metrics.
- Consolidated JSON summaries and publication-ready figures.
- Automated smoke and regression tests.

## Reproducibility

### Convert RadioML

```powershell
python scripts\convert_radioml2016.py `
  --config configs\dataset_radioml2016_four_class_v1.yaml
```

### Run the unscaled external evaluations

```powershell
python scripts\evaluate_seed_sweep.py `
  --config configs\evaluate_radioml2016_mixed_iq_baseline_seed_sweep_v1.yaml

python scripts\evaluate_seed_sweep.py `
  --config configs\evaluate_radioml2016_joint_residual_equalizer_seed_sweep_v1.yaml

python scripts\evaluate_seed_sweep.py `
  --config configs\evaluate_radioml2016_frozen_backbone_equalizer_seed_sweep_v1.yaml
```

### Run the post-hoc scaled diagnostics

```powershell
python scripts\evaluate_seed_sweep.py `
  --config configs\evaluate_radioml2016_joint_residual_equalizer_scaled_x40_seed_sweep_v1.yaml

python scripts\evaluate_seed_sweep.py `
  --config configs\evaluate_radioml2016_frozen_backbone_equalizer_scaled_x112_seed_sweep_v1.yaml
```

### Regenerate the consolidated analysis

```powershell
python scripts\analyze_radioml2016_external_transfer.py `
  --config configs\compare_radioml2016_external_transfer_v1.yaml
```

### Validate the repository

```powershell
python -m ruff check src tests scripts
python -m pytest -W error
```

At milestone completion, the repository contains **571 passing automated tests**.

## Limitations

1. RadioML 2016.10A is an external synthetic generator benchmark, not an over-the-air captured dataset.
2. Only the four classes shared with the current project are evaluated.
3. RadioML windows contain 128 samples, while the original models were trained using 2,048 samples.
4. The scaled test evaluations are post-hoc diagnostics rather than untouched confirmatory results.
5. The input multipliers are model- and dataset-specific operating-point adjustments, not physical signal calibrations.
6. The analysis uses five training seeds, which is sufficient to expose systematic effects but not to estimate extremely precise confidence intervals.
7. The current models still struggle to separate QPSK and 8PSK reliably on the external benchmark.

## Conclusion

The plain mixed-I/Q baseline is the strongest zero-shot external-transfer system.

The initial failure of the frozen residual model is not a pure generator-domain failure. It is primarily an input-scale-sensitive residual-adaptation failure that is amplified by RadioML’s much smaller signal amplitude. Validation-selected scaling restores the frozen model to baseline-level performance across the full SNR range.

The jointly trained residual model remains valuable for synthetic multipath robustness, but its stronger sensitivity to short windows and weaker external 16QAM performance make it unsuitable as the default external-transfer model.

The milestone demonstrates that robust RF model evaluation must control window length, amplitude scale, SNR support, and test-set selection status before attributing performance changes to dataset-domain shift.
