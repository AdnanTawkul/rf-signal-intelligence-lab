# Channel-Shift Detection from IQ Features v1

## Executive Summary

This milestone evaluates whether an RF modulation classifier can detect that its input channel has shifted from the clean training distribution to frequency-selective multipath.

The study begins with model-output uncertainty scores and then introduces deterministic IQ-derived channel features. The central result is clear:

> **Model-output uncertainty is not a reliable multipath detector, while explicit IQ-derived features provide strong and consistent channel-shift detection.**

Across 75 downstream checkpoints and three shifted conditions:

- Output energy remains close to chance: AUROC `0.5275`, `0.5328`, and `0.4843` for mild, moderate, and severe multipath.
- A single lag-8 complex-autocorrelation feature reaches AUROC `0.8019`, `0.9314`, and `0.9497`.
- A development-selected linear detector using all 21 IQ features reaches AUROC `0.8312`, `0.9512`, and `0.9794`.
- Adding checkpoint-specific output energy produces only marginal AUROC changes but slightly improves the average FPR at 95% shift recall.

The selected primary detector is the **all-IQ linear detector**. The **IQ-plus-energy fusion detector** is retained as an optional operating-point variant when reducing FPR@95TPR is more important than maximizing mean AUROC.

## Research Question

The preceding confidence-calibration study showed that one scalar temperature fitted on clean validation data does not transfer reliably to multipath. This raises a deployment question:

> Can the system detect that the RF channel has shifted before trusting or calibrating the classifier confidence?

The milestone addresses four sub-questions:

1. Do output-derived uncertainty scores identify multipath shift?
2. Which deterministic IQ features are sensitive to the channel change?
3. Can a development-only multivariate IQ detector improve over the strongest single feature?
4. Does fusing model-output energy with IQ features provide additional value?

## Experimental Assets

### Paired held-out datasets

The study uses four row-aligned test datasets:

| Condition | Examples | IQ shape |
|---|---:|---|
| Clean | 1,400 | `[1400, 2, 2048]` |
| Mild multipath | 1,400 | `[1400, 2, 2048]` |
| Moderate multipath | 1,400 | `[1400, 2, 2048]` |
| Severe multipath | 1,400 | `[1400, 2, 2048]` |

The following metadata match exactly across all four conditions:

- Class label
- SNR
- Carrier-frequency offset
- Carrier-phase offset
- Amplitude scale
- Integer timing shift
- Rayleigh-fading flag
- Example seed

This pairing isolates the channel condition: corresponding rows represent the same underlying class, SNR, and impairment metadata with only the multipath realization changed.

### Prediction artifacts

The existing calibration workflow provides 300 row-aligned prediction artifacts:

```text
75 checkpoints x 4 held-out conditions = 300 artifacts
```

Every artifact contains labels, predictions, logits, probabilities, SNR values, and class names. The fusion audit verified:

- `300/300` calibration artifacts aligned with the IQ feature rows
- `75/75` unique checkpoints contained all four conditions
- Labels matched exactly
- SNR values matched within an absolute tolerance of `1e-8`
- All output-energy scores were finite

### Downstream checkpoints

The 75 checkpoints span:

- Five label fractions: `1%`, `5%`, `10%`, `25%`, `100%`
- Three initialization methods: random, SimCLR, VICReg
- Five downstream seeds: `2026` through `2030`

## Leakage-Controlled Split

A deterministic class/SNR-stratified split uses `example_seed` as the grouping key:

| Partition | Paired examples |
|---|---:|
| Development | 700 |
| Untouched test | 700 |
| Total | 1,400 |
| Class/SNR strata | 28 |

All clean and shifted versions of one `example_seed` stay in the same partition. Grouped cross-validation also keeps every version of one example in the same fold.

The test partition is not used for:

- Feature-direction selection
- Standardization
- L2 regularization selection
- Linear coefficient fitting
- Fusion coefficient fitting

## IQ Feature Representation

Each IQ window is mean-centered and normalized by its centered complex RMS power before feature extraction. This removes trivial gain dependence and focuses the detector on waveform structure.

The representation contains 21 deterministic features.

### Amplitude features

1. `amplitude_mean`
2. `amplitude_coefficient_of_variation`
3. `amplitude_skewness`
4. `amplitude_excess_kurtosis`
5. `peak_to_rms_ratio`

### I/Q geometry

6. `iq_variance_log_ratio_abs`
7. `iq_correlation_abs`
8. `iq_covariance_log_condition`

### Differential phase

9. `dphase_mean_abs_normalized`
10. `dphase_circular_dispersion`
11. `dphase_circular_std`

### Normalized complex autocorrelation

12. `autocorrelation_abs_lag_1`
13. `autocorrelation_abs_lag_2`
14. `autocorrelation_abs_lag_4`
15. `autocorrelation_abs_lag_8`

### Spectral features

16. `spectral_entropy`
17. `spectral_flatness`
18. `spectral_peak_fraction`
19. `spectral_occupancy_fraction`
20. `spectral_centroid_abs`
21. `spectral_spread`

All feature artifacts are versioned NPZ files and include the paired metadata required to verify row alignment.

## Metrics

Every clean-versus-shift comparison reports:

- **AUROC** — threshold-independent discrimination; higher is better
- **Average precision** — precision-recall summary; higher is better
- **FPR@95TPR** — fraction of clean windows falsely flagged when retaining 95% shift recall; lower is better

The score direction is frozen using development data. Larger final detector scores always mean more shift-like.

## Output-Only Baseline

The initial baseline evaluates:

- Maximum-softmax-probability uncertainty
- Predictive entropy
- Negative logit margin
- Energy

These scores are evaluated across all 75 checkpoints and three shifted conditions. None produces an operationally useful detector. Energy is the strongest output-only candidate, but its direction changes with channel severity.

Mean energy across checkpoints:

| Condition | Mean energy | Standard deviation |
|---|---:|---:|
| Clean | -4.839611 | 0.804567 |
| Mild | -4.541073 | 0.697480 |
| Moderate | -4.569466 | 0.686020 |
| Severe | -5.418207 | 0.862798 |

Mild and moderate shift move energy upward relative to clean, whereas severe shift moves it downward. A universal fixed direction is therefore not physically or operationally stable.

## Individual IQ-Feature Screening

Each of the 21 features is evaluated independently. Its direction is selected once using the balanced pooled development data and then frozen for all three test conditions.

The strongest individual feature is:

> **`autocorrelation_abs_lag_8`**

| Condition | AUROC | Average precision | FPR@95TPR |
|---|---:|---:|---:|
| Mild | 0.8019 | 0.8337 | 0.7529 |
| Moderate | 0.9314 | 0.9486 | 0.4157 |
| Severe | 0.9497 | 0.9614 | 0.3800 |

Its mean AUROC across the three shifted conditions is `0.8943`.

The result shows that multipath changes the temporal correlation structure in a way that is much more detectable than the classifier's own confidence response.

## All-IQ Linear Detector

The multivariate detector uses all 21 standardized IQ features.

### Training protocol

- Balanced clean-versus-shift development rows
- Pooled mild, moderate, and severe shifted development data
- Five grouped cross-validation folds
- Group key: `example_seed`
- L2 candidate strengths from `1e-6` through `100`
- Candidate selection by mean validation AUROC, then mean FPR95, then worst-fold AUROC
- Final fit on all development rows
- Frozen evaluation on the 700-example test partition

The selected L2 strength is `0.1`. The pooled development AUROC is `0.9176`.

The dominant learned input is lag-8 autocorrelation, with differential-phase and amplitude-shape terms providing complementary information. Coefficient signs are conditional multivariate effects and should not be interpreted as independent physical directions.

### Untouched test results

| Condition | AUROC | Average precision | FPR@95TPR |
|---|---:|---:|---:|
| Mild | 0.8312 | 0.8619 | 0.7671 |
| Moderate | 0.9512 | 0.9614 | 0.3129 |
| Severe | 0.9794 | 0.9829 | 0.1229 |

Relative to lag-8 alone, all-IQ improves AUROC by:

- `+0.0293` on mild multipath
- `+0.0198` on moderate multipath
- `+0.0297` on severe multipath

It also substantially improves FPR95 on moderate and severe shift.

## Four-System Comparison

The final comparison evaluates four predefined systems over all 75 checkpoints:

1. Lag-8 autocorrelation
2. All-IQ linear detector
3. Checkpoint-specific output energy
4. IQ-plus-energy linear fusion

The resulting matrix contains:

```text
75 checkpoints x 3 shifted conditions x 4 systems = 900 comparisons
```

### Aggregate held-out results

| Condition | System | Mean AUROC | Mean AP | Mean FPR@95TPR | AUROC >= 0.8 |
|---|---|---:|---:|---:|---:|
| Mild | Lag-8 autocorrelation | 0.8019 | 0.8337 | 0.7529 | 75/75 |
| Mild | All-IQ linear | **0.8312** | **0.8619** | 0.7671 | 75/75 |
| Mild | Output energy | 0.5275 | 0.5293 | 0.9156 | 0/75 |
| Mild | IQ + energy fusion | 0.8281 | 0.8573 | **0.7366** | 75/75 |
| Moderate | Lag-8 autocorrelation | 0.9314 | 0.9486 | 0.4157 | 75/75 |
| Moderate | All-IQ linear | 0.9512 | 0.9614 | **0.3129** | 75/75 |
| Moderate | Output energy | 0.5328 | 0.5533 | 0.9052 | 0/75 |
| Moderate | IQ + energy fusion | **0.9521** | **0.9621** | 0.3173 | 75/75 |
| Severe | Lag-8 autocorrelation | 0.9497 | 0.9614 | 0.3800 | 75/75 |
| Severe | All-IQ linear | 0.9794 | **0.9829** | 0.1229 | 75/75 |
| Severe | Output energy | 0.4843 | 0.5236 | 0.9275 | 0/75 |
| Severe | IQ + energy fusion | **0.9798** | 0.9828 | **0.1200** | 75/75 |

### Three-condition macro summary

| System | Mean AUROC | Mean AP | Mean FPR@95TPR |
|---|---:|---:|---:|
| Lag-8 autocorrelation | 0.8943 | 0.9146 | 0.5162 |
| **All-IQ linear** | **0.9206** | **0.9354** | 0.4010 |
| Output energy | 0.5149 | 0.5354 | 0.9161 |
| IQ + energy fusion | 0.9200 | 0.9341 | **0.3913** |

The all-IQ detector has the highest mean AUROC and mean average precision. Fusion has the lowest mean FPR95, but its mean AUROC is slightly lower.

## Paired Fusion Changes

The table below reports fusion minus baseline across the 75 paired checkpoints.

| Condition | Baseline | Mean AUROC change | Mean FPR95 change | AUROC improved | FPR95 improved |
|---|---|---:|---:|---:|---:|
| Mild | Lag-8 | +0.0262 | -0.0162 | 75/75 | 57/75 |
| Mild | All-IQ linear | -0.0031 | -0.0305 | 7/75 | 71/75 |
| Mild | Output energy | +0.3006 | -0.1790 | 75/75 | 75/75 |
| Moderate | Lag-8 | +0.0207 | -0.0984 | 75/75 | 75/75 |
| Moderate | All-IQ linear | +0.0009 | +0.0045 | 58/75 | 32/75 |
| Moderate | Output energy | +0.4193 | -0.5879 | 75/75 | 75/75 |
| Severe | Lag-8 | +0.0301 | -0.2600 | 75/75 | 75/75 |
| Severe | All-IQ linear | +0.0004 | -0.0028 | 60/75 | 41/75 |
| Severe | Output energy | +0.4955 | -0.8075 | 75/75 | 75/75 |

Fusion adds little beyond the all-IQ model in AUROC terms. Its main value is operating-point adjustment, especially the mild FPR95 reduction from `0.7671` to `0.7366`.

## System Selection

### Primary detector

> **All-IQ linear detector**

Reasons:

- Highest mean AUROC across the three channel severities
- Highest mean average precision
- Strong performance on every shifted condition
- Independent of classifier checkpoint and initialization method
- Directly measures signal structure rather than relying on classifier confidence
- Simpler interpretation than a checkpoint-specific fusion system

### Optional operating-point variant

> **IQ-plus-energy fusion detector**

Use this variant only when the deployment objective prioritizes slightly lower mean FPR95 over maximum mean AUROC. It does not establish output energy as independently useful; the IQ features provide nearly all of the discrimination.

### Rejected primary baseline

> **Output energy alone**

It is rejected because:

- Mean AUROC is close to chance
- Severe-shift AUROC is below chance
- FPR95 remains above `0.90` for all severities
- Score direction is not stable across channel severity

## Figures

### AUROC comparison

![Channel-shift detector AUROC](figures/channel_shift_detector_auroc_v1.png)

The all-IQ and fusion detectors substantially outperform output energy. Error bars show variability across the 75 checkpoints.

### FPR at 95% shift recall

![Channel-shift detector FPR95](figures/channel_shift_detector_fpr95_v1.png)

Moderate and severe shifts can be detected at useful high-recall operating points. Mild shift remains difficult.

### Fusion AUROC change

![Fusion AUROC change](figures/channel_shift_fusion_auroc_change_v1.png)

Fusion improves strongly over output energy and lag-8, but contributes almost no AUROC improvement over all-IQ.

### Fusion FPR95 change

![Fusion FPR95 change](figures/channel_shift_fusion_fpr95_change_v1.png)

Negative values indicate improvement. Fusion improves mild FPR95 relative to all-IQ, is slightly worse on moderate shift, and is nearly neutral on severe shift.

## Reproduction

### 1. Extract versioned IQ features

```powershell
python scripts\extract_iq_channel_features.py `
  --config configs\extract_iq_channel_features_v1.yaml
```

### 2. Evaluate individual IQ features

```powershell
python scripts\analyze_iq_feature_shift_detection.py `
  --config configs\analyze_iq_feature_shift_detection_v1.yaml
```

### 3. Fit the all-IQ linear detector

```powershell
python scripts\analyze_linear_iq_shift_detection.py `
  --config configs\analyze_linear_iq_shift_detection_v1.yaml
```

### 4. Regenerate the output-only baseline

```powershell
python scripts\analyze_output_channel_shift.py `
  --config configs\analyze_output_channel_shift_v1.yaml
```

### 5. Run the four-system comparison

```powershell
python scripts\compare_channel_shift_detectors.py `
  --config configs\compare_channel_shift_detectors_v1.yaml
```

### 6. Regenerate figures

```powershell
python scripts\visualize_channel_shift_detectors.py `
  --config configs\visualize_channel_shift_detectors_v1.yaml
```

### 7. Run quality checks

```powershell
python -m ruff check src tests scripts
python -m pytest -W error
git diff --check
```

Validated milestone state:

```text
Ruff: passed
Pytest: 1,051 passed
Comparison records: 900
Calibration/IQ alignment: 300/300
Unique checkpoints: 75/75
```

## Main Artifacts

### Configurations

- `configs/extract_iq_channel_features_v1.yaml`
- `configs/analyze_output_channel_shift_v1.yaml`
- `configs/analyze_iq_feature_shift_detection_v1.yaml`
- `configs/analyze_linear_iq_shift_detection_v1.yaml`
- `configs/compare_channel_shift_detectors_v1.yaml`
- `configs/visualize_channel_shift_detectors_v1.yaml`

### Scripts

- `scripts/extract_iq_channel_features.py`
- `scripts/analyze_output_channel_shift.py`
- `scripts/analyze_iq_feature_shift_detection.py`
- `scripts/analyze_linear_iq_shift_detection.py`
- `scripts/compare_channel_shift_detectors.py`
- `scripts/visualize_channel_shift_detectors.py`

### Reusable modules

- `src/rfsil/evaluation/channel_shift.py`
- `src/rfsil/evaluation/channel_shift_analysis.py`
- `src/rfsil/evaluation/iq_channel_features.py`
- `src/rfsil/evaluation/iq_feature_artifacts.py`
- `src/rfsil/evaluation/paired_shift_split.py`
- `src/rfsil/evaluation/iq_feature_detection.py`
- `src/rfsil/evaluation/linear_shift_detector.py`
- `src/rfsil/evaluation/shift_detector_comparison.py`
- `src/rfsil/evaluation/shift_detector_visualization.py`

### Generated local results

- `results/output_channel_shift_detection_v1/aggregate_metrics.json`
- `results/iq_channel_features_v1/`
- `results/iq_feature_shift_detection_v1/aggregate_metrics.json`
- `results/linear_iq_shift_detection_v1/aggregate_metrics.json`
- `results/channel_shift_detector_comparison_v1/aggregate_metrics.json`

### Report figures

- `reports/figures/channel_shift_detector_auroc_v1.png`
- `reports/figures/channel_shift_detector_fpr95_v1.png`
- `reports/figures/channel_shift_fusion_auroc_change_v1.png`
- `reports/figures/channel_shift_fusion_fpr95_change_v1.png`

## Limitations

- The evaluation uses synthetic paired datasets. Real receiver and over-the-air transfer are not established.
- The same feature definitions are not yet validated across different sample rates, bandwidths, or window lengths.
- The detector recognizes distribution shift but does not identify a physical channel model or estimate channel taps.
- Mild multipath remains difficult at 95% recall.
- The all-IQ model is trained on pooled mild, moderate, and severe development data; transfer to unseen impairment families requires separate validation.
- The energy-fusion system is checkpoint-specific and therefore more complex to deploy than the all-IQ detector.
- Detection alone does not recalibrate probabilities. Channel-aware calibration remains a separate next milestone.
- AUROC summarizes ranking and does not define the final deployment threshold or the cost of false alarms.

## Deployment Implications

A practical deployment path is:

1. Compute the 21 deterministic IQ features for every inference window.
2. Score the window with the frozen all-IQ detector.
3. Apply an operating threshold selected for the target false-alarm budget.
4. If the window is flagged, mark the classifier output as distribution-shifted.
5. Route the output to a channel-aware calibration, abstention, or human-review policy.

The detector should not be used as proof that a signal is clean. It is one layer in a deployment-confidence system and must be validated on the target receiver and acquisition chain.

## Conclusion

The study demonstrates that channel shift must be detected from the RF signal itself rather than inferred from classifier confidence.

The classifier can remain highly confident while becoming wrong under severe multipath. In contrast, deterministic temporal, phase, amplitude, and spectral IQ features expose the changed channel structure. A lightweight linear model over those features reaches strong discrimination without changing the modulation classifier.

The selected result is:

> **All-IQ linear: AUROC 0.8312 on mild, 0.9512 on moderate, and 0.9794 on severe multipath across 75 downstream checkpoints.**

This detector provides the missing signal-side gate required for the next milestone: channel-aware confidence calibration under detected distribution shift.
