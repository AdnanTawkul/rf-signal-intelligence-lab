# SSL Confidence Calibration Under Channel Shift

## Executive summary

This study evaluates whether scalar temperature scaling fitted on the clean validation split improves confidence calibration for RF modulation classifiers when transferred to held-out channel conditions.

The experiment covers:

- 75 trained checkpoints
- 5 labeled-data fractions: 1%, 5%, 10%, 25%, and 100%
- 3 initialization strategies: random initialization, SimCLR, and VICReg
- 5 training seeds per label-fraction and initialization combination
- 4 held-out conditions: clean, mild multipath, moderate multipath, and severe multipath
- 300 held-out calibration evaluations in total

The central result is that validation-fitted temperature scaling is effective in distribution but unreliable under channel shift. On clean held-out data, it improves negative log-likelihood (NLL), expected calibration error (ECE), and Brier score for 71 of 75 models. Under multipath, the same frozen temperatures usually worsen confidence quality, with the degradation increasing as channel severity grows.

A notable exception is the 1% label regime. These models are often overconfident and generally benefit from temperature scaling across all conditions. From 5% labels upward, fitted temperatures frequently sharpen predictions, helping clean calibration while amplifying overconfidence under multipath.

Temperature scaling preserved classification accuracy in all 300 evaluations.

## Research question

The experiment addresses the following deployment question:

> Can a single scalar temperature fitted on clean validation logits provide reliable confidence estimates when the RF channel changes at inference time?

This matters because classification accuracy alone does not indicate whether predicted probabilities are trustworthy. A model may preserve reasonable accuracy while assigning excessive confidence to incorrect predictions under distribution shift.

## Methodology

### Models and training regimes

The analysis uses the completed SSL label-efficiency checkpoint sweep:

| Dimension | Values |
|---|---|
| Label fractions | 1%, 5%, 10%, 25%, 100% |
| Initialization | Random, SimCLR, VICReg |
| Seeds | 2026-2030 |
| Checkpoints | 75 |

### Calibration fitting

For each checkpoint:

1. The best saved checkpoint was re-evaluated on the shared clean validation split.
2. Original logits and probabilities were exported.
3. One positive scalar temperature was fitted by minimizing validation NLL.
4. The fitted temperature was frozen.
5. The same temperature was applied to clean, mild, moderate, and severe held-out artifacts.

The validation split contains 1,400 examples. Regenerated validation accuracy was required to match the checkpoint metadata exactly before fitting.

### Held-out evaluation

Each of the 75 fitted temperatures was transferred to four held-out conditions, producing 300 evaluation records.

The following metrics were measured before and after scaling:

- Accuracy
- Mean confidence
- Expected calibration error
- Maximum calibration error
- Negative log-likelihood
- Multiclass Brier score
- Selective accuracy at retained coverages of 50%, 80%, 90%, 95%, and 100%

A positive metric change means the calibrated result is larger than the baseline. For NLL, ECE, MCE, and Brier score, lower values are better.

## Integrity checks

All generated artifacts passed the following checks:

- 75/75 validation calibration artifacts present
- 75/75 temperature summaries present
- 300/300 held-out calibration artifacts present
- All held-out artifacts contain 1,400 examples
- Class order is consistent: BPSK, QPSK, 8PSK, 16QAM
- Stored predictions match logit argmax
- Every fitted temperature is positive and finite
- Validation NLL never increases because the fitting workflow falls back to `T = 1` when necessary
- Classification accuracy is preserved in 300/300 held-out evaluations

Fitted temperatures range from 0.374018 to 2.235871.

## Aggregate transfer results by channel condition

| Held-out condition | Mean NLL change | Mean ECE change | Mean Brier change | NLL improved | ECE improved |
|---|---:|---:|---:|---:|---:|
| Clean | -0.03001 | -0.03100 | -0.00837 | 71/75 | 71/75 |
| Mild multipath | +0.16737 | +0.00916 | +0.00056 | 16/75 | 16/75 |
| Moderate multipath | +0.92714 | +0.02241 | +0.01541 | 16/75 | 16/75 |
| Severe multipath | +2.31736 | +0.01144 | +0.01760 | 16/75 | 16/75 |

The clean result is consistently positive: calibration reduces all three proper scoring metrics on average and improves NLL and ECE in 94.7% of models.

The transfer result reverses under multipath. Mean NLL degradation grows from +0.167 under mild multipath to +2.317 under severe multipath. Only 16 of 75 models improve under each shifted condition.

![Held-out NLL transfer](figures/ssl_calibration_nll_transfer_v1.png)

![Held-out ECE transfer](figures/ssl_calibration_ece_transfer_v1.png)

## Aggregate transfer results by labeled-data fraction

| Label fraction | Mean NLL change | Mean ECE change | Mean Brier change | NLL improved | ECE improved |
|---|---:|---:|---:|---:|---:|
| 1% | -0.45053 | -0.05921 | -0.04271 | 56/60 | 56/60 |
| 5% | +0.50902 | +0.01250 | +0.01179 | 16/60 | 17/60 |
| 10% | +0.97762 | +0.01547 | +0.01611 | 17/60 | 16/60 |
| 25% | +1.54486 | +0.02136 | +0.02177 | 15/60 | 15/60 |
| 100% | +1.64634 | +0.02488 | +0.02454 | 15/60 | 15/60 |

The 1% label regime behaves differently from the other regimes. It improves in 56 of 60 evaluations and has negative average changes for NLL, ECE, and Brier score.

From 5% labels onward, the average transfer effect is harmful. The mean NLL degradation increases with the label fraction and reaches +1.646 at 100% labels.

## Interpretation of fitted temperatures

A temperature above 1 softens probabilities, while a temperature below 1 sharpens them.

The low-label models frequently require softening. This is consistent with the strong improvement observed at 1% labels across both clean and shifted conditions.

Higher-label models frequently receive temperatures below 1. These temperatures sharpen already confident predictions. On clean data this can correct underconfidence, but under multipath it amplifies confidence in incorrect predictions.

![Validation-fitted temperatures](figures/ssl_calibration_temperatures_v1.png)

## Representative reliability behavior

For the 100%-label random-initialization checkpoint with seed 2026:

| Condition | Metric | Baseline | Temperature-scaled |
|---|---|---:|---:|
| Clean | ECE | 0.033 | 0.016 |
| Clean | NLL | 0.109 | 0.087 |
| Severe multipath | ECE | 0.525 | 0.558 |
| Severe multipath | NLL | 4.551 | 8.257 |

The same fitted temperature improves the clean reliability profile but substantially worsens severe-multipath confidence. The severe result demonstrates that a low clean ECE does not imply confidence robustness under channel shift.

![Representative reliability diagrams](figures/ssl_calibration_reliability_examples_v1.png)

## Selective classification

Selective classification retains only the most confident examples and rejects the remainder. Temperature scaling produced only very small changes in selective accuracy.

Across all models, the mean selective-accuracy changes were generally slightly negative. The largest condition-level average reduction was approximately 0.00162 in accuracy, equivalent to 0.162 percentage points.

This indicates that scalar temperature scaling changes probability magnitudes but does not create a meaningfully better confidence ranking for rejection decisions.

![Selective accuracy changes](figures/ssl_calibration_selective_accuracy_v1.png)

## Main findings

### 1. Clean calibration succeeds

Temperature scaling is effective on held-out data that resembles the clean validation distribution. NLL, ECE, and Brier score all improve on average, and NLL/ECE improve for 71 of 75 checkpoints.

### 2. Multipath transfer fails systematically

The validation-fitted temperature does not remain reliable when the channel changes. Confidence quality worsens for most models under mild, moderate, and severe multipath.

### 3. Failure severity increases with channel shift

Mean NLL degradation rises monotonically with multipath severity:

- Mild: +0.16737
- Moderate: +0.92714
- Severe: +2.31736

### 4. The 1% label regime is an exception

At 1% labels, temperature scaling improves NLL and ECE in 56 of 60 evaluations. These models appear sufficiently overconfident that probability softening remains beneficial across the tested conditions.

### 5. More labeled data does not guarantee safer confidence

Higher-label models are more accurate, but their clean-fitted temperatures can sharpen probabilities and produce worse confidence under shift. Accuracy and calibration robustness are therefore separate properties.

### 6. Accuracy preservation is not calibration robustness

Argmax predictions and classification accuracy remain unchanged by positive scalar temperature scaling. Nevertheless, NLL can degrade severely because incorrect predictions become more confident.

### 7. Selective prediction gains are negligible

The calibrated confidence scores do not provide a reliably better ordering of examples for confidence-based rejection.

## Deployment implications

A clean-validation temperature should not be used as the sole confidence correction in a channel-varying deployment.

Recommended deployment behavior:

1. Preserve both calibrated and uncalibrated confidence during evaluation and logging.
2. Treat the clean-fitted temperature as valid only for conditions close to the calibration distribution.
3. Detect channel or distribution shift before trusting calibrated probabilities.
4. Avoid fixed confidence thresholds derived only from clean validation.
5. Evaluate rejection policies on representative shifted conditions.
6. Consider condition-aware or adaptive calibration.
7. Include calibration metrics in deployment monitoring, not only classification accuracy.

Potential follow-up methods include:

- Per-condition temperature scaling
- SNR-aware calibration
- Multipath-severity-aware calibration
- Vector or matrix scaling
- Calibration models conditioned on learned channel features
- Online recalibration with bounded adaptation
- Ensemble uncertainty
- Out-of-distribution or channel-shift detection

## Limitations

### Validation reuse

The same validation split was used for checkpoint selection and temperature fitting. This is practical for the current engineering milestone but can produce optimistic in-distribution calibration estimates. A separate calibration split would provide a cleaner statistical evaluation.

### Scalar calibration capacity

One scalar temperature cannot correct class-specific, SNR-specific, or channel-specific miscalibration.

### Binned calibration metrics

ECE and MCE depend on the selected bin count and may be unstable for sparsely populated bins. NLL and Brier score should be treated as primary proper scoring rules, with ECE and MCE as complementary diagnostics.

### Synthetic condition coverage

The conclusions apply to the tested clean and multipath distributions. Additional hardware captures, interference profiles, frequency offsets, timing offsets, and unseen channel families should be evaluated before deployment claims are broadened.

### Selective-risk scope

The analysis uses confidence ranking from maximum class probability. Other uncertainty scores or learned rejection models may behave differently.

## Reproducibility

### Primary configurations

- `configs/fit_ssl_validation_temperatures_v1.yaml`
- `configs/backfill_ssl_calibration_predictions_v1.yaml`
- `configs/analyze_ssl_calibration_v1.yaml`
- `configs/visualize_ssl_calibration_v1.yaml`

### Primary commands

```powershell
python scripts\fit_ssl_validation_temperatures.py `
  --config configs\fit_ssl_validation_temperatures_v1.yaml

python scripts\backfill_ssl_calibration_predictions.py `
  --config configs\backfill_ssl_calibration_predictions_v1.yaml

python scripts\analyze_ssl_calibration.py `
  --config configs\analyze_ssl_calibration_v1.yaml

python scripts\visualize_ssl_calibration.py `
  --config configs\visualize_ssl_calibration_v1.yaml
```

### Main generated analysis artifact

- `results/ssl_calibration_analysis_v1/aggregate_metrics.json`

### Figures

- `reports/figures/ssl_calibration_nll_transfer_v1.png`
- `reports/figures/ssl_calibration_ece_transfer_v1.png`
- `reports/figures/ssl_calibration_temperatures_v1.png`
- `reports/figures/ssl_calibration_selective_accuracy_v1.png`
- `reports/figures/ssl_calibration_reliability_examples_v1.png`

### Validation status

At completion of the visualization unit:

- Ruff passed across `src`, `tests`, and `scripts`
- 840 tests passed
- All five figure files were generated and verified as non-empty

## Conclusion

Scalar temperature scaling fitted on clean validation data is useful for in-distribution confidence calibration but is not robust to channel distribution shift.

The experiment shows a clear separation between clean calibration quality and deployment reliability. For most models with at least 5% labeled data, clean-fitted temperatures sharpen predictions and improve clean NLL/ECE, yet strongly worsen confidence under multipath. The 1% label regime is the main exception because probability softening remains broadly beneficial.

The practical conclusion is:

> Validation-fitted scalar temperature scaling should be treated as an in-distribution correction, not as a universal deployment-confidence solution. Reliable RF deployment requires channel-aware calibration, explicit shift detection, or both.
