# Streamlit Demo v1

## Summary

This report documents the first complete local Streamlit interface for the RF Signal Intelligence Lab. The demo converts the project from a collection of command-line research and deployment components into an interactive local analysis tool for raw I/Q examples.

The demo supports three workflows:

1. Single-window modulation recognition.
2. Single-window IQ channel-shift assessment.
3. Long-signal or pre-windowed-batch analysis with per-window timelines.

The implementation is part of the `feature/streamlit-demo` branch and was validated after the long-analysis commit `1cab1b4`.

## Goals

The Streamlit milestone had four goals:

- Make the trained modulation classifier usable from a local browser interface.
- Show signal-domain evidence, not only a class label.
- Surface channel-shift diagnostics in a form that is understandable during inference.
- Export path-safe JSON and CSV artifacts that can be shared without leaking local filesystem paths.

The app is intentionally local. It does not require cloud services, paid APIs, external inference endpoints, or the OpenAI API.

## User Interface

The app is launched from the repository root:

```powershell
python -m streamlit run app.py
```

The default configuration file is:

```text
configs/streamlit_demo_v1.yaml
```

The configuration selects the default checkpoint, expected sample count, input scale, device mode, top-k display count, visualization settings, long-analysis defaults, and exported IQ shift-detector artifact.

## Single-Window Classification Workflow

The first workflow accepts `.npy` and `.npz` IQ inputs. It supports both two-channel real IQ tensors and NPZ datasets with metadata. For the validation batch example, the app detected:

| Field | Value |
|---|---:|
| Windows | 1,400 |
| Channels | 2 |
| Samples per window | 2,048 |
| Selected window position | 0 |
| Ground-truth class index | 2 |
| SNR | 4.00 dB |

The signal overview shows:

- I/Q waveform versus time
- I/Q constellation
- Power spectrum

![Streamlit single-window signal overview](figures/streamlit_demo_single_window_v1.png)

For the example validation window, the model predicted:

| Metric | Value |
|---|---:|
| Predicted modulation | 8psk |
| Confidence | 68.38% |
| Sample index | 0 |
| Ground truth | 8psk |
| Correct | true |

The GUI displays the full top-k probability table and provides both:

- `rf_iq_prediction.json`
- `rf_iq_prediction_probabilities.csv`

The exported JSON stores only the public source name `validation.npz` and a relative checkpoint reference.

## Channel-Shift Assessment Workflow

The second workflow runs the exported all-IQ linear detector over the selected IQ window. The detector uses 21 deterministic IQ features and a development-selected high-recall operating point.

The displayed warning explicitly states that the detector score is not a probability and not a channel-severity estimate.

![Streamlit channel-shift assessment](figures/streamlit_demo_shift_assessment_v1.png)

For the example validation window, the detector returned:

| Metric | Value |
|---|---:|
| Detector | all_iq_linear_shift_detector_v1 |
| Shift score | -0.3895 |
| Threshold | -0.4219 |
| Score margin | +0.0324 |
| Development AUROC | 0.9176 |
| Decision rule | score >= threshold |
| Decision | shift-like |

The operating point targets 95% shift recall and produced 55.3% false positives on clean development windows. The interface therefore treats one flag as a warning rather than proof of channel degradation.

The feature-contribution table highlights the largest absolute standardized linear-model contributions. The strongest terms for the example were:

| Rank | Feature | Contribution |
|---:|---|---:|
| 1 | autocorrelation_abs_lag_8 | -0.2335 |
| 2 | dphase_mean_abs_normalized | -0.1654 |
| 3 | amplitude_skewness | +0.0203 |
| 4 | autocorrelation_abs_lag_4 | -0.0171 |
| 5 | spectral_peak_fraction | -0.0113 |

The GUI exports:

- `rf_iq_shift_assessment.json`
- `rf_iq_shift_feature_contributions.csv`

Both exports are path-safe.

## Long-Signal and Pre-Windowed Batch Workflow

The third workflow analyzes either one continuous long IQ recording or a pre-windowed batch. The app detects the input layout automatically.

Supported layouts include:

| Layout | Interpretation |
|---|---|
| `[batch, 2, 2048]` | Pre-windowed batch |
| `[1, 2, N]`, with `N > 2048` | Continuous long signal |
| `[2, N]` | Continuous long signal |

For a continuous long signal, the app windows the signal with the configured stride and remainder policy. For a pre-windowed dataset batch, the app treats each row as an independent analysis window and uses window order as the horizontal axis.

![Streamlit long-signal and batch analysis](figures/streamlit_demo_long_analysis_v1.png)

The validation demonstration used the pre-windowed `validation.npz` batch. To keep the UI responsive, the run analyzed the first 256 windows from a 1,400-window batch.

| Metric | Value |
|---|---:|
| Source mode | prewindowed_batch |
| Source item count | 1,400 |
| Analyzed windows | 256 |
| Truncated | true |
| Window size | 2,048 |
| Stride setting | 1,024 |
| Remainder policy | drop |
| Aggregate prediction | qpsk |
| Aggregate confidence | 31.98% |
| Shift-like windows | 138 / 256 |
| Shift-like fraction | 53.91% |
| Mean window confidence | 74.82% |

The aggregate probability table was:

| Class | Probability |
|---|---:|
| bpsk | 0.2609 |
| qpsk | 0.3198 |
| 8psk | 0.2360 |
| 16qam | 0.1833 |

The long-analysis workflow displays:

- aggregate probability bar chart
- aggregate probability table
- per-window confidence timeline
- shift-score timeline with threshold
- predicted-class timeline
- complete per-window results table

It exports:

- `rf_long_iq_analysis.json`
- `rf_long_iq_aggregate_probabilities.csv`
- `rf_long_iq_windows.csv`

The JSON contains a compact summary plus per-window predictions, probabilities, true labels when present, SNR values when present, shift scores, and shift decisions. The CSV export makes the per-window table suitable for spreadsheet review or external plotting.

## Export Contract

The demo exports both machine-readable JSON and table-oriented CSV files.

| File | Purpose |
|---|---|
| `rf_iq_prediction.json` | Single-window model prediction |
| `rf_iq_prediction_probabilities.csv` | Single-window top-k/probability table |
| `rf_iq_shift_assessment.json` | Single-window shift assessment |
| `rf_iq_shift_feature_contributions.csv` | Shift feature-contribution table |
| `rf_long_iq_analysis.json` | Long-signal or batch analysis summary and window records |
| `rf_long_iq_aggregate_probabilities.csv` | Long-analysis aggregate probability table |
| `rf_long_iq_windows.csv` | Long-analysis per-window table |

Path-safety validation checks confirmed that exported GUI artifacts do not include local absolute Windows paths such as `G:\...`.

## Implementation Notes

The demo code is organized as:

| File | Role |
|---|---|
| `app.py` | Streamlit application entry point |
| `configs/streamlit_demo_v1.yaml` | Demo configuration |
| `src/rfsil/demo/application.py` | Configuration loading and validation |
| `src/rfsil/demo/shift_service.py` | GUI-ready shift-assessment service |
| `src/rfsil/demo/long_signal_service.py` | GUI-ready long-analysis service |
| `tests/test_demo_application.py` | Demo configuration tests |
| `tests/test_demo_shift_service.py` | Shift-service tests |
| `tests/test_demo_long_signal_service.py` | Long-analysis service tests |

The implementation reuses existing deployment components rather than duplicating inference logic:

- `IQInferenceEngine.from_checkpoint`
- `load_iq_file`
- `predict_window_batches`
- `window_iq_signal`
- `aggregate_window_predictions`
- `IQShiftDetectorArtifact.assess_iq`

This keeps the GUI aligned with the command-line deployment path.

## Validation

The final validation before committing the long-analysis GUI unit passed:

| Check | Result |
|---|---|
| Python compile check for app and demo modules | Passed |
| Ruff static analysis over `src`, `tests`, `scripts`, and `app.py` | Passed |
| Focused GUI tests | 26 passed |
| Full regression suite | 1,091 passed |
| `git diff --check` | Passed |
| GUI export path-safety checks | Passed |

The long-analysis unit was committed as:

```text
1cab1b4 Add long IQ analysis to Streamlit demo
```

The earlier channel-shift GUI unit was committed as:

```text
9dbd909 Add channel shift assessment to Streamlit demo
```

## Interpretation Guidance

The GUI is intended for engineering inspection, model demonstration, and portfolio communication. It should not be interpreted as an operational RF monitoring system.

Important interpretation rules:

- The modulation confidence is the model's softmax confidence, not a calibrated posterior under all channel conditions.
- The channel-shift score is a binary detector score, not a probability.
- A shift-like flag is a warning that the IQ structure resembles the selected shifted development condition.
- Mild multipath remains difficult at high recall, so false-positive rates can be high.
- The long-analysis aggregate prediction over a mixed validation batch is not a single real transmission label. It summarizes the first analyzed windows.
- For pre-windowed dataset batches, the horizontal axis is window order, not continuous time.

## Limitations

- The app is local and not hardened as a hosted web service.
- The default long-analysis configuration caps the number of analyzed windows to preserve interactivity.
- The GUI currently uses static Matplotlib and Streamlit charts rather than a custom front-end.
- The current IQ shift detector was validated on 2,048-sample windows.
- Transfer to different sample rates, receivers, and over-the-air data remains future work.
- The demo does not yet include ONNX, TensorRT, quantized inference, or a persistent local inference service.
- Channel-aware calibration based on detected shift is not yet implemented.

## Next Work

Recommended next steps are:

1. Finalize README references and screenshots.
2. Open and merge the Streamlit demo pull request.
3. Tag a `v1.1.0` release.
4. Add channel-aware calibration experiments.
5. Add ONNX export and optimized inference.
6. Add persistent local inference service support.
7. Evaluate on over-the-air receiver data.
