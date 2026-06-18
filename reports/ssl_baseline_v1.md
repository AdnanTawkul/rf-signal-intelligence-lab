# Self-Supervised Learning Baseline v1

## Overview

This report documents the first self-supervised learning (SSL) baseline for the RF Signal Intelligence Lab. It compares SimCLR and VICReg against the selected supervised GroupNorm CNN and asks three questions:

1. Do frozen SSL embeddings separate BPSK, QPSK, 8PSK, and 16QAM?
2. Does SSL initialization improve full-data supervised training?
3. Does SSL initialization improve performance when only 280 labeled examples are available?

All SSL runs use the same GroupNorm CNN encoder with channels `(32, 64, 128)`, kernel size `7`, eight GroupNorm groups, and a 128-dimensional pooled embedding. SSL pretraining uses 5,600 training examples and 1,400 validation examples without labels.

For supervised fine-tuning, only encoder and normalization parameters are imported from SSL checkpoints. The four-class classifier is freshly initialized, and model selection remains validation-based.

## SimCLR

SimCLR was trained for 50 epochs with NT-Xent loss, batch size 256, learning rate `3e-4`, weight decay `1e-4`, and temperature `0.1`. The augmentation pipeline included phase rotation, amplitude scaling, time shifting, and additional AWGN.

The selected checkpoint was epoch 47:

- Best validation loss: `1.0415`
- Validation positive-pair cosine similarity: `0.9545`
- Validation projection standard deviation: `0.0855`

The representation did not collapse, but frozen downstream performance was limited:

| Probe | Validation accuracy |
|---|---:|
| Linear probe | 74.21% |
| Nonlinear MLP probe | 74.50% |

The MLP improved by only 0.29 percentage points, indicating that the main issue is not merely linear separability.

The SimCLR linear probe was evaluated once on the held-out test split:

- Overall: 74.86%
- BPSK: 99.14%
- QPSK: 47.14%
- 8PSK: 58.00%
- 16QAM: 95.14%

The dominant failure mode was QPSK/8PSK confusion.

## VICReg

VICReg used the same encoder and data protocol. Its loss weights were invariance `25`, variance `25`, and covariance `1`. It was trained for 50 epochs.

The selected checkpoint was epoch 50:

- Validation total loss: `26.6120`
- Invariance loss: `0.0020`
- Variance loss: `0.7723`
- Covariance loss: `7.2524`
- Positive-pair cosine similarity: `0.9975`
- Projection standard deviation: `0.6138`

The representation did not collapse, but frozen performance was weaker than SimCLR:

| Probe | Validation accuracy |
|---|---:|
| Linear probe | 67.21% |
| Nonlinear MLP probe | 67.21% |

The nonlinear probe produced no improvement.

## Full-data supervised fine-tuning

The SSL encoders were used to initialize the supervised GroupNorm CNN while keeping a freshly initialized classifier.

| Initialization | Best validation accuracy | Best epoch | Change vs. random |
|---|---:|---:|---:|
| Random | 96.43% | 30 | — |
| SimCLR | 93.86% | 27 | -2.57 pp |
| VICReg | 94.07% | 21 | -2.36 pp |

Neither SSL initialization improved full-data supervised training.

## Low-label protocol

The low-label subset is selected jointly by modulation class and SNR:

- 4 classes
- 7 SNR levels
- 10 examples per class-SNR stratum
- 280 labeled training examples
- Full 1,400-example validation split

With batch size 256, the 280-example subset produces two optimizer updates per epoch. The low-label runs therefore use 330 epochs, approximately matching the 660-update budget of the 30-epoch full-data baseline.

For every paired seed, random, SimCLR, and VICReg use the same training seed and subset-selection seed.

## Five-seed paired low-label results

Seeds: `2026`, `2027`, `2028`, `2029`, and `2030`.

| Initialization | Mean validation accuracy | Standard deviation |
|---|---:|---:|
| Random | 92.81% | 0.68% |
| SimCLR | 91.43% | 0.78% |
| VICReg | 92.91% | 0.88% |

Paired changes relative to random initialization:

| Initialization | Mean paired change | Standard deviation | Seeds improved |
|---|---:|---:|---:|
| SimCLR | -1.39 pp | 0.75 pp | 0/5 |
| VICReg | +0.10 pp | 0.95 pp | 2/5 |

## Interpretation

### SimCLR

SimCLR is consistently harmful under the current low-label protocol. It underperformed random initialization for all five paired seeds, with an average reduction of 1.39 percentage points. Together with the weak frozen probes, this shows that the current contrastive objective and augmentation design do not preserve enough modulation-discriminative information.

### VICReg

VICReg is practically equivalent to random initialization. Its mean advantage is only 0.10 percentage points, much smaller than the paired seed variation, and it improved only two of five seeds. The earlier single-seed advantage was therefore not reproducible.

## Final decision

1. The supervised GroupNorm CNN remains the selected full-data baseline.
2. Random initialization remains the selected low-label supervised baseline.
3. SimCLR is rejected for the current architecture and augmentation design.
4. VICReg is considered practically neutral relative to random initialization.
5. Neither SSL method is promoted as a selected model.
6. Rejected SSL fine-tuning candidates are not evaluated on the test split.

## Engineering outcomes

The SSL branch adds reusable infrastructure for:

- SimCLR and VICReg pretraining
- RF-specific augmentations
- Frozen linear and nonlinear probes
- Safe encoder-only checkpoint initialization
- Fresh supervised classifier initialization
- Deterministic class-SNR stratified subsets
- Optimizer-step-matched low-label comparisons
- Paired five-seed evaluation
- Reproducible checkpoints, summaries, histories, and figures

## Reproduction

SimCLR pretraining:

```powershell
python scripts\pretrain_simclr.py `
  --config configs\pretrain_simclr_v1.yaml
```

VICReg pretraining:

```powershell
python scripts\pretrain_vicreg.py `
  --config configs\pretrain_vicreg_v1.yaml
```

Paired low-label sweep:

```powershell
python scripts\run_low_label_ssl_seed_sweep.py `
  --config configs\low_label_ssl_step_matched_seed_sweep_v1.yaml
```

Aggregate results:

```text
results/low_label_ssl_step_matched_seed_sweep_v1/aggregate.json
```

## Limitations and future work

These conclusions are specific to the current synthetic dataset, four modulation classes, GroupNorm CNN, augmentation policies, projection heads, loss hyperparameters, and 280-label protocol.

Potential future directions include phase-aware positive-pair construction, weaker or conditional phase augmentation, time-frequency or cyclostationary representations, supervised contrastive learning, semi-supervised fine-tuning, larger encoders, and more realistic channel or over-the-air data.
