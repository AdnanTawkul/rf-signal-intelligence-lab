# RF Signal Intelligence Lab

A local, reproducible AI engineering project for RF modulation recognition from synthetic raw IQ signals.

The repository covers the full experimental path from signal generation and channel simulation to PyTorch training, held-out evaluation, multi-seed reproducibility studies, architecture ablations, representation analysis, classifier-head refitting, and deployment-oriented extensions. It is designed as a serious portfolio project for AI, RF, signal-processing, and applied ML engineering roles.

## Current Status

The project currently includes:

- Synthetic BPSK, QPSK, 8PSK, and 16QAM generation
- Root-raised-cosine pulse shaping
- AWGN with controlled SNR
- Carrier frequency and phase offsets
- Amplitude scaling
- Zero-padded timing shifts
- Flat Rayleigh block fading
- IQ, constellation, waveform, and spectrogram visualizations
- Reproducible balanced dataset generation
- PyTorch `Dataset` and `DataLoader` integration
- A compact one-dimensional CNN classifier
- Configurable BatchNorm and GroupNorm support
- Optional per-example RMS IQ normalization
- Public extraction of the 128-dimensional pooled CNN embedding
- Validation-selected frozen linear-head refitting
- Conversion of standardized logistic-regression parameters into a native PyTorch `Linear` head
- Native checkpoint export with no scikit-learn dependency at inference time
- GPU training on an NVIDIA RTX 4080 SUPER
- Held-out evaluation by class and SNR
- Confusion-matrix and class-by-SNR error analysis
- Five-seed training, refit, and test reproducibility studies
- Frequency-selective tapped-delay-line multipath simulation
- Reusable clean, mild, moderate, and severe channel profiles
- Balanced mixed-multipath supervised training
- Jointly trained learnable residual I/Q front end
- Frozen-backbone residual adaptation with 2,944 trainable parameters
- Paired five-seed evaluation across four channel conditions
- Correction-magnitude and pooled-confusion analysis
- RMS-normalization and GroupNorm ablations
- Automated tests with `pytest`
- Static analysis with Ruff

For the original clean-channel benchmark, the selected system remains the GroupNorm CNN with a validation-selected frozen linear-head refit. For frequency-selective multipath, the selected maximum-robustness model uses a jointly trained residual signal front end, while a frozen-backbone variant provides parameter-efficient adaptation by training only 2,944 parameters. The next major research milestones are self-supervised representation learning, uncertainty calibration, public-dataset validation, ONNX export, local latency benchmarking, and a Streamlit demo.

## Selected Supervised Baseline

The statistically defensible selected result is:

> **96.50% ± 0.26 percentage points held-out test accuracy across five independent training seeds**

Five independently trained GroupNorm checkpoints were refitted independently. For each seed, the CNN encoder was frozen, logistic-regression regularization was selected on the validation split, and the resulting standardized classifier was converted into an equivalent native PyTorch linear layer before evaluation on the untouched test split.

| Metric | Result |
|---|---:|
| Mean validation accuracy after head refit | 96.26% |
| Validation standard deviation | 0.21 percentage points |
| Mean held-out test accuracy | 96.50% |
| Test standard deviation | 0.26 percentage points |
| Minimum test accuracy | 96.14% |
| Maximum test accuracy | 96.93% |
| Mean accuracy at -4 dB | 78.40% |
| Mean accuracy at 0 dB | 97.10% |
| Test examples per run | 1,400 |

### Individual Held-Out Test Results

| Seed | Test accuracy |
|---:|---:|
| 2026 | 96.93% |
| 2027 | 96.43% |
| 2028 | 96.43% |
| 2029 | 96.57% |
| 2030 | 96.14% |

### Mean Per-Class Accuracy Across Five Seeds

| Modulation | Mean accuracy | Standard deviation |
|---|---:|---:|
| BPSK | 99.94% | 0.11 percentage points |
| QPSK | 93.14% | 1.02 percentage points |
| 8PSK | 93.94% | 0.42 percentage points |
| 16QAM | 98.97% | 0.29 percentage points |

### Mean Accuracy by SNR Across Five Seeds

| SNR | Mean accuracy | Standard deviation |
|---:|---:|---:|
| -4 dB | 78.40% | 1.39 percentage points |
| 0 dB | 97.10% | 0.73 percentage points |
| 4 dB | 100.00% | 0.00 percentage points |
| 8 dB | 100.00% | 0.00 percentage points |
| 12 dB | 100.00% | 0.00 percentage points |
| 16 dB | 100.00% | 0.00 percentage points |
| 20 dB | 100.00% | 0.00 percentage points |

![GroupNorm frozen-head-refit five-seed held-out evaluation](reports/figures/baseline_cnn_groupnorm_head_refit_seed_sweep_v1_test.png)

Detailed methodology, BatchNorm results, RMS-normalization ablation, GroupNorm comparison, class-by-SNR diagnosis, frozen-embedding experiments, and five-seed head-refit evidence are documented in [Baseline CNN v1 Results](reports/baseline_cnn_v1.md).

## Selected Multipath-Robust Models

The multipath benchmark uses a balanced training distribution containing 25% clean, 25% mild, 25% moderate, and 25% severe channel examples.

| Model | Clean | Mild | Moderate | Severe |
|---|---:|---:|---:|---:|
| Mixed-I/Q baseline | 93.34% | 90.10% | 78.93% | 56.64% |
| Joint residual front end | 93.23% | 91.69% | **84.97%** | **65.59%** |
| Frozen-backbone residual front end | **94.21%** | **91.71%** | 84.67% | 63.84% |

The jointly trained residual front end is the selected **maximum-robustness model**. Relative to the mixed-I/Q baseline, it improves moderate accuracy by 6.04 percentage points and severe accuracy by 8.94 percentage points.

The frozen-backbone variant is the selected **parameter-efficient adaptation model**. It freezes the existing 73,092-parameter classifier and trains only a 2,944-parameter signal front end. It improves clean, mild, moderate, and severe accuracy on all five paired seeds.

Both approaches substantially reduce the dominant QPSK/8PSK-to-16QAM failure pathway under frequency-selective multipath.

![Mixed-IQ baseline versus jointly trained residual front end](reports/figures/residual_equalizer_comparison_v1.png)

![Mixed-IQ baseline versus frozen-backbone residual front end](reports/figures/frozen_backbone_equalizer_comparison_v1.png)

Detailed architecture, training protocols, correction analysis, paired-seed results, SNR analysis, and pooled confusion matrices are documented in [Learnable Residual Signal Front End v1](reports/learnable_residual_front_end_v1.md).

## Why the Frozen Head Refit Was Selected

The earlier GroupNorm baseline achieved:

```text
95.29% ± 0.45 percentage points
```

The frozen linear-head refit improved the five-seed held-out result to:

```text
96.50% ± 0.26 percentage points
```

Compared with the original GroupNorm classifier head, the refit:

- Increased mean validation accuracy by 0.61 percentage points
- Improved validation accuracy for all five seeds
- Increased mean held-out test accuracy by 1.21 percentage points
- Improved held-out test accuracy for all five seeds
- Reduced test standard deviation by 0.19 percentage points
- Increased the minimum test accuracy by 1.43 percentage points
- Increased mean QPSK accuracy by 0.91 percentage points
- Increased mean 8PSK accuracy by 4.00 percentage points
- Increased mean accuracy at -4 dB by 3.50 percentage points
- Increased mean accuracy at 0 dB by 3.70 percentage points

The worst refitted test run, 96.14%, exceeded the best original GroupNorm run, 96.07%.

The refit does not change the CNN architecture or parameter count. It replaces only the final `Linear(128, 4)` decision boundary, and the saved checkpoint runs entirely in PyTorch.

## Signal and Dataset Pipeline

Each synthetic classification example follows this pipeline:

1. Random modulation-symbol generation
2. Root-raised-cosine transmit pulse shaping
3. Fixed-length waveform extraction
4. Amplitude scaling
5. Optional flat Rayleigh fading
6. Optional frequency-selective tapped-delay-line multipath
7. Carrier frequency offset
8. Carrier phase offset
9. Zero-padded integer timing shift
10. AWGN at the configured SNR
11. Conversion to a two-channel tensor

The model input format is:

```text
[batch, 2, samples]
```

- Channel 0: in-phase component
- Channel 1: quadrature component
- Default baseline sample length: 2,048 samples

The generated dataset stores labels and impairment metadata, including SNR, frequency offset, phase offset, amplitude scale, timing shift, fading state, multipath condition, and generation seed.

## Selected Clean-Channel Model

The selected clean-channel system is a compact one-dimensional CNN encoder with GroupNorm and a validation-selected linear classifier head.

```text
Input: [batch, 2, 2048]

Conv block: 2 → 32
Conv block: 32 → 64
Conv block: 64 → 128
Adaptive global average pooling
128-dimensional embedding
Dropout
Linear classifier: 128 → 4
```

Each convolutional block contains:

- `Conv1d`
- GroupNorm with 8 groups
- GELU activation
- Max pooling

Trainable parameters:

```text
73,092
```

The frozen-head procedure keeps the trained CNN encoder fixed, extracts the pooled 128-dimensional embedding, fits a standardized multinomial logistic-regression classifier on training embeddings, selects regularization using validation accuracy, converts the selected classifier into raw-feature PyTorch weights, and writes those weights into the existing final linear layer.

RMS input normalization remains implemented as an optional checkpointed setting, but it is disabled in the selected model because the five-seed ablation reduced mean test accuracy and increased variance.

## Frozen Head Refit Protocol

For every seed:

1. Train the GroupNorm CNN using the normal 30-epoch training pipeline.
2. Select the checkpoint with the highest validation accuracy.
3. Freeze the CNN encoder.
4. Extract training and validation embeddings with `model.extract_features(iq)`.
5. Fit `StandardScaler + LogisticRegression` candidates using only training embeddings.
6. Select the regularization value `C` using validation accuracy.
7. Break validation ties in favor of the smaller `C`, corresponding to stronger regularization.
8. Convert the standardized classifier to equivalent raw-embedding parameters.
9. Replace the checkpoint's final PyTorch linear layer.
10. Evaluate the resulting native checkpoint once on the held-out test split.

Candidate regularization values:

```text
0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0
```

Selected values:

| Seed | Selected C |
|---:|---:|
| 2026 | 1.0 |
| 2027 | 10.0 |
| 2028 | 0.1 |
| 2029 | 3.0 |
| 2030 | 1.0 |

For standardized embeddings

```text
z = (x - mean) / scale
```

the equivalent raw-feature linear layer is:

```text
W_raw = W_standardized / scale
b_raw = b_standardized - W_raw @ mean
```

This conversion allows the refitted classifier to be stored and deployed as a normal PyTorch checkpoint without requiring scikit-learn during inference.

## Reproducibility Findings

### GroupNorm Encoder Validation Performance

Best-checkpoint validation accuracy before head refitting:

```text
95.64% ± 0.40 percentage points
```

Final-epoch validation accuracy before head refitting:

```text
94.10% ± 2.34 percentage points
```

### Validation-Selected Head Refit

Validation accuracy after independently refitting each seed:

```text
96.26% ± 0.21 percentage points
```

| Seed | Original validation | Refitted validation | Change |
|---:|---:|---:|---:|
| 2026 | 96.43% | 96.64% | +0.21 pp |
| 2027 | 95.50% | 96.29% | +0.79 pp |
| 2028 | 95.43% | 96.14% | +0.71 pp |
| 2029 | 95.29% | 96.14% | +0.86 pp |
| 2030 | 95.57% | 96.07% | +0.50 pp |

Every seed improved on validation, and validation variation decreased.

### BatchNorm Comparison

The earlier BatchNorm model produced:

```text
Best validation: 94.17% ± 0.31 percentage points
Final validation: 88.39% ± 4.13 percentage points
Held-out test: 94.24% ± 0.29 percentage points
```

GroupNorm substantially improved late-training behavior and reduced dependence on unstable BatchNorm running statistics. The frozen-head refit then improved the learned decision boundary without modifying the encoder.

## RMS Normalization Ablation

Per-example complex RMS normalization was tested without changing the dataset, architecture, optimizer, batch size, epoch count, or seeds.

Its single-seed result looked promising, but the five-seed result was worse:

| Metric | BatchNorm baseline | RMS-normalized |
|---|---:|---:|
| Mean test accuracy | 94.24% | 93.93% |
| Test standard deviation | 0.29 pp | 0.80 pp |
| Minimum test accuracy | 93.93% | 92.43% |
| Mean accuracy at -4 dB | 70.50% | 68.90% |

RMS normalization is therefore retained as an ablation and optional feature, not used as the default preprocessing path.

## Quick Start

### 1. Clone the repository

```powershell
git clone https://github.com/AdnanTawkul/rf-signal-intelligence-lab.git
cd rf-signal-intelligence-lab
```

### 2. Create and activate the virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Upgrade packaging tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install PyTorch with CUDA support

The development environment uses the official PyTorch CUDA 12.8 wheel channel:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install project dependencies

```powershell
python -m pip install -r requirements.txt
```

### 6. Verify the GPU

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA DEVICE')"
```

Expected hardware in the original development environment:

```text
NVIDIA GeForce RTX 4080 SUPER
```

## Reproduce the Selected Baseline

### Generate the baseline dataset

```powershell
python scripts\generate_dataset.py --config configs\dataset_baseline_v1.yaml
```

Expected split sizes:

| Split | Examples |
|---|---:|
| Train | 5,600 |
| Validation | 1,400 |
| Test | 1,400 |

Generated datasets are stored under `data/processed/` and are intentionally excluded from Git.

### Run the five-seed GroupNorm training study

```powershell
python scripts\run_baseline_seed_sweep.py --config configs\baseline_groupnorm_seed_sweep_v1.yaml
```

### Refit all five classifier heads

```powershell
python scripts\run_frozen_head_refit_seed_sweep.py --config configs\refit_groupnorm_head_seed_sweep_v1.yaml
```

### Evaluate all five native refitted checkpoints

```powershell
python scripts\evaluate_seed_sweep.py --config configs\evaluate_groupnorm_head_refit_seed_sweep_v1.yaml
```

### Refit and evaluate one checkpoint

```powershell
python scripts\refit_frozen_head.py --config configs\refit_groupnorm_head_v1.yaml
python scripts\evaluate_baseline.py --config configs\evaluate_groupnorm_head_refit_v1.yaml
python scripts\analyze_baseline_errors.py --config configs\analyze_groupnorm_head_refit_v1.yaml
```

## Reproduce the Multipath-Robust Models

The following commands assume that the clean and multipath datasets referenced by the configurations are available under `data/processed/`.

### Joint residual front end

Train all five seeds:

```powershell
python scripts\run_baseline_seed_sweep.py --config configs\baseline_groupnorm_residual_equalizer_seed_sweep_v1.yaml
```

Evaluate clean, mild, moderate, and severe conditions:

```powershell
$configs = @(
    "configs\evaluate_groupnorm_residual_equalizer_seed_sweep_v1.yaml",
    "configs\evaluate_groupnorm_residual_equalizer_mild_seed_sweep_v1.yaml",
    "configs\evaluate_groupnorm_residual_equalizer_moderate_seed_sweep_v1.yaml",
    "configs\evaluate_groupnorm_residual_equalizer_severe_seed_sweep_v1.yaml"
)

foreach ($config in $configs) {
    python scripts\evaluate_seed_sweep.py --config $config
}
```

Generate the consolidated comparison:

```powershell
python scripts\compare_multipath_mitigation.py --config configs\compare_residual_equalizer_v1.yaml
```

### Frozen-backbone residual front end

Train five paired front ends while loading and freezing the matching baseline checkpoint for each seed:

```powershell
python scripts\run_baseline_seed_sweep.py --config configs\baseline_groupnorm_frozen_backbone_equalizer_seed_sweep_v1.yaml
```

Evaluate all four channel conditions:

```powershell
$configs = @(
    "configs\evaluate_groupnorm_frozen_backbone_equalizer_seed_sweep_v1.yaml",
    "configs\evaluate_groupnorm_frozen_backbone_equalizer_mild_seed_sweep_v1.yaml",
    "configs\evaluate_groupnorm_frozen_backbone_equalizer_moderate_seed_sweep_v1.yaml",
    "configs\evaluate_groupnorm_frozen_backbone_equalizer_severe_seed_sweep_v1.yaml"
)

foreach ($config in $configs) {
    python scripts\evaluate_seed_sweep.py --config $config
}
```

Generate the consolidated frozen-backbone comparison:

```powershell
python scripts\compare_multipath_mitigation.py --config configs\compare_frozen_backbone_equalizer_v1.yaml
```

## Reproduce Historical Ablations

### Original BatchNorm five-seed study

```powershell
python scripts\run_baseline_seed_sweep.py --config configs\baseline_seed_sweep_v1.yaml
python scripts\evaluate_seed_sweep.py --config configs\evaluate_seed_sweep_v1.yaml
```

### RMS-normalized five-seed study

```powershell
python scripts\run_baseline_seed_sweep.py --config configs\baseline_rms_seed_sweep_v1.yaml
python scripts\evaluate_seed_sweep.py --config configs\evaluate_rms_seed_sweep_v1.yaml
```

### Original GroupNorm five-seed study

```powershell
python scripts\run_baseline_seed_sweep.py --config configs\baseline_groupnorm_seed_sweep_v1.yaml
python scripts\evaluate_seed_sweep.py --config configs\evaluate_groupnorm_seed_sweep_v1.yaml
```

## Quality Checks

Run the complete test suite with warnings treated as errors:

```powershell
python -m pytest -W error
```

Run static analysis:

```powershell
python -m ruff check src tests scripts
```

Check whitespace and patch integrity:

```powershell
git diff --check
```

At the current milestone, the repository contains **548 passing tests**.

## Repository Structure

```text
rf-signal-intelligence-lab/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
├── scripts/
├── src/
│   └── rfsil/
│       ├── data/
│       ├── dsp/
│       ├── models/
│       ├── ssl/
│       ├── training/
│       ├── evaluation/
│       ├── deployment/
│       └── demo/
├── tests/
├── reports/
│   └── figures/
└── results/
```

## Environment

The original development environment uses:

- Windows
- PowerShell
- Python 3.12
- PyTorch 2.11.0 with CUDA 12.8
- NVIDIA RTX 4080 SUPER
- 64 GB RAM
- PyCharm
- Git
- GitHub Desktop

The project runs locally. It does not require cloud services, paid APIs, or the OpenAI API.

## Safety Scope

This project is limited to synthetic and public-dataset RF signal analysis for education, research, and portfolio development.

It does not include:

- Jamming
- Evasion
- Targeting
- Weapon guidance
- Operational military procedures
- Instructions for disrupting communication systems

## Known Limitations

- The current benchmark is entirely synthetic.
- Training and test data come from the same signal-generator family.
- Real receiver effects and hardware-specific distortions are not yet represented fully.
- Public RF datasets have not yet been integrated.
- Low-SNR and severe-multipath PSK discrimination remain the dominant failure regimes.
- Residual front ends reduce, but do not eliminate, QPSK/8PSK confusion and PSK-to-16QAM collapse under severe multipath.
- The learned front-end transformations are not established physical channel inverses.
- Confidence calibration and uncertainty estimation are not yet implemented.
- No self-supervised representation-learning result is available yet.
- Deployment benchmarking and ONNX export are not yet implemented.

## Roadmap

### Completed

- [x] Reproducible Windows and CUDA environment
- [x] Synthetic modulation generation
- [x] Signal impairment pipeline
- [x] Root-raised-cosine pulse shaping
- [x] IQ, constellation, waveform, and spectrogram visualizations
- [x] Balanced dataset generation
- [x] PyTorch dataset loader
- [x] Baseline one-dimensional CNN
- [x] GPU training pipeline
- [x] Held-out evaluation
- [x] Confusion matrix
- [x] Accuracy-by-SNR analysis
- [x] Class-by-SNR error analysis
- [x] Five-seed reproducibility study
- [x] RMS-normalization ablation
- [x] BatchNorm versus GroupNorm ablation
- [x] GroupNorm baseline selection
- [x] GroupNorm 8PSK regression diagnosis
- [x] Public CNN embedding extraction interface
- [x] Frozen linear-head refit
- [x] Native PyTorch classifier-head conversion
- [x] Five-seed frozen-head reproducibility study
- [x] Frequency-selective multipath channel profiles
- [x] Balanced mixed-multipath supervised training
- [x] Paired clean, mild, moderate, and severe evaluation
- [x] Joint learnable residual signal front end
- [x] Frozen-backbone parameter-efficient adaptation
- [x] Five-seed residual-front-end robustness studies
- [x] Correction-magnitude and pooled-confusion analysis
- [x] Automated testing and Ruff checks

### Next

- [ ] Low-SNR-aware training experiments
- [ ] Confidence calibration
- [ ] Uncertainty estimation
- [ ] Self-supervised contrastive encoder
- [ ] Self-supervised linear evaluation and fine-tuning
- [ ] Public RF dataset integration
- [ ] ONNX export
- [ ] Local latency benchmark
- [ ] Streamlit demo
- [ ] Final technical report and GitHub release

## License

This project is licensed under the MIT License.
