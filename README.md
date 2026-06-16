# RF Signal Intelligence Lab

A local, reproducible AI engineering project for RF modulation recognition from synthetic raw IQ signals.

The repository covers the complete path from signal generation and channel simulation to PyTorch training, held-out evaluation, reproducibility analysis, and deployment-oriented extensions. It is designed as a serious portfolio project for AI, RF, signal-processing, and applied ML engineering roles.

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
- A compact one-dimensional CNN baseline
- GPU training on an NVIDIA RTX 4080 SUPER
- Held-out evaluation by class and SNR
- Confusion-matrix and class-by-SNR error analysis
- Five-seed training and test reproducibility studies
- Automated tests with `pytest`
- Static analysis with Ruff

The next research milestones are self-supervised representation learning, uncertainty calibration, ONNX export, local latency benchmarking, and a Streamlit demo.

## Baseline Result

The statistically defensible Baseline CNN v1 result is:

> **94.24% ± 0.29 percentage points held-out test accuracy across five independent training seeds**

Five independently trained checkpoints were evaluated on the same untouched test split.

| Metric | Result |
|---|---:|
| Mean test accuracy | 94.24% |
| Standard deviation | 0.29 percentage points |
| Minimum test accuracy | 93.93% |
| Maximum test accuracy | 94.71% |
| Mean accuracy at -4 dB | 70.50% |
| Test examples per run | 1,400 |

### Mean Per-Class Accuracy Across Five Seeds

| Modulation | Mean accuracy | Standard deviation |
|---|---:|---:|
| BPSK | 99.77% | 0.33 percentage points |
| QPSK | 87.77% | 2.32 percentage points |
| 8PSK | 93.09% | 3.15 percentage points |
| 16QAM | 96.34% | 1.72 percentage points |

### Mean Accuracy by SNR Across Five Seeds

| SNR | Mean accuracy | Standard deviation |
|---:|---:|---:|
| -4 dB | 70.50% | 1.10 percentage points |
| 0 dB | 94.60% | 1.07 percentage points |
| 4 dB | 99.10% | 0.37 percentage points |
| 8 dB | 98.50% | 0.45 percentage points |
| 12 dB | 99.30% | 0.24 percentage points |
| 16 dB | 97.90% | 0.20 percentage points |
| 20 dB | 99.80% | 0.24 percentage points |

![Baseline CNN five-seed evaluation](reports/figures/baseline_cnn_seed_sweep_v1_test.png)

The dominant weakness is low-SNR phase discrimination. QPSK at -4 dB was the worst class-SNR group in the original baseline analysis.

Detailed methodology, single-run metrics, confusion analysis, limitations, and five-seed results are documented in [Baseline CNN v1 Results](reports/baseline_cnn_v1.md).

## Signal and Dataset Pipeline

Each synthetic classification example follows this pipeline:

1. Random modulation-symbol generation
2. Root-raised-cosine transmit pulse shaping
3. Fixed-length waveform extraction
4. Amplitude scaling
5. Optional flat Rayleigh fading
6. Carrier frequency offset
7. Carrier phase offset
8. Zero-padded integer timing shift
9. AWGN at the configured SNR
10. Conversion to a two-channel tensor

The model input format is:

```text
[batch, 2, samples]
```

- Channel 0: in-phase component
- Channel 1: quadrature component
- Default baseline sample length: 2,048 samples

The generated dataset format also stores labels and impairment metadata, including SNR, frequency offset, phase offset, amplitude scale, timing shift, fading state, and generation seed.

## Baseline Model

Baseline CNN v1 is a compact one-dimensional convolutional classifier.

```text
Input: [batch, 2, 2048]

Conv block: 2 → 32
Conv block: 32 → 64
Conv block: 64 → 128
Adaptive global average pooling
Dropout
Linear classifier: 128 → 4
```

Each convolutional block contains:

- `Conv1d`
- Batch normalization
- GELU activation
- Max pooling

Trainable parameters:

```text
73,092
```

## Reproducibility Findings

Best-checkpoint validation performance was stable across five seeds:

```text
94.17% ± 0.31 percentage points
```

Final-epoch validation performance was much less stable:

```text
88.39% ± 4.13 percentage points
```

This distinction matters. The architecture learns repeatable features, but the training trajectory can become unstable late in training. Best-validation checkpoint selection is therefore part of the current experimental protocol.

## Visual Results

### Held-Out Confusion Matrix

![Baseline CNN v1 confusion matrix](reports/figures/baseline_cnn_v1_confusion_matrix.png)

### Accuracy by SNR

![Baseline CNN v1 accuracy by SNR](reports/figures/baseline_cnn_v1_accuracy_by_snr.png)

### Class-by-SNR Accuracy

![Baseline CNN v1 class-by-SNR accuracy](reports/figures/baseline_cnn_v1_class_snr_accuracy.png)

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

## Reproduce the Baseline

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

### Train Baseline CNN v1

```powershell
python scripts\train_baseline.py --config configs\train_baseline_v1.yaml
```

### Evaluate the selected baseline checkpoint

```powershell
python scripts\evaluate_baseline.py --config configs\evaluate_baseline_v1.yaml
```

### Run class-by-SNR error analysis

```powershell
python scripts\analyze_baseline_errors.py --config configs\analyze_baseline_v1.yaml
```

### Run the five-seed training study

```powershell
python scripts\run_baseline_seed_sweep.py --config configs\baseline_seed_sweep_v1.yaml
```

### Evaluate all five checkpoints on the held-out test split

```powershell
python scripts\evaluate_seed_sweep.py --config configs\evaluate_seed_sweep_v1.yaml
```

## Quality Checks

Run the complete test suite:

```powershell
python -m pytest
```

Run static analysis:

```powershell
python -m ruff check src tests scripts
```

Check whitespace and patch integrity:

```powershell
git diff --check
```

At the current baseline milestone, the repository contains 196 passing tests.

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
- Low-SNR QPSK and 8PSK discrimination remains the main model weakness.
- Confidence calibration and uncertainty estimation are not yet implemented.
- The baseline still relies on BatchNorm and best-checkpoint selection because final-epoch validation behavior is unstable.
- No self-supervised or architecture-ablation result has been completed yet.

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
- [x] Automated testing and Ruff checks

### Next

- [ ] GroupNorm and normalization ablations
- [ ] Low-SNR-aware training experiments
- [ ] Confidence calibration
- [ ] Uncertainty estimation
- [ ] Self-supervised contrastive encoder
- [ ] Linear evaluation and fine-tuning
- [ ] Public RF dataset integration
- [ ] ONNX export
- [ ] Local latency benchmark
- [ ] Streamlit demo
- [ ] Final technical report and GitHub release

## License

This project is licensed under the MIT License.
