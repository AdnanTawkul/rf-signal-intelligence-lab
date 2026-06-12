@"
# RF Signal Intelligence Lab

A local, reproducible AI research project for RF modulation recognition from synthetic raw IQ signal data.

This repository is designed as a serious AI engineering and signal-processing portfolio project. It focuses on clean synthetic-data generation, signal impairments, modulation classification, self-supervised representation learning, robustness analysis, ONNX export, and a local Streamlit demo.

## Project Goals

- Generate synthetic RF IQ datasets using public, non-operational signal models.
- Apply realistic signal impairments such as AWGN, SNR control, frequency offset, phase offset, amplitude scaling, time shift, and fading.
- Visualize IQ signals using time-domain plots, constellation diagrams, and spectrograms.
- Train PyTorch baseline models for modulation recognition.
- Evaluate model performance by modulation type and SNR level.
- Add self-supervised contrastive learning for RF representation learning.
- Export trained models to ONNX.
- Benchmark local inference latency.
- Provide a local Streamlit demo app.

## Safety Scope

This project is limited to public-dataset and synthetic-data RF signal analysis for educational and research portfolio purposes.

It does not include jamming, evasion, targeting, weapon guidance, operational military procedures, or instructions for interfering with communication systems.

## Hardware Target

Development machine:

- Windows
- NVIDIA RTX 4080 / RTX 4080 SUPER
- 64 GB RAM
- Local GPU execution
- No cloud
- No paid APIs

## Current Environment

- Python: 3.12
- PyTorch: 2.11.0+cu128
- CUDA wheel channel: cu128
- Development: Windows, PowerShell, PyCharm, Git, GitHub Desktop

## Planned Milestones

1. Project setup and reproducible environment
2. Synthetic IQ signal generator
3. Signal impairment pipeline
4. RF visualizations
5. Dataset generation scripts
6. PyTorch CNN baseline
7. Evaluation by modulation and SNR
8. Confusion matrix and accuracy-by-SNR plots
9. Self-supervised contrastive encoder
10. Fine-tuning and linear evaluation
11. Robustness and uncertainty analysis
12. ONNX export
13. Local latency benchmark
14. Streamlit demo app
15. Tests, report, and final GitHub polish

## Repository Structure

```text
rf-signal-intelligence-lab/
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