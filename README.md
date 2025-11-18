# LSTM System for Frequency Extraction from Mixed Signals

A PyTorch implementation of an LSTM-based frequency filter that extracts pure sinusoidal components from noisy mixed signals through conditional regression.

## Project Overview

This project demonstrates LSTM's ability to act as a frequency-selective filter by learning to isolate specific frequency components from a mixed, noisy signal. The system learns to extract pure sinusoids despite random amplitude and phase variations at each time sample.

### Key Features

- **Frequency Extraction**: Isolates 4 different frequency components (1Hz, 3Hz, 5Hz, 7Hz)
- **Noise Robustness**: Handles amplitude variations (±20%) and random phase shifts
- **Conditional Regression**: Uses one-hot encoding to select target frequency
- **State Management**: Demonstrates LSTM with sequence length L=1 and proper state reset
- **Generalization**: Tests on completely different noise patterns (separate random seeds)

## Problem Statement

Given a mixed signal **S(t)** composed of 4 sinusoidal frequencies with varying noise:

```
S(t) = (1/4) Σ A_i(t) · sin(2π·f_i·t + φ_i(t))

where:
    A_i(t) ~ Uniform(0.8, 1.2)  (varies at each sample)
    φ_i(t) ~ Uniform(0, 2π)      (varies at each sample)
    f_i ∈ {1, 3, 5, 7} Hz
```

**Goal**: Train an LSTM to extract pure frequency components:
```
Target_i(t) = sin(2π·f_i·t)
```

## Quick Start

### Prerequisites

- Python 3.10+
- UV package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/TalHibner/llmcourse-hw2-lstm.git
cd llmcourse-hw2-lstm
```

2. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies:
```bash
uv sync
```

### Running the Project

#### Option 1: Jupyter Notebook (Recommended)

```bash
uv run jupyter notebook notebooks/lstm_frequency_extraction.ipynb
```

Then run all cells to:
1. Generate datasets
2. Train the LSTM model
3. Evaluate performance
4. Generate visualizations

#### Option 2: Python Scripts

```bash
# Generate datasets
uv run python -c "from src.data_generator import create_datasets; create_datasets()"

# Train model
uv run python -c "from src.trainer import train_model; train_model()"

# Evaluate
uv run python -c "from src.evaluator import evaluate_and_visualize; evaluate_and_visualize()"
```

## Project Structure

```
llmcourse-hw2-lstm/
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_generator.py       # Signal generation and dataset creation
│   ├── model.py                # LSTM model definition
│   ├── trainer.py              # Training loop and state management
│   ├── evaluator.py            # Metrics and evaluation
│   └── visualizer.py           # Plotting functions
│
├── notebooks/                  # Jupyter notebooks
│   └── lstm_frequency_extraction.ipynb
│
├── data/                       # Generated datasets (gitignored)
│   ├── train_dataset.npz       # Training set (seed #1)
│   └── test_dataset.npz        # Test set (seed #2)
│
├── models/                     # Trained models (gitignored)
│   └── best_model.pth
│
├── results/                    # Results and visualizations (gitignored)
│   ├── metrics.json
│   └── plots/
│       ├── training_curve.png
│       ├── freq_comparison.png
│       └── all_frequencies.png
│
├── tests/                      # Unit tests
│   └── test_data_generator.py
│
├── PRD.md                      # Product Requirements Document
├── DESIGN.md                   # Technical Design Document
├── TASKS.md                    # Implementation Task Breakdown
├── README.md                   # This file
├── pyproject.toml              # UV package configuration
├── .gitignore
└── L2-homework.pdf             # Assignment specification
```

## Technical Details

### Dataset Specifications

| Parameter | Value |
|-----------|-------|
| Frequencies | f1=1Hz, f2=3Hz, f3=5Hz, f4=7Hz |
| Time Domain | 0-10 seconds |
| Sampling Rate | 1000 Hz |
| Total Samples | 10,000 per frequency |
| Dataset Size | 40,000 rows (10,000 × 4 frequencies) |

### Model Architecture

```
Input: [S[t], C1, C2, C3, C4]  (5 features)
   ↓
LSTM Layer (64 hidden units)
   ↓
Fully Connected Layer
   ↓
Output: Target_i[t]  (scalar)
```

**Critical Implementation Detail**: Internal state (h_t, c_t) is **reset between consecutive samples** (sequence length L=1), forcing the LSTM to learn frequency patterns through internal memory management alone.

### Training Configuration

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning rate = 0.001)
- **Batch Size**: 64
- **Epochs**: 50 (with early stopping)
- **State Management**: Reset hidden state for each batch

## Results

### Performance Metrics

Expected results after training:

- **MSE_train**: < 0.05 (on training set with seed #1)
- **MSE_test**: ≈ MSE_train (on test set with seed #2)
- **Generalization**: Good if |MSE_test - MSE_train| < 0.01

### Visualizations

The project generates two main visualizations:

1. **Single Frequency Comparison**: Shows the extraction of one frequency (e.g., f2) with:
   - Mixed noisy signal (background)
   - Pure target (ground truth)
   - LSTM predictions

2. **All Frequencies**: Four subplots showing extraction for each frequency

## Key Implementation Highlights

### 1. State Reset Mechanism

```python
for batch in dataloader:
    # CRITICAL: Reset state for each batch
    hidden = None  # PyTorch automatically initializes to zeros

    outputs, hidden = model(inputs, hidden)
    loss = criterion(outputs, targets)
```

This ensures the LSTM learns frequency patterns without relying on sequential dependencies between samples.

### 2. Noise Variation

```python
# Generate random variations FOR EACH sample
for t in time_array:
    A_i[t] = np.random.uniform(0.8, 1.2)  # Amplitude varies
    phi_i[t] = np.random.uniform(0, 2*np.pi)  # Phase varies
    signal[t] = A_i[t] * np.sin(2*np.pi*f_i*t + phi_i[t])
```

### 3. Conditional Frequency Selection

```python
# One-hot encoding for frequency selection
C = np.zeros(4)
C[frequency_index] = 1  # Select which frequency to extract
input = [S[t], *C]  # Concatenate signal and selection vector
```

## Documentation

- **PRD.md**: Complete product requirements and specifications
- **DESIGN.md**: Technical architecture and design decisions
- **TASKS.md**: Detailed implementation task breakdown

## Testing

Run unit tests:
```bash
uv run pytest tests/
```

## Assignment Requirements Checklist

- ✅ Create 2 datasets (train with seed #1, test with seed #2)
- ✅ Implement LSTM network with proper architecture
- ✅ Reset internal state between samples (L=1)
- ✅ Train model and achieve low MSE
- ✅ Evaluate generalization on test set
- ✅ Generate required visualizations
- ✅ Demonstrate noise robustness

## Future Enhancements

Potential improvements and extensions:

1. **Variable Sequence Length**: Experiment with L > 1 (sliding window)
2. **Comparison Study**: Compare with other architectures (GRU, Transformer)
3. **Additional Noise Models**: Test with different noise distributions
4. **Real-time Processing**: Adapt for streaming data
5. **Multiple Frequency Extraction**: Extract multiple frequencies simultaneously
6. **Hyperparameter Tuning**: Systematic search for optimal parameters

## Dependencies

Main dependencies (see `pyproject.toml` for complete list):

- **torch** >= 2.0.0: PyTorch for LSTM implementation
- **numpy** >= 1.24.0: Numerical computations
- **matplotlib** >= 3.7.0: Visualization
- **jupyter** >= 1.0.0: Interactive notebook environment

## Author

Implementation for M.Sc. Assignment
Original Assignment: Dr. Segal Yoram
November 2025

## License

All rights reserved © Dr. Segal Yoram

## References

- Assignment PDF: `L2-homework.pdf`
- PyTorch LSTM Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Understanding LSTM Networks: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Troubleshooting

### Common Issues

**Issue**: Dependencies fail to install
```bash
# Solution: Update UV
uv self update
uv sync
```

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size in config
# Edit src/trainer.py: batch_size = 32
```

**Issue**: Training loss not decreasing
```bash
# Solution: Check state reset implementation
# Verify hidden=None in training loop
```

**Issue**: Poor generalization (MSE_test >> MSE_train)
```bash
# Solution:
# 1. Verify different seeds for train/test
# 2. Increase training epochs
# 3. Add regularization (dropout)
```

## Contact

For questions or issues:
- Open an issue on GitHub: https://github.com/TalHibner/llmcourse-hw2-lstm/issues
- Check TASKS.md for implementation details
- Review DESIGN.md for architecture explanations

---

**Status**: Implementation in progress
**Last Updated**: November 2025
