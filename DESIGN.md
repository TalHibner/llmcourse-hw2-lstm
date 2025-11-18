# Technical Design Document
## LSTM Frequency Extraction System

**Version**: 1.0
**Last Updated**: November 2025

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LSTM Frequency Filter                     │
│                                                              │
│  Input: [S[t], C]  →  LSTM Network  →  Output: Target_i[t] │
│                           ↓                                  │
│                  Internal State Reset                        │
│                     (h_t, c_t) = 0                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 High-Level Flow

1. **Data Generation Module**: Creates training and test datasets with controlled noise
2. **LSTM Model Module**: Neural network for conditional frequency extraction
3. **Training Module**: Trains model with proper state management
4. **Evaluation Module**: Computes metrics and generates visualizations

---

## 2. Data Architecture

### 2.1 Signal Generation Pipeline

```
Random Seeds (1 & 2)
        ↓
Generate Random Variations
    ├─ A_i(t) ~ Uniform(0.8, 1.2)
    └─ φ_i(t) ~ Uniform(0, 2π)
        ↓
Create Noisy Components
    Sinus_i^noisy(t) = A_i(t) · sin(2π·f_i·t + φ_i(t))
        ↓
Mix Signals
    S(t) = (1/4) Σ Sinus_i^noisy(t)
        ↓
Generate Ground Truth
    Target_i(t) = sin(2π·f_i·t)
```

### 2.2 Dataset Schema

#### Time Series Data
```python
t = np.linspace(0, 10, 10000)  # Time array
# For each t[k]:
#   - Generate 4 noisy sinusoids with random A_i, φ_i
#   - Compute mixed signal S[k]
#   - Store pure targets Target_i[k] for all 4 frequencies
```

#### Training Dataset Structure
```
Shape: (40000, 6)
Columns: [S[t], C1, C2, C3, C4, Target]

Breakdown:
- Rows 1-10000:     Frequency f1 (C=[1,0,0,0])
- Rows 10001-20000: Frequency f2 (C=[0,1,0,0])
- Rows 20001-30000: Frequency f3 (C=[0,0,1,0])
- Rows 30001-40000: Frequency f4 (C=[0,0,0,1])
```

### 2.3 Data Storage Format

**Training Dataset**: NumPy arrays or PyTorch tensors
```python
{
    'S': np.array([...]),          # Shape: (40000,)
    'C': np.array([...]),          # Shape: (40000, 4)
    'Target': np.array([...]),     # Shape: (40000,)
    'metadata': {
        'seed': 1,
        'frequencies': [1, 3, 5, 7],
        'fs': 1000,
        'duration': 10
    }
}
```

---

## 3. Model Architecture

### 3.1 Network Design

```
Input Layer (5 units)
    ↓
[S[t], C1, C2, C3, C4]
    ↓
LSTM Layer(s)
    - Hidden size: configurable (e.g., 64, 128)
    - Number of layers: configurable (e.g., 1, 2)
    - State management: MANUAL RESET between samples
    ↓
Fully Connected Layer
    - Output size: 1
    ↓
Output: Target_i[t] (scalar)
```

### 3.2 PyTorch Implementation Design

```python
class FrequencyExtractorLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: Input tensor [batch, seq_len=1, features=5]
            hidden_state: Tuple (h_0, c_0) or None for reset
        Returns:
            output: Predicted target [batch, 1]
            new_hidden_state: Updated (h_n, c_n)
        """
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        output = self.fc(lstm_out[:, -1, :])  # Take last timestep
        return output, hidden_state
```

### 3.3 State Management Strategy

#### For L = 1 (Default Approach)

```python
# Training loop with state reset
for batch in dataloader:
    # CRITICAL: Reset hidden state for each batch
    hidden = None  # This forces PyTorch to initialize with zeros

    outputs, hidden = model(inputs, hidden)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

**Key Point**: By setting `hidden=None`, LSTM automatically initializes (h_0, c_0) to zeros, effectively resetting state between batches.

#### Alternative for L > 1 (Optional)

```python
# Sliding window approach
for sequence_batch in dataloader:
    # sequence_batch shape: [batch, seq_len, features]
    hidden = None  # Reset at start of each independent sequence

    outputs, hidden = model(sequence_batch, hidden)
    # outputs shape: [batch, seq_len, 1]

    # Only use last output for loss
    loss = criterion(outputs[:, -1, :], targets)
```

---

## 4. Training Architecture

### 4.1 Training Configuration

```python
TRAINING_CONFIG = {
    # Model
    'input_size': 5,        # S[t] + 4 C values
    'hidden_size': 64,      # LSTM hidden units (tunable)
    'num_layers': 1,        # LSTM layers (tunable)

    # Training
    'batch_size': 64,       # Samples per batch
    'learning_rate': 0.001, # Adam optimizer
    'num_epochs': 50,       # Training epochs (tunable)

    # Loss
    'criterion': 'MSE',     # Mean Squared Error

    # State Management
    'sequence_length': 1,   # L = 1 (default)
    'reset_state': True,    # Reset between batches
}
```

### 4.2 Training Loop Design

```python
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for batch_idx, (S, C, targets) in enumerate(dataloader):
        # Prepare input: concatenate S and C
        inputs = torch.cat([S.unsqueeze(-1), C], dim=-1)
        inputs = inputs.unsqueeze(1)  # Add seq dimension [batch, 1, 5]

        # CRITICAL: Reset state
        hidden = None

        # Forward pass
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.squeeze(), targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
```

### 4.3 Data Loading Strategy

```python
# Custom Dataset
class FrequencyDataset(Dataset):
    def __init__(self, S, C, targets):
        self.S = torch.FloatTensor(S)
        self.C = torch.FloatTensor(C)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        return self.S[idx], self.C[idx], self.targets[idx]

# DataLoader with shuffling
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,  # Important for learning
    num_workers=0
)
```

---

## 5. Evaluation Architecture

### 5.1 Metrics Computation

```python
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_mse = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for S, C, targets in dataloader:
            inputs = torch.cat([S.unsqueeze(-1), C], dim=-1)
            inputs = inputs.unsqueeze(1)

            # Reset state for evaluation
            hidden = None
            outputs, _ = model(inputs, hidden)

            mse = criterion(outputs.squeeze(), targets)
            total_mse += mse.item()

            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    avg_mse = total_mse / len(dataloader)
    return avg_mse, predictions, actuals
```

### 5.2 Visualization Architecture

#### Graph 1: Single Frequency Detailed Comparison
```python
def plot_single_frequency_comparison(t, S, lstm_output, target, freq_idx):
    """
    Shows:
    1. Mixed noisy signal S (background, green dots)
    2. Pure target (line)
    3. LSTM predictions (red dots)
    """
    plt.figure(figsize=(14, 6))
    plt.scatter(t, S, c='green', alpha=0.3, s=1, label='Noisy Mixed S')
    plt.plot(t, target, 'b-', linewidth=2, label=f'Target f{freq_idx}')
    plt.scatter(t, lstm_output, c='red', s=10, label='LSTM Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Frequency Extraction: f{freq_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
```

#### Graph 2: All Four Frequencies
```python
def plot_all_frequencies(t, predictions_dict, targets_dict):
    """
    4 subplots, one for each frequency
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    frequencies = [1, 3, 5, 7]

    for idx, (ax, freq) in enumerate(zip(axes.flat, frequencies)):
        ax.plot(t, targets_dict[freq], 'b-', label='Target')
        ax.plot(t, predictions_dict[freq], 'r--', label='LSTM', alpha=0.7)
        ax.set_title(f'f{idx+1} = {freq} Hz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
```

---

## 6. Technology Stack

### 6.1 Core Dependencies

```toml
[project]
dependencies = [
    "torch>=2.0.0",           # PyTorch for LSTM
    "numpy>=1.24.0",          # Numerical computations
    "matplotlib>=3.7.0",      # Visualization
    "jupyter>=1.0.0",         # Interactive development
    "scikit-learn>=1.3.0",    # Optional: additional metrics
]
```

### 6.2 Development Tools

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",          # Testing
    "black>=23.0.0",          # Code formatting
    "ipykernel>=6.25.0",      # Jupyter kernel
]
```

---

## 7. Module Structure

```
llmcourse-hw2-lstm/
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Signal generation and dataset creation
│   ├── model.py                # LSTM model definition
│   ├── trainer.py              # Training loop and state management
│   ├── evaluator.py            # Metrics and evaluation
│   └── visualizer.py           # Plotting functions
│
├── notebooks/
│   └── lstm_frequency_extraction.ipynb  # Main implementation
│
├── data/
│   ├── train_dataset.npz       # Generated training data
│   └── test_dataset.npz        # Generated test data
│
├── models/
│   └── best_model.pth          # Trained model weights
│
├── results/
│   ├── metrics.json            # Evaluation metrics
│   └── plots/                  # Generated visualizations
│
├── tests/
│   └── test_data_generator.py  # Unit tests
│
├── pyproject.toml              # UV package configuration
├── .gitignore
├── PRD.md
├── DESIGN.md
├── TASKS.md
└── README.md
```

---

## 8. Key Design Decisions

### 8.1 Why PyTorch?
- Native LSTM implementation with explicit state management
- Easy to control hidden state (h_t, c_t) reset
- Excellent for research and educational purposes
- Good visualization integration

### 8.2 Why Sequence Length L = 1?
- **Pedagogical focus**: Demonstrates that LSTM can learn frequency patterns through internal memory management alone
- Shows the network doesn't rely on sequential presentation of data
- Simpler to implement and understand state reset
- Proves the network learns the inherent periodic structure

### 8.3 State Reset Strategy
Using `hidden=None` in PyTorch LSTM:
- Automatic zero initialization
- Clean and explicit
- No manual tensor creation needed
- Matches the pedagogical requirement

### 8.4 Batch Processing
- Batch size = 64 (reasonable for 40K samples)
- Shuffle training data (breaks any accidental temporal patterns)
- No shuffling for test data (maintain order for visualization)

---

## 9. Performance Considerations

### 9.1 Expected Complexity
- **Training time**: O(epochs × samples × model_parameters)
- **Memory**: Modest (40K samples, small LSTM)
- **Expected**: < 5 minutes on CPU, < 1 minute on GPU

### 9.2 Optimization Opportunities
1. Use GPU if available (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`)
2. Adjust batch size based on available memory
3. Early stopping based on validation MSE
4. Learning rate scheduling

---

## 10. Testing Strategy

### 10.1 Unit Tests
- `test_signal_generation`: Verify noise variations at each sample
- `test_dataset_structure`: Verify shape and format
- `test_model_forward`: Verify model output shape
- `test_state_reset`: Verify hidden state is properly reset

### 10.2 Integration Tests
- End-to-end training for 1 epoch
- Verify MSE decreases over epochs
- Verify model can predict on test set

### 10.3 Validation Checks
- MSE_train should decrease during training
- MSE_test ≈ MSE_train (within reasonable tolerance)
- Visual inspection of extracted frequencies

---

## 11. Extensibility

### 11.1 Future Enhancements
- Support for different sequence lengths (L > 1)
- Comparison with other architectures (GRU, Transformer)
- Additional noise models
- Real-time frequency extraction
- Multiple simultaneous frequency extraction

### 11.2 Hyperparameter Tuning
- Hidden size: [32, 64, 128, 256]
- Number of layers: [1, 2, 3]
- Learning rate: [0.0001, 0.001, 0.01]
- Batch size: [32, 64, 128]

---

## 12. Success Criteria

The design is successful if:

1. ✅ Clear separation of concerns (data, model, training, evaluation)
2. ✅ Proper state management implementation
3. ✅ Reproducible results (fixed random seeds)
4. ✅ Efficient training (< 10 minutes)
5. ✅ Clean visualization outputs
6. ✅ Extensible and maintainable code

---

## Appendix A: Mathematical Formulation

### Signal Model
```
S(t) = (1/4) Σ_{i=1}^{4} A_i(t) · sin(2π·f_i·t + φ_i(t))

where:
    A_i(t) ~ Uniform(0.8, 1.2)  ∀t
    φ_i(t) ~ Uniform(0, 2π)      ∀t
    f_i ∈ {1, 3, 5, 7} Hz
```

### Conditional Regression
```
f_LSTM: (S[t], C) → Target_i[t]

where:
    C ∈ {[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]}
    Target_i[t] = sin(2π·f_i·t)
```

### Loss Function
```
L = MSE = (1/N) Σ_{n=1}^{N} (ŷ_n - y_n)²

where:
    ŷ_n = f_LSTM(S[t_n], C_n)
    y_n = Target_i[t_n]
```
