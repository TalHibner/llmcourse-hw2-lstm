# Implementation Tasks Breakdown
## LSTM Frequency Extraction System

**Status Key**: ⬜ Not Started | 🔄 In Progress | ✅ Completed

---

## Phase 1: Project Setup

### Task 1.1: Environment Configuration ⬜
**Estimated Time**: 15 minutes
**Priority**: High

**Steps**:
1. Initialize UV package manager
   ```bash
   uv init
   ```
2. Create `pyproject.toml` with dependencies:
   - torch >= 2.0.0
   - numpy >= 1.24.0
   - matplotlib >= 3.7.0
   - jupyter >= 1.0.0
3. Install dependencies
   ```bash
   uv sync
   ```

**Acceptance Criteria**:
- [ ] `pyproject.toml` created and configured
- [ ] All dependencies installed successfully
- [ ] Can import torch, numpy, matplotlib

---

### Task 1.2: Project Structure ⬜
**Estimated Time**: 10 minutes
**Priority**: High

**Steps**:
1. Create directory structure:
   ```
   src/
   notebooks/
   data/
   models/
   results/plots/
   tests/
   ```
2. Create `.gitignore` for Python projects
3. Initialize empty `__init__.py` in `src/`

**Acceptance Criteria**:
- [ ] All directories created
- [ ] `.gitignore` includes: `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `data/`, `models/`, `results/`

---

## Phase 2: Data Generation

### Task 2.1: Signal Generator Module ⬜
**Estimated Time**: 45 minutes
**Priority**: High
**File**: `src/data_generator.py`

**Steps**:
1. Implement `generate_noisy_sinusoid()` function
   - Input: frequency, time array, random seed
   - Generate random A_i(t) ~ Uniform(0.8, 1.2) for EACH sample
   - Generate random φ_i(t) ~ Uniform(0, 2π) for EACH sample
   - Return: Sinus_i^noisy(t) = A_i(t) · sin(2π·f_i·t + φ_i(t))

2. Implement `generate_mixed_signal()` function
   - Input: frequencies=[1,3,5,7], time array, random seed
   - Generate 4 noisy sinusoids
   - Return: S(t) = (1/4) Σ Sinus_i^noisy(t)

3. Implement `generate_ground_truth()` function
   - Input: frequency, time array
   - Return: Target_i(t) = sin(2π·f_i·t)

**Acceptance Criteria**:
- [ ] Functions generate correct signal shapes
- [ ] Noise varies at EACH time sample (verify with assertions)
- [ ] Different seeds produce different noise patterns
- [ ] Ground truth is pure sinusoid without noise

**Test**:
```python
# Verify noise varies per sample
t = np.linspace(0, 10, 10000)
s1 = generate_noisy_sinusoid(1, t, seed=1)
s2 = generate_noisy_sinusoid(1, t, seed=1)  # Same seed
s3 = generate_noisy_sinusoid(1, t, seed=2)  # Different seed
assert np.allclose(s1, s2)  # Reproducible
assert not np.allclose(s1, s3)  # Different
```

---

### Task 2.2: Dataset Builder ⬜
**Estimated Time**: 30 minutes
**Priority**: High
**File**: `src/data_generator.py`

**Steps**:
1. Implement `create_frequency_dataset()` function
   - Generate time array: t = linspace(0, 10, 10000)
   - Generate mixed signal S(t) with given seed
   - Generate 4 ground truth targets
   - Create dataset with 40,000 rows:
     * Rows 1-10000: (S[t], [1,0,0,0], Target_1[t])
     * Rows 10001-20000: (S[t], [0,1,0,0], Target_2[t])
     * Rows 20001-30000: (S[t], [0,0,1,0], Target_3[t])
     * Rows 30001-40000: (S[t], [0,0,0,1], Target_4[t])

2. Implement `save_dataset()` function
   - Save as .npz file with keys: 'S', 'C', 'Target', 'metadata'

3. Implement `load_dataset()` function
   - Load .npz file and return arrays

**Acceptance Criteria**:
- [ ] Dataset has shape (40000, ) for S and Target
- [ ] C matrix has shape (40000, 4) with one-hot encoding
- [ ] Each frequency appears exactly 10,000 times
- [ ] Data is reproducible with same seed
- [ ] Can save and load successfully

---

### Task 2.3: Generate Train and Test Datasets ⬜
**Estimated Time**: 10 minutes
**Priority**: High

**Steps**:
1. Generate training dataset with seed #1
2. Generate test dataset with seed #2
3. Save both to `data/` directory
4. Verify datasets are different (different noise)

**Acceptance Criteria**:
- [ ] `data/train_dataset.npz` created
- [ ] `data/test_dataset.npz` created
- [ ] Both have 40,000 samples
- [ ] S values differ between datasets (different noise)
- [ ] Target values are identical (same pure frequencies)

---

## Phase 3: Model Implementation

### Task 3.1: LSTM Model Architecture ⬜
**Estimated Time**: 30 minutes
**Priority**: High
**File**: `src/model.py`

**Steps**:
1. Implement `FrequencyExtractorLSTM` class
   ```python
   class FrequencyExtractorLSTM(nn.Module):
       def __init__(self, input_size=5, hidden_size=64, num_layers=1):
           # LSTM layer
           # Fully connected output layer

       def forward(self, x, hidden=None):
           # LSTM forward pass
           # FC layer
           # Return output and new hidden state
   ```

2. Add helper methods:
   - `get_num_parameters()`: Count trainable parameters
   - `reset_state()`: Explicitly create zero hidden state

**Acceptance Criteria**:
- [ ] Model accepts input shape [batch, seq_len=1, features=5]
- [ ] Model outputs shape [batch, 1]
- [ ] Returns hidden state tuple (h_n, c_n)
- [ ] Can handle hidden=None (automatic zero initialization)
- [ ] No errors when running forward pass

**Test**:
```python
model = FrequencyExtractorLSTM()
x = torch.randn(32, 1, 5)  # batch=32, seq=1, features=5
output, hidden = model(x, hidden=None)
assert output.shape == (32, 1)
assert hidden[0].shape == (1, 32, 64)  # h_n
assert hidden[1].shape == (1, 32, 64)  # c_n
```

---

### Task 3.2: PyTorch Dataset Class ⬜
**Estimated Time**: 20 minutes
**Priority**: High
**File**: `src/model.py`

**Steps**:
1. Implement `FrequencyDataset` class
   ```python
   class FrequencyDataset(Dataset):
       def __init__(self, S, C, targets):
           # Convert to tensors

       def __len__(self):
           # Return dataset size

       def __getitem__(self, idx):
           # Return (S[idx], C[idx], target[idx])
   ```

2. Test with DataLoader

**Acceptance Criteria**:
- [ ] Returns tensors of correct shape
- [ ] Compatible with PyTorch DataLoader
- [ ] Can iterate through batches

---

## Phase 4: Training Pipeline

### Task 4.1: Trainer Module ⬜
**Estimated Time**: 45 minutes
**Priority**: High
**File**: `src/trainer.py`

**Steps**:
1. Implement `train_epoch()` function
   - Loop through batches
   - **CRITICAL**: Reset hidden state for each batch (hidden=None)
   - Concatenate S and C inputs
   - Forward pass
   - Compute MSE loss
   - Backward pass
   - Return average epoch loss

2. Implement `train_model()` function
   - Loop through epochs
   - Call train_epoch()
   - Track losses
   - Save best model
   - Early stopping (optional)

3. Implement `get_config()` function
   - Return training hyperparameters dictionary

**Acceptance Criteria**:
- [ ] Training loop runs without errors
- [ ] Loss decreases over epochs
- [ ] Hidden state is reset between batches (verify in code)
- [ ] Model weights are updated
- [ ] Best model is saved

**Critical Implementation**:
```python
for batch in dataloader:
    # ✅ CORRECT: Reset state
    hidden = None

    outputs, hidden = model(inputs, hidden)

    # ❌ WRONG: Don't do this
    # if hidden is None:
    #     hidden = model.init_hidden(batch_size)
    # outputs, hidden = model(inputs, hidden.detach())
```

---

### Task 4.2: Training Execution ⬜
**Estimated Time**: 30 minutes
**Priority**: High

**Steps**:
1. Load training dataset
2. Create DataLoader with batch_size=64, shuffle=True
3. Initialize model
4. Define optimizer (Adam, lr=0.001)
5. Define criterion (MSELoss)
6. Train for 50 epochs (or until convergence)
7. Save training loss history
8. Save final model

**Acceptance Criteria**:
- [ ] Training completes successfully
- [ ] Final training MSE < 0.01 (reasonable threshold)
- [ ] Model saved to `models/best_model.pth`
- [ ] Loss history saved for plotting

---

## Phase 5: Evaluation

### Task 5.1: Evaluator Module ⬜
**Estimated Time**: 30 minutes
**Priority**: High
**File**: `src/evaluator.py`

**Steps**:
1. Implement `evaluate_model()` function
   - Set model to eval mode
   - Loop through dataset
   - **CRITICAL**: Reset hidden state for each batch
   - Collect predictions
   - Compute MSE
   - Return MSE, predictions, targets

2. Implement `compute_metrics()` function
   - Calculate MSE_train
   - Calculate MSE_test
   - Check generalization: |MSE_test - MSE_train|
   - Return metrics dictionary

**Acceptance Criteria**:
- [ ] Evaluation runs without errors
- [ ] MSE computed correctly
- [ ] Predictions collected for all samples
- [ ] Can evaluate on both train and test sets

---

### Task 5.2: Generate Predictions ⬜
**Estimated Time**: 20 minutes
**Priority**: High

**Steps**:
1. Load trained model
2. Evaluate on training set (seed #1)
3. Evaluate on test set (seed #2)
4. Organize predictions by frequency
5. Save results

**Acceptance Criteria**:
- [ ] Predictions generated for all 40,000 samples
- [ ] Organized by frequency (f1, f2, f3, f4)
- [ ] Results saved to `results/predictions.npz`

---

## Phase 6: Visualization

### Task 6.1: Visualizer Module ⬜
**Estimated Time**: 45 minutes
**Priority**: High
**File**: `src/visualizer.py`

**Steps**:
1. Implement `plot_training_curve()` function
   - Plot training loss over epochs

2. Implement `plot_single_frequency_comparison()` function
   - Plot for one frequency (e.g., f2):
     * Green dots: Noisy mixed signal S (background, alpha=0.3)
     * Blue line: Pure target (ground truth)
     * Red dots: LSTM predictions
   - Add legend, grid, labels
   - Save to `results/plots/freq_comparison.png`

3. Implement `plot_all_frequencies()` function
   - Create 2x2 subplot
   - Each subplot shows one frequency extraction
   - Plot target (blue line) vs LSTM output (red dashed line)
   - Save to `results/plots/all_frequencies.png`

4. Implement `plot_metrics_comparison()` function
   - Bar chart comparing MSE_train vs MSE_test

**Acceptance Criteria**:
- [ ] Graphs are clear and professional
- [ ] Legends and labels present
- [ ] Saved as high-resolution images
- [ ] Demonstrates successful frequency extraction

---

### Task 6.2: Generate All Visualizations ⬜
**Estimated Time**: 15 minutes
**Priority**: High

**Steps**:
1. Load predictions
2. Generate all required plots
3. Save to `results/plots/`

**Acceptance Criteria**:
- [ ] All plots generated and saved
- [ ] Visual inspection confirms good extraction
- [ ] Plots match requirements in PRD section 3.4

---

## Phase 7: Documentation and Notebook

### Task 7.1: Create Jupyter Notebook ⬜
**Estimated Time**: 60 minutes
**Priority**: High
**File**: `notebooks/lstm_frequency_extraction.ipynb`

**Steps**:
1. **Section 1: Introduction**
   - Problem statement
   - Assignment overview
   - Import libraries

2. **Section 2: Data Generation**
   - Generate training and test datasets
   - Visualize mixed signals and targets
   - Explain noise characteristics

3. **Section 3: Model Architecture**
   - Define LSTM model
   - Explain state management strategy
   - Show model summary

4. **Section 4: Training**
   - Configure training parameters
   - Train model
   - Plot training curve
   - Explain state reset mechanism

5. **Section 5: Evaluation**
   - Evaluate on train and test sets
   - Display MSE metrics
   - Check generalization

6. **Section 6: Results Visualization**
   - Graph 1: Single frequency detailed comparison
   - Graph 2: All four frequencies
   - Analysis and discussion

7. **Section 7: Conclusions**
   - Summary of findings
   - Success criteria verification
   - Future improvements

**Acceptance Criteria**:
- [ ] Notebook runs end-to-end without errors
- [ ] All sections complete with explanations
- [ ] Visualizations embedded and clear
- [ ] Markdown cells explain each step
- [ ] Code is well-commented

---

### Task 7.2: Create README.md ⬜
**Estimated Time**: 30 minutes
**Priority**: Medium
**File**: `README.md`

**Steps**:
1. Project overview
2. Installation instructions (UV setup)
3. Usage guide (how to run notebook)
4. Project structure explanation
5. Results summary
6. References

**Acceptance Criteria**:
- [ ] Clear and comprehensive
- [ ] Easy to follow for new users
- [ ] Includes quick start guide

---

## Phase 8: Testing and Validation

### Task 8.1: Unit Tests ⬜
**Estimated Time**: 45 minutes
**Priority**: Medium
**File**: `tests/test_data_generator.py`

**Steps**:
1. Test signal generation functions
2. Test dataset creation
3. Test model forward pass
4. Test state reset

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] Code coverage > 70%

---

### Task 8.2: End-to-End Validation ⬜
**Estimated Time**: 30 minutes
**Priority**: High

**Steps**:
1. Run complete pipeline from scratch
2. Verify MSE metrics meet success criteria:
   - MSE_train < 0.05
   - |MSE_test - MSE_train| < 0.01 (good generalization)
3. Visual inspection of graphs
4. Verify all deliverables present

**Acceptance Criteria**:
- [ ] Pipeline runs successfully
- [ ] Metrics meet thresholds
- [ ] Visualizations show clear frequency extraction
- [ ] All files committed to git

---

## Phase 9: Final Deliverables

### Task 9.1: Code Cleanup ⬜
**Estimated Time**: 20 minutes
**Priority**: Medium

**Steps**:
1. Remove debug print statements
2. Add docstrings to all functions
3. Format code (black formatter)
4. Remove unused imports

**Acceptance Criteria**:
- [ ] Code is clean and professional
- [ ] Consistent formatting
- [ ] Well-documented

---

### Task 9.2: Final Git Commit and Push ⬜
**Estimated Time**: 10 minutes
**Priority**: High

**Steps**:
1. Review all changes
2. Commit final implementation
3. Push to GitHub
4. Verify repository is complete

**Acceptance Criteria**:
- [ ] All files committed
- [ ] Pushed to remote repository
- [ ] Repository is public and accessible

---

## Summary Checklist

### Documentation ✅
- [ ] PRD.md created
- [ ] DESIGN.md created
- [ ] TASKS.md created
- [ ] README.md created

### Code Implementation ⬜
- [ ] Data generation module
- [ ] LSTM model module
- [ ] Training module
- [ ] Evaluation module
- [ ] Visualization module

### Execution ⬜
- [ ] Training and test datasets generated
- [ ] Model trained successfully
- [ ] Evaluation metrics computed
- [ ] Visualizations created

### Results ⬜
- [ ] MSE_train < 0.05
- [ ] MSE_test ≈ MSE_train
- [ ] Visual confirmation of frequency extraction

### Final Deliverables ⬜
- [ ] Jupyter notebook complete
- [ ] All code committed to Git
- [ ] Pushed to GitHub
- [ ] Project is reproducible

---

## Time Estimate Summary

| Phase | Estimated Time |
|-------|---------------|
| 1. Project Setup | 25 min |
| 2. Data Generation | 85 min |
| 3. Model Implementation | 50 min |
| 4. Training Pipeline | 75 min |
| 5. Evaluation | 50 min |
| 6. Visualization | 60 min |
| 7. Documentation | 90 min |
| 8. Testing | 75 min |
| 9. Final Deliverables | 30 min |
| **Total** | **~8.5 hours** |

---

## Dependencies Between Tasks

```
1.1 → 1.2 → 2.1 → 2.2 → 2.3
                    ↓
              3.1 → 3.2 → 4.1 → 4.2
                              ↓
                        5.1 → 5.2 → 6.1 → 6.2
                                          ↓
                                    7.1 → 7.2
                                          ↓
                                    8.1 → 8.2 → 9.1 → 9.2
```

**Critical Path**: 1.1 → 1.2 → 2.1 → 2.2 → 2.3 → 3.1 → 3.2 → 4.1 → 4.2 → 5.1 → 5.2 → 6.2 → 7.1 → 9.2

---

## Notes

- **Critical Tasks** (must complete): All tasks in Phase 1-7
- **Optional Tasks**: Phase 8.1 (Unit Tests) - recommended but not required
- **State Reset**: This is the MOST CRITICAL implementation detail - verify multiple times
- **Seeds**: Always use seed #1 for training, seed #2 for testing
- **Visualization**: Graph quality matters - make them publication-ready

---

**Last Updated**: Task list created
**Next Action**: Begin Task 1.1 (Environment Configuration)
