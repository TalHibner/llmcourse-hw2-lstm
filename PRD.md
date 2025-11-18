# Product Requirements Document (PRD)
## LSTM System for Frequency Extraction from Mixed Signals

**Author**: Dr. Segal Yoram
**Date**: November 2025
**Version**: 1.0

---

## 1. Executive Summary

This project develops an LSTM-based system that extracts pure frequency components from noisy mixed signals. The system demonstrates LSTM's ability to act as a frequency-selective filter through conditional regression, learning to isolate specific sinusoidal components despite random amplitude and phase variations.

---

## 2. Problem Statement

### 2.1 Overview
Given a mixed signal **S** composed of 4 different sinusoidal frequencies with varying noise, the goal is to train an LSTM network to extract each pure frequency component separately while completely ignoring noise.

### 2.2 Key Challenge
The noise is **critical**: amplitude A_i(t) and phase φ_i(t) must vary randomly at **each sample** t, making traditional filtering approaches insufficient. The LSTM must learn the underlying frequency structure from noisy training data.

---

## 3. System Requirements

### 3.1 Functional Requirements

#### FR1: Data Generation
- **FR1.1**: Generate mixed noisy signal S(t) from 4 frequency components
- **FR1.2**: Apply random amplitude variations: A_i(t) ~ Uniform(0.8, 1.2) at each sample
- **FR1.3**: Apply random phase variations: φ_i(t) ~ Uniform(0, 2π) at each sample
- **FR1.4**: Create pure ground truth targets: Target_i(t) = sin(2π · f_i · t)
- **FR1.5**: Generate separate datasets for training (seed #1) and testing (seed #2)

#### FR2: Model Architecture
- **FR2.1**: Implement LSTM network for conditional regression
- **FR2.2**: Accept inputs: S[t] (scalar) + C (one-hot vector, size 4)
- **FR2.3**: Output: Pure sinusoid value Target_i[t] (scalar)
- **FR2.4**: Use sequence length L = 1 by default
- **FR2.5**: **CRITICAL**: Reset internal state (h_t, c_t) between consecutive samples

#### FR3: Training
- **FR3.1**: Train on 40,000 samples (10,000 time points × 4 frequencies)
- **FR3.2**: Optimize using MSE loss
- **FR3.3**: Ensure proper internal state management

#### FR4: Evaluation
- **FR4.1**: Calculate MSE on training set (seed #1)
- **FR4.2**: Calculate MSE on test set (seed #2)
- **FR4.3**: Verify generalization: MSE_test ≈ MSE_train
- **FR4.4**: Generate visualization graphs

### 3.2 Technical Specifications

#### Signal Parameters
| Parameter | Value |
|-----------|-------|
| Frequencies | f1=1Hz, f2=3Hz, f3=5Hz, f4=7Hz |
| Time Domain | 0-10 seconds |
| Sampling Rate (Fs) | 1000 Hz |
| Total Samples | 10,000 |

#### Noisy Signal Formula
```
Sinus_i^noisy(t) = A_i(t) · sin(2π · f_i · t + φ_i(t))
S(t) = (1/4) Σ_{i=1}^{4} Sinus_i^noisy(t)
```

#### Ground Truth Formula
```
Target_i(t) = sin(2π · f_i · t)
```

#### Dataset Structure
Each row in the training dataset:
- **Input**: [S[t], C1, C2, C3, C4] where C is one-hot encoded
- **Output**: Target_i[t] corresponding to the selected frequency

Example rows:
| Row | t (sec) | S[t] (noisy) | C (selection) | Target (pure) |
|-----|---------|--------------|---------------|---------------|
| 1 | 0.000 | 0.8124 | [1,0,0,0] | 0.0000 |
| 10001 | 0.000 | 0.8124 | [0,1,0,0] | 0.0000 |
| 10002 | 0.001 | 0.7932 | [0,1,0,0] | 0.0188 |

### 3.3 Performance Metrics

#### Success Criteria
1. **MSE_train**: Mean Squared Error on training set
   ```
   MSE_train = (1/40000) Σ_{j=1}^{40000} (LSTM(S_train[t], C) - Target[t])²
   ```

2. **MSE_test**: Mean Squared Error on test set (completely different noise)
   ```
   MSE_test = (1/40000) Σ_{j=1}^{40000} (LSTM(S_test[t], C) - Target[t])²
   ```

3. **Generalization Check**: MSE_test ≈ MSE_train (system generalizes well to new noise)

### 3.4 Visualization Requirements

#### Graph 1: Single Frequency Comparison
For one selected frequency (e.g., f2), display on the same plot:
1. Target_2 (pure sine) - as a line
2. LSTM Output - as points/dots
3. S (noisy mixed signal) - as background (green dots)

#### Graph 2: All 4 Extracted Frequencies
Four sub-plots, each showing the extraction for frequency f_i separately

---

## 4. Critical Implementation Notes

### 4.1 Internal State Management (PEDAGOGICAL FOCUS)

**The internal state of LSTM** consists of:
- **h_t**: Hidden State
- **c_t**: Cell State

These enable the network to learn **temporal dependencies** between samples.

#### For L = 1 (This Assignment):
**CRITICAL REQUIREMENT**: The internal state (h_t, c_t) **MUST be reset** (zeroed) between consecutive samples.

| Scenario | Required Action | Conceptual Explanation |
|----------|----------------|------------------------|
| Regular LSTM (L > 1) | Maintain state across sequence. Pass state as input to next step. | Network assumes sequential connection between steps. |
| This Assignment (L = 1) | **Reset state** at each batch/sample. Zero out both h_t and c_t. | Network learns patterns and passes through memory management, treating inputs as next step. |

**Justification**: This shows the network can learn to extract frequencies purely through its internal memory management, without relying on sequential patterns in the data presentation.

### 4.2 Alternative Approach (Optional Enhancement)

Students are welcome to work with **L ≠ 1** (e.g., L=10 or L=50 with sliding window):
- **Requirement**: Include detailed justification explaining the choice
- **Must explain**: How this contributes to LSTM's temporal advantage
- **Must explain**: How the output is handled in this case

---

## 5. Success Definition

The system is successful when:

1. ✅ Two datasets created (train and test) with noise varying at each sample
2. ✅ LSTM network built that receives [S[t], C] and returns pure Target_i[t]
3. ✅ Internal state properly managed (reset between samples for L=1)
4. ✅ Performance evaluated using MSE and visualizations
5. ✅ System generalizes to new noise (MSE_test ≈ MSE_train)

**The key to success** is proper internal state management and learning the frequency-periodic structure of Target_i while ignoring random noise.

---

## 6. Deliverables

1. Complete dataset generation code (train and test)
2. LSTM model implementation with proper state management
3. Training pipeline
4. Evaluation metrics (MSE_train, MSE_test)
5. Visualization graphs (as specified in section 3.4)
6. Documentation and analysis of results

---

## 7. References

Assignment PDF: L2-homework.pdf
Author: Dr. Segal Yoram
All Rights Reserved ©
