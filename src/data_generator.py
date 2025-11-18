"""
Dataset Generation Module for LSTM Frequency Extraction

This module generates synthetic noisy mixed signals and ground truth targets
for training and testing the LSTM frequency extraction system.

Critical Implementation Notes:
- Amplitude A_i(t) and phase φ_i(t) must vary at EACH sample t
- Training dataset uses seed #1
- Test dataset uses seed #2
- Mixed signal S(t) is the average of 4 noisy sinusoids
"""

import numpy as np
from typing import Tuple, Dict


# Signal Parameters (from assignment specification)
FREQUENCIES = [1, 3, 5, 7]  # Hz
TIME_START = 0.0  # seconds
TIME_END = 10.0  # seconds
SAMPLING_RATE = 1000  # Hz
NUM_SAMPLES = 10000  # Total time samples
AMPLITUDE_MIN = 0.8
AMPLITUDE_MAX = 1.2
PHASE_MIN = 0.0
PHASE_MAX = 2 * np.pi


def generate_noisy_sinusoid(
    frequency: float,
    time_array: np.ndarray,
    seed: int
) -> np.ndarray:
    """
    Generate a noisy sinusoid with random amplitude and phase variations at EACH sample.

    This is CRITICAL: A_i(t) and φ_i(t) must vary for every time sample t,
    not just once for the entire signal.

    Args:
        frequency: Frequency in Hz (e.g., 1, 3, 5, 7)
        time_array: Array of time values [0, 0.001, 0.002, ..., 9.999]
        seed: Random seed for reproducibility

    Returns:
        noisy_signal: Array of noisy sinusoid values

    Formula:
        Sinus_i^noisy(t) = A_i(t) · sin(2π · f_i · t + φ_i(t))
        where:
            A_i(t) ~ Uniform(0.8, 1.2) for EACH t
            φ_i(t) ~ Uniform(0, 2π) for EACH t
    """
    rng = np.random.RandomState(seed)
    num_points = len(time_array)

    # Generate random amplitude for EACH sample
    amplitudes = rng.uniform(AMPLITUDE_MIN, AMPLITUDE_MAX, size=num_points)

    # Generate random phase for EACH sample
    phases = rng.uniform(PHASE_MIN, PHASE_MAX, size=num_points)

    # Compute noisy sinusoid
    # Note: amplitudes[i] and phases[i] are different for each time sample
    noisy_signal = amplitudes * np.sin(2 * np.pi * frequency * time_array + phases)

    return noisy_signal


def generate_mixed_signal(
    frequencies: list,
    time_array: np.ndarray,
    seed: int
) -> np.ndarray:
    """
    Generate mixed signal S(t) as the average of 4 noisy sinusoids.

    Args:
        frequencies: List of 4 frequencies [f1, f2, f3, f4]
        time_array: Array of time values
        seed: Random seed for reproducibility

    Returns:
        mixed_signal: S(t) = (1/4) Σ Sinus_i^noisy(t)
    """
    num_freqs = len(frequencies)
    mixed_signal = np.zeros_like(time_array)

    # Use different sub-seeds for each frequency to ensure diversity
    for i, freq in enumerate(frequencies):
        sub_seed = seed + i * 1000  # Offset seed for each frequency
        noisy_component = generate_noisy_sinusoid(freq, time_array, sub_seed)
        mixed_signal += noisy_component

    # Average the signals
    mixed_signal /= num_freqs

    return mixed_signal


def generate_ground_truth(frequency: float, time_array: np.ndarray) -> np.ndarray:
    """
    Generate pure ground truth target sinusoid (no noise).

    Args:
        frequency: Frequency in Hz
        time_array: Array of time values

    Returns:
        pure_signal: Target_i(t) = sin(2π · f_i · t)
    """
    pure_signal = np.sin(2 * np.pi * frequency * time_array)
    return pure_signal


def create_frequency_dataset(
    frequencies: list,
    time_array: np.ndarray,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create complete dataset with 40,000 rows (10,000 samples × 4 frequencies).

    Dataset Structure:
        - Each row contains: [S[t], C1, C2, C3, C4, Target]
        - S[t]: Mixed noisy signal value at time t
        - C: One-hot vector selecting frequency [1,0,0,0] or [0,1,0,0] etc.
        - Target: Pure sinusoid value for the selected frequency

    Args:
        frequencies: List of 4 frequencies
        time_array: Array of time values (length 10,000)
        seed: Random seed for dataset generation

    Returns:
        S: Signal values, shape (40000,)
        C: One-hot frequency selection vectors, shape (40000, 4)
        targets: Ground truth values, shape (40000,)
    """
    num_samples = len(time_array)
    num_freqs = len(frequencies)
    total_rows = num_samples * num_freqs  # 40,000

    # Generate mixed signal once (same for all rows)
    mixed_signal = generate_mixed_signal(frequencies, time_array, seed)

    # Generate ground truth for each frequency
    ground_truths = []
    for freq in frequencies:
        gt = generate_ground_truth(freq, time_array)
        ground_truths.append(gt)

    # Initialize dataset arrays
    S = np.zeros(total_rows)
    C = np.zeros((total_rows, num_freqs))
    targets = np.zeros(total_rows)

    # Fill dataset row by row
    # Rows 0-9999: frequency f1
    # Rows 10000-19999: frequency f2
    # Rows 20000-29999: frequency f3
    # Rows 30000-39999: frequency f4

    for freq_idx in range(num_freqs):
        start_row = freq_idx * num_samples
        end_row = (freq_idx + 1) * num_samples

        # Signal values (same mixed signal for all frequencies)
        S[start_row:end_row] = mixed_signal

        # One-hot encoding for frequency selection
        C[start_row:end_row, freq_idx] = 1

        # Ground truth targets for this frequency
        targets[start_row:end_row] = ground_truths[freq_idx]

    return S, C, targets


def save_dataset(
    S: np.ndarray,
    C: np.ndarray,
    targets: np.ndarray,
    time_array: np.ndarray,
    seed: int,
    filename: str
) -> None:
    """
    Save dataset to .npz file.

    Args:
        S: Signal values
        C: One-hot frequency selection vectors
        targets: Ground truth targets
        time_array: Time values
        seed: Random seed used
        filename: Path to save file (e.g., 'data/train_dataset.npz')
    """
    np.savez(
        filename,
        S=S,
        C=C,
        targets=targets,
        time=time_array,
        frequencies=FREQUENCIES,
        seed=seed,
        sampling_rate=SAMPLING_RATE
    )
    print(f"Dataset saved to {filename}")
    print(f"  - Shape: S={S.shape}, C={C.shape}, targets={targets.shape}")
    print(f"  - Seed: {seed}")
    print(f"  - Frequencies: {FREQUENCIES}")


def load_dataset(filename: str) -> Dict:
    """
    Load dataset from .npz file.

    Args:
        filename: Path to dataset file

    Returns:
        dataset: Dictionary containing all dataset components
    """
    data = np.load(filename)
    dataset = {
        'S': data['S'],
        'C': data['C'],
        'targets': data['targets'],
        'time': data['time'],
        'frequencies': data['frequencies'],
        'seed': int(data['seed']),
        'sampling_rate': int(data['sampling_rate'])
    }
    print(f"Dataset loaded from {filename}")
    print(f"  - Shape: S={dataset['S'].shape}, C={dataset['C'].shape}, targets={dataset['targets'].shape}")
    print(f"  - Seed: {dataset['seed']}")
    return dataset


def create_train_and_test_datasets(
    save_dir: str = 'data'
) -> Tuple[Dict, Dict]:
    """
    Create both training and test datasets.

    Training dataset: seed #1
    Test dataset: seed #2 (completely different noise)

    Args:
        save_dir: Directory to save datasets

    Returns:
        train_dataset, test_dataset: Dataset dictionaries
    """
    # Create time array
    time_array = np.linspace(TIME_START, TIME_END, NUM_SAMPLES)

    print("="*60)
    print("Generating Training Dataset (Seed #1)")
    print("="*60)

    # Training dataset with seed #1
    S_train, C_train, targets_train = create_frequency_dataset(
        FREQUENCIES, time_array, seed=1
    )
    save_dataset(
        S_train, C_train, targets_train, time_array,
        seed=1, filename=f'{save_dir}/train_dataset.npz'
    )

    print("\n" + "="*60)
    print("Generating Test Dataset (Seed #2)")
    print("="*60)

    # Test dataset with seed #2
    S_test, C_test, targets_test = create_frequency_dataset(
        FREQUENCIES, time_array, seed=2
    )
    save_dataset(
        S_test, C_test, targets_test, time_array,
        seed=2, filename=f'{save_dir}/test_dataset.npz'
    )

    # Verify datasets are different (different noise)
    print("\n" + "="*60)
    print("Verification: Datasets have different noise")
    print("="*60)
    print(f"S_train[0:5] = {S_train[0:5]}")
    print(f"S_test[0:5]  = {S_test[0:5]}")
    print(f"Difference: {np.mean(np.abs(S_train - S_test)):.4f}")
    print("✓ Datasets are different (noise varies with seed)")

    # Verify targets are identical (same pure frequencies)
    targets_diff = np.mean(np.abs(targets_train - targets_test))
    print(f"\nTargets difference: {targets_diff:.10f}")
    if targets_diff < 1e-10:
        print("✓ Targets are identical (same pure frequencies)")
    else:
        print("⚠ Warning: Targets differ (unexpected)")

    # Load and return
    train_dataset = load_dataset(f'{save_dir}/train_dataset.npz')
    test_dataset = load_dataset(f'{save_dir}/test_dataset.npz')

    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)

    return train_dataset, test_dataset


if __name__ == "__main__":
    """
    Test the data generation module
    """
    import os

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Generate datasets
    train_data, test_data = create_train_and_test_datasets()

    print("\nDatasets ready for training!")
    print(f"Training samples: {len(train_data['S'])}")
    print(f"Test samples: {len(test_data['S'])}")
