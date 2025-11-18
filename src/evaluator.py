"""
Evaluation Module for LSTM Frequency Extraction

This module handles model evaluation and prediction generation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple
import json


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model and collect predictions.

    CRITICAL: Hidden state is reset for each batch (hidden=None)

    Args:
        model: LSTM model
        dataloader: Data loader
        device: 'cpu' or 'cuda'

    Returns:
        mse: Mean Squared Error
        predictions: Model predictions
        targets: Ground truth targets
    """
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_targets = []
    total_squared_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for S, C, targets in dataloader:
            S = S.to(device)
            C = C.to(device)
            targets = targets.to(device)

            # Prepare input
            inputs = torch.cat([S.unsqueeze(-1), C], dim=-1)
            inputs = inputs.unsqueeze(1)

            # ✅ CRITICAL: Reset hidden state
            hidden = None

            # Forward pass
            outputs, _ = model(inputs, hidden)

            # Collect predictions and targets
            preds = outputs.squeeze().cpu().numpy()
            targs = targets.cpu().numpy()

            all_predictions.extend(preds)
            all_targets.extend(targs)

            # Accumulate squared error
            squared_error = ((preds - targs) ** 2).sum()
            total_squared_error += squared_error
            num_samples += len(targs)

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Calculate MSE
    mse = total_squared_error / num_samples

    return mse, predictions, targets


def evaluate_by_frequency(
    model: nn.Module,
    S: np.ndarray,
    C: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 64,
    device: str = 'cpu'
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Evaluate model and organize predictions by frequency.

    Args:
        model: LSTM model
        S: Signal values (40000,)
        C: One-hot vectors (40000, 4)
        targets: Ground truth (40000,)
        batch_size: Batch size for evaluation
        device: 'cpu' or 'cuda'

    Returns:
        results: Dictionary with frequency-separated predictions
    """
    from src.model import create_dataloader

    # Create dataloader
    dataloader = create_dataloader(S, C, targets, batch_size=batch_size, shuffle=False)

    # Get all predictions
    mse, predictions, targets_array = evaluate_model(model, dataloader, device)

    # Organize by frequency
    # Each frequency has 10,000 consecutive samples
    num_samples_per_freq = 10000
    results = {}

    for freq_idx in range(4):
        start_idx = freq_idx * num_samples_per_freq
        end_idx = (freq_idx + 1) * num_samples_per_freq

        results[freq_idx] = {
            'predictions': predictions[start_idx:end_idx],
            'targets': targets_array[start_idx:end_idx],
            'signal': S[start_idx:end_idx],
            'mse': np.mean((predictions[start_idx:end_idx] - targets_array[start_idx:end_idx]) ** 2)
        }

    return results


def compute_metrics(
    train_mse: float,
    test_mse: float
) -> Dict:
    """
    Compute evaluation metrics and generalization check.

    Args:
        train_mse: MSE on training set
        test_mse: MSE on test set

    Returns:
        metrics: Dictionary of metrics
    """
    generalization_gap = abs(test_mse - train_mse)
    generalization_ratio = test_mse / train_mse if train_mse > 0 else float('inf')

    metrics = {
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'generalization_gap': float(generalization_gap),
        'generalization_ratio': float(generalization_ratio),
        'generalizes_well': generalization_gap < 0.01  # Threshold from assignment
    }

    return metrics


def print_metrics(metrics: Dict):
    """
    Print evaluation metrics in a formatted way.

    Args:
        metrics: Metrics dictionary
    """
    print("="*70)
    print("Evaluation Metrics")
    print("="*70)
    print(f"Training MSE:         {metrics['train_mse']:.6f}")
    print(f"Test MSE:             {metrics['test_mse']:.6f}")
    print(f"Generalization Gap:   {metrics['generalization_gap']:.6f}")
    print(f"Generalization Ratio: {metrics['generalization_ratio']:.4f}")
    print()

    if metrics['generalizes_well']:
        print("✓ Model generalizes well to new noise!")
        print(f"  (Gap = {metrics['generalization_gap']:.6f} < 0.01)")
    else:
        print("⚠ Model may be overfitting")
        print(f"  (Gap = {metrics['generalization_gap']:.6f} >= 0.01)")

    print("="*70)


def save_metrics(metrics: Dict, filename: str = 'results/metrics.json'):
    """
    Save metrics to JSON file.

    Args:
        metrics: Metrics dictionary
        filename: Output filename
    """
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {filename}")


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    S: np.ndarray,
    time: np.ndarray,
    filename: str = 'results/predictions.npz'
):
    """
    Save predictions to file.

    Args:
        predictions: Model predictions
        targets: Ground truth
        S: Input signal
        time: Time array
        filename: Output filename
    """
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    np.savez(
        filename,
        predictions=predictions,
        targets=targets,
        S=S,
        time=time
    )

    print(f"Predictions saved to {filename}")


if __name__ == "__main__":
    """
    Test the evaluation module
    """
    from src.model import FrequencyExtractorLSTM, create_dataloader
    from src.data_generator import load_dataset
    import os

    print("="*70)
    print("Testing Evaluation Module")
    print("="*70)

    # Check if datasets exist
    if not os.path.exists('data/test_dataset.npz'):
        print("Error: Test dataset not found. Run data_generator.py first.")
        exit(1)

    # Load test data
    test_data = load_dataset('data/test_dataset.npz')

    # Create a simple model (untrained, just for testing)
    model = FrequencyExtractorLSTM(input_size=5, hidden_size=32, num_layers=1)

    # Create dataloader
    dataloader = create_dataloader(
        test_data['S'][:1000],  # Use subset for testing
        test_data['C'][:1000],
        test_data['targets'][:1000],
        batch_size=64,
        shuffle=False
    )

    # Evaluate
    print("\nEvaluating model (untrained, random predictions)...")
    mse, predictions, targets = evaluate_model(model, dataloader, device='cpu')

    print(f"\nMSE: {mse:.6f}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample targets: {targets[:5]}")

    # Test metrics computation
    print("\n" + "-"*70)
    print("Testing metrics computation")
    print("-"*70)

    metrics = compute_metrics(train_mse=0.02, test_mse=0.025)
    print_metrics(metrics)

    print("\n" + "="*70)
    print("Evaluation Test Complete!")
    print("="*70)
