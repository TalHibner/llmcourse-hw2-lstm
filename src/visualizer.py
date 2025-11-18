"""
Visualization Module for LSTM Frequency Extraction

This module creates visualizations as specified in the assignment:
1. Training curve (loss over epochs)
2. Single frequency detailed comparison (Graph 1)
3. All four frequencies extraction (Graph 2)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os


def plot_training_curve(
    history: Dict,
    save_path: str = 'results/plots/training_curve.png'
):
    """
    Plot training and test loss over epochs.

    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    if not history or 'epochs' not in history:
        print("Warning: No training history available. Skipping training curve plot.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history['epochs']
    train_loss = history['train_loss']
    test_loss = history['test_loss']

    # Loss curves
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_loss, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Learning rate
    if 'learning_rates' in history:
        ax2.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curve saved to {save_path}")
    plt.close()


def plot_single_frequency_comparison(
    time: np.ndarray,
    signal_mixed: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    frequency: float,
    frequency_idx: int,
    save_path: str = 'results/plots/freq_comparison.png',
    num_samples: int = 1000
):
    """
    Plot detailed comparison for a single frequency (Graph 1 from assignment).

    Shows:
    1. Mixed noisy signal S (background, green dots)
    2. Pure target (blue line)
    3. LSTM predictions (red dots)

    Args:
        time: Time array
        signal_mixed: Mixed noisy signal S
        predictions: LSTM predictions
        targets: Ground truth pure sinusoid
        frequency: Frequency value (e.g., 3 Hz)
        frequency_idx: Frequency index (0-3)
        save_path: Path to save figure
        num_samples: Number of samples to plot (for readability)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Use subset for better visualization
    time_plot = time[:num_samples]
    signal_plot = signal_mixed[:num_samples]
    pred_plot = predictions[:num_samples]
    target_plot = targets[:num_samples]

    plt.figure(figsize=(16, 6))

    # Background: Mixed noisy signal (green dots, small, transparent)
    plt.scatter(time_plot, signal_plot, c='green', alpha=0.2, s=5,
                label='Mixed Noisy Signal S', zorder=1)

    # Ground truth: Pure target (blue line)
    plt.plot(time_plot, target_plot, 'b-', linewidth=2.5,
             label=f'Target (Pure f{frequency_idx+1} = {frequency} Hz)', zorder=3)

    # LSTM predictions (red dots)
    plt.scatter(time_plot, pred_plot, c='red', s=15, alpha=0.7,
                label='LSTM Predictions', zorder=2)

    plt.xlabel('Time (seconds)', fontsize=13)
    plt.ylabel('Amplitude', fontsize=13)
    plt.title(f'Frequency Extraction: f{frequency_idx+1} = {frequency} Hz',
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(time_plot[0], time_plot[-1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Single frequency comparison saved to {save_path}")
    plt.close()


def plot_all_frequencies(
    time: np.ndarray,
    results_by_freq: Dict,
    frequencies: List[float] = [1, 3, 5, 7],
    save_path: str = 'results/plots/all_frequencies.png',
    num_samples: int = 1000
):
    """
    Plot all four frequency extractions (Graph 2 from assignment).

    Creates 2x2 subplot showing extraction for each frequency.

    Args:
        time: Time array
        results_by_freq: Dictionary with predictions/targets per frequency
        frequencies: List of frequency values
        save_path: Path to save figure
        num_samples: Number of samples to plot per subplot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    time_plot = time[:num_samples]

    for freq_idx, (ax, freq) in enumerate(zip(axes, frequencies)):
        freq_data = results_by_freq[freq_idx]

        predictions = freq_data['predictions'][:num_samples]
        targets = freq_data['targets'][:num_samples]
        mse = freq_data['mse']

        # Plot target (blue line)
        ax.plot(time_plot, targets, 'b-', linewidth=2, label='Target (Pure)', alpha=0.9)

        # Plot predictions (red dashed line)
        ax.plot(time_plot, predictions, 'r--', linewidth=1.5, label='LSTM Output', alpha=0.8)

        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(f'f{freq_idx+1} = {freq} Hz (MSE: {mse:.6f})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_plot[0], time_plot[-1])

    plt.suptitle('LSTM Frequency Extraction - All Frequencies',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"All frequencies plot saved to {save_path}")
    plt.close()


def plot_metrics_comparison(
    metrics: Dict,
    save_path: str = 'results/plots/metrics_comparison.png'
):
    """
    Plot bar chart comparing train and test MSE.

    Args:
        metrics: Metrics dictionary
        save_path: Path to save figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Training MSE', 'Test MSE']
    values = [metrics['train_mse'], metrics['test_mse']]
    colors = ['#1f77b4', '#ff7f0e']

    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.6f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('MSE', fontsize=13)
    ax.set_title('Model Performance: Train vs Test', fontsize=15, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Add generalization info
    gap = metrics['generalization_gap']
    generalization_text = f"Generalization Gap: {gap:.6f}"
    if metrics['generalizes_well']:
        generalization_text += " ✓ (Good)"
        color = 'green'
    else:
        generalization_text += " ⚠ (Overfitting)"
        color = 'orange'

    ax.text(0.5, 0.95, generalization_text,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Metrics comparison saved to {save_path}")
    plt.close()


def plot_signal_comparison(
    time: np.ndarray,
    signal_noisy: np.ndarray,
    signal_pure: np.ndarray,
    frequency: float,
    save_path: str = 'results/plots/signal_comparison.png',
    num_samples: int = 500
):
    """
    Compare noisy vs pure signal for a single frequency component.

    Args:
        time: Time array
        signal_noisy: Noisy sinusoid
        signal_pure: Pure sinusoid
        frequency: Frequency value
        save_path: Path to save figure
        num_samples: Number of samples to plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    time_plot = time[:num_samples]
    noisy_plot = signal_noisy[:num_samples]
    pure_plot = signal_pure[:num_samples]

    plt.figure(figsize=(14, 6))

    plt.plot(time_plot, pure_plot, 'b-', linewidth=2, label='Pure Signal', alpha=0.8)
    plt.scatter(time_plot, noisy_plot, c='red', s=10, alpha=0.5, label='Noisy Signal')

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title(f'Signal Comparison: {frequency} Hz (Noisy vs Pure)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Signal comparison saved to {save_path}")
    plt.close()


def create_all_visualizations(
    history: Dict,
    train_metrics: Dict,
    test_results: Dict,
    time: np.ndarray,
    frequencies: List[float] = [1, 3, 5, 7],
    output_dir: str = 'results/plots'
):
    """
    Create all required visualizations.

    Args:
        history: Training history
        train_metrics: Training metrics
        test_results: Test results by frequency
        time: Time array
        frequencies: List of frequencies
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("Generating Visualizations")
    print("="*70)

    # 1. Training curve
    print("\n1. Generating training curve...")
    plot_training_curve(history, f'{output_dir}/training_curve.png')

    # 2. Metrics comparison
    print("\n2. Generating metrics comparison...")
    plot_metrics_comparison(train_metrics, f'{output_dir}/metrics_comparison.png')

    # 3. Single frequency detailed comparison (f2 = 3 Hz)
    print("\n3. Generating single frequency comparison (f2 = 3 Hz)...")
    freq_idx = 1  # f2 = 3 Hz
    plot_single_frequency_comparison(
        time,
        test_results[freq_idx]['signal'],
        test_results[freq_idx]['predictions'],
        test_results[freq_idx]['targets'],
        frequencies[freq_idx],
        freq_idx,
        f'{output_dir}/freq_comparison.png',
        num_samples=1000
    )

    # 4. All frequencies
    print("\n4. Generating all frequencies plot...")
    plot_all_frequencies(
        time,
        test_results,
        frequencies,
        f'{output_dir}/all_frequencies.png',
        num_samples=1000
    )

    print("\n" + "="*70)
    print("All Visualizations Complete!")
    print("="*70)
    print(f"Plots saved to: {output_dir}/")


if __name__ == "__main__":
    """
    Test visualization module
    """
    print("="*70)
    print("Testing Visualization Module")
    print("="*70)

    # Create dummy data
    time = np.linspace(0, 10, 10000)
    frequencies = [1, 3, 5, 7]

    # Dummy training history
    history = {
        'epochs': list(range(1, 51)),
        'train_loss': [0.1 * np.exp(-0.05*i) + 0.01 for i in range(50)],
        'test_loss': [0.12 * np.exp(-0.05*i) + 0.015 for i in range(50)],
        'learning_rates': [0.001] * 50
    }

    # Dummy metrics
    metrics = {
        'train_mse': 0.015,
        'test_mse': 0.018,
        'generalization_gap': 0.003,
        'generalization_ratio': 1.2,
        'generalizes_well': True
    }

    # Dummy frequency results
    results_by_freq = {}
    for freq_idx, freq in enumerate(frequencies):
        signal_pure = np.sin(2 * np.pi * freq * time)
        noise = np.random.randn(len(time)) * 0.1
        signal_noisy = signal_pure + noise
        predictions = signal_pure + np.random.randn(len(time)) * 0.05

        results_by_freq[freq_idx] = {
            'predictions': predictions,
            'targets': signal_pure,
            'signal': signal_noisy,
            'mse': 0.01 + freq_idx * 0.002
        }

    # Create visualizations
    print("\nCreating test visualizations...")
    os.makedirs('results/plots', exist_ok=True)

    plot_training_curve(history)
    plot_metrics_comparison(metrics)
    plot_single_frequency_comparison(
        time, results_by_freq[1]['signal'],
        results_by_freq[1]['predictions'],
        results_by_freq[1]['targets'],
        frequencies[1], 1
    )
    plot_all_frequencies(time, results_by_freq, frequencies)

    print("\n" + "="*70)
    print("Visualization Test Complete!")
    print("="*70)
