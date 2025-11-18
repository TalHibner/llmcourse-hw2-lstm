"""
Unit tests for visualizer module.

Tests cover:
- Training curve plotting
- Single frequency comparison plotting
- All frequencies plotting
- Metrics comparison plotting
- Signal comparison plotting
- File creation and saving
"""

import numpy as np
import pytest
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualizer import (
    plot_training_curve,
    plot_single_frequency_comparison,
    plot_all_frequencies,
    plot_metrics_comparison,
    plot_signal_comparison,
    create_all_visualizations
)


class TestPlotTrainingCurve:
    """Test training curve plotting."""

    def test_plot_training_curve_creates_file(self):
        """Test that plot creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'training_curve.png')

            history = {
                'epochs': list(range(1, 11)),
                'train_loss': [0.1 - i*0.005 for i in range(10)],
                'test_loss': [0.12 - i*0.005 for i in range(10)],
                'learning_rates': [0.001] * 10
            }

            plot_training_curve(history, save_path)

            assert os.path.exists(save_path)

    def test_plot_training_curve_without_lr(self):
        """Test plotting without learning rate history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'training_curve.png')

            history = {
                'epochs': list(range(1, 11)),
                'train_loss': [0.1 - i*0.005 for i in range(10)],
                'test_loss': [0.12 - i*0.005 for i in range(10)]
            }

            plot_training_curve(history, save_path)

            assert os.path.exists(save_path)

    def test_plot_training_curve_empty_history(self):
        """Test plotting with empty history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'training_curve.png')

            history = {}

            # Should handle empty history gracefully
            plot_training_curve(history, save_path)

            # File should not be created
            assert not os.path.exists(save_path)

    def test_plot_training_curve_creates_directory(self):
        """Test that function creates output directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'plots', 'training_curve.png')

            history = {
                'epochs': list(range(1, 6)),
                'train_loss': [0.05, 0.04, 0.03, 0.02, 0.01],
                'test_loss': [0.06, 0.05, 0.04, 0.03, 0.02]
            }

            plot_training_curve(history, save_path)

            assert os.path.exists(save_path)


class TestPlotSingleFrequencyComparison:
    """Test single frequency comparison plotting."""

    def test_plot_single_frequency_creates_file(self):
        """Test that plot creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'freq_comparison.png')

            time = np.linspace(0, 10, 10000)
            frequency = 3.0
            signal_pure = np.sin(2 * np.pi * frequency * time)
            signal_mixed = signal_pure + np.random.randn(len(time)) * 0.1
            predictions = signal_pure + np.random.randn(len(time)) * 0.05

            plot_single_frequency_comparison(
                time, signal_mixed, predictions, signal_pure,
                frequency, 1, save_path, num_samples=1000
            )

            assert os.path.exists(save_path)

    def test_plot_single_frequency_different_samples(self):
        """Test plotting with different number of samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'freq_comparison.png')

            time = np.linspace(0, 10, 5000)
            frequency = 5.0
            signal_pure = np.sin(2 * np.pi * frequency * time)
            signal_mixed = signal_pure + np.random.randn(len(time)) * 0.1
            predictions = signal_pure + np.random.randn(len(time)) * 0.05

            # Test with 500 samples
            plot_single_frequency_comparison(
                time, signal_mixed, predictions, signal_pure,
                frequency, 2, save_path, num_samples=500
            )

            assert os.path.exists(save_path)

    def test_plot_single_frequency_all_frequencies(self):
        """Test plotting for each frequency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frequencies = [1, 3, 5, 7]
            time = np.linspace(0, 10, 10000)

            for freq_idx, freq in enumerate(frequencies):
                save_path = os.path.join(tmpdir, f'freq_{freq_idx}.png')

                signal_pure = np.sin(2 * np.pi * freq * time)
                signal_mixed = signal_pure + np.random.randn(len(time)) * 0.1
                predictions = signal_pure + np.random.randn(len(time)) * 0.05

                plot_single_frequency_comparison(
                    time, signal_mixed, predictions, signal_pure,
                    freq, freq_idx, save_path
                )

                assert os.path.exists(save_path)


class TestPlotAllFrequencies:
    """Test all frequencies plotting."""

    def test_plot_all_frequencies_creates_file(self):
        """Test that plot creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'all_frequencies.png')

            time = np.linspace(0, 10, 10000)
            frequencies = [1, 3, 5, 7]

            results_by_freq = {}
            for freq_idx, freq in enumerate(frequencies):
                signal_pure = np.sin(2 * np.pi * freq * time)
                predictions = signal_pure + np.random.randn(len(time)) * 0.05

                results_by_freq[freq_idx] = {
                    'predictions': predictions,
                    'targets': signal_pure,
                    'signal': signal_pure + np.random.randn(len(time)) * 0.1,
                    'mse': 0.01 + freq_idx * 0.002
                }

            plot_all_frequencies(time, results_by_freq, frequencies, save_path)

            assert os.path.exists(save_path)

    def test_plot_all_frequencies_different_samples(self):
        """Test plotting with different number of samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'all_frequencies.png')

            time = np.linspace(0, 10, 5000)
            frequencies = [1, 3, 5, 7]

            results_by_freq = {}
            for freq_idx, freq in enumerate(frequencies):
                signal_pure = np.sin(2 * np.pi * freq * time)
                predictions = signal_pure + np.random.randn(len(time)) * 0.05

                results_by_freq[freq_idx] = {
                    'predictions': predictions,
                    'targets': signal_pure,
                    'signal': signal_pure,
                    'mse': 0.01
                }

            plot_all_frequencies(time, results_by_freq, frequencies, save_path, num_samples=500)

            assert os.path.exists(save_path)

    def test_plot_all_frequencies_structure(self):
        """Test that function handles correct data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'all_frequencies.png')

            time = np.linspace(0, 10, 1000)
            frequencies = [1, 3, 5, 7]

            results_by_freq = {
                0: {
                    'predictions': np.random.randn(1000),
                    'targets': np.random.randn(1000),
                    'signal': np.random.randn(1000),
                    'mse': 0.01
                },
                1: {
                    'predictions': np.random.randn(1000),
                    'targets': np.random.randn(1000),
                    'signal': np.random.randn(1000),
                    'mse': 0.02
                },
                2: {
                    'predictions': np.random.randn(1000),
                    'targets': np.random.randn(1000),
                    'signal': np.random.randn(1000),
                    'mse': 0.015
                },
                3: {
                    'predictions': np.random.randn(1000),
                    'targets': np.random.randn(1000),
                    'signal': np.random.randn(1000),
                    'mse': 0.018
                }
            }

            plot_all_frequencies(time, results_by_freq, frequencies, save_path)

            assert os.path.exists(save_path)


class TestPlotMetricsComparison:
    """Test metrics comparison plotting."""

    def test_plot_metrics_comparison_creates_file(self):
        """Test that plot creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'metrics_comparison.png')

            metrics = {
                'train_mse': 0.02,
                'test_mse': 0.025,
                'generalization_gap': 0.005,
                'generalization_ratio': 1.25,
                'generalizes_well': True
            }

            plot_metrics_comparison(metrics, save_path)

            assert os.path.exists(save_path)

    def test_plot_metrics_comparison_good_generalization(self):
        """Test plotting with good generalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'metrics_good.png')

            metrics = {
                'train_mse': 0.01,
                'test_mse': 0.012,
                'generalization_gap': 0.002,
                'generalization_ratio': 1.2,
                'generalizes_well': True
            }

            plot_metrics_comparison(metrics, save_path)

            assert os.path.exists(save_path)

    def test_plot_metrics_comparison_poor_generalization(self):
        """Test plotting with poor generalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'metrics_poor.png')

            metrics = {
                'train_mse': 0.01,
                'test_mse': 0.05,
                'generalization_gap': 0.04,
                'generalization_ratio': 5.0,
                'generalizes_well': False
            }

            plot_metrics_comparison(metrics, save_path)

            assert os.path.exists(save_path)


class TestPlotSignalComparison:
    """Test signal comparison plotting."""

    def test_plot_signal_comparison_creates_file(self):
        """Test that plot creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'signal_comparison.png')

            time = np.linspace(0, 10, 5000)
            frequency = 3.0
            signal_pure = np.sin(2 * np.pi * frequency * time)
            signal_noisy = signal_pure + np.random.randn(len(time)) * 0.2

            plot_signal_comparison(
                time, signal_noisy, signal_pure,
                frequency, save_path, num_samples=500
            )

            assert os.path.exists(save_path)

    def test_plot_signal_comparison_different_frequencies(self):
        """Test plotting for different frequencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frequencies = [1, 3, 5, 7]
            time = np.linspace(0, 10, 5000)

            for freq in frequencies:
                save_path = os.path.join(tmpdir, f'signal_{freq}Hz.png')

                signal_pure = np.sin(2 * np.pi * freq * time)
                signal_noisy = signal_pure + np.random.randn(len(time)) * 0.1

                plot_signal_comparison(
                    time, signal_noisy, signal_pure,
                    freq, save_path
                )

                assert os.path.exists(save_path)


class TestCreateAllVisualizations:
    """Test comprehensive visualization creation."""

    def test_create_all_visualizations(self):
        """Test creating all visualizations at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'plots')

            # Create dummy data
            time = np.linspace(0, 10, 10000)
            frequencies = [1, 3, 5, 7]

            history = {
                'epochs': list(range(1, 21)),
                'train_loss': [0.1 * np.exp(-0.05*i) + 0.01 for i in range(20)],
                'test_loss': [0.12 * np.exp(-0.05*i) + 0.015 for i in range(20)],
                'learning_rates': [0.001] * 20
            }

            train_metrics = {
                'train_mse': 0.015,
                'test_mse': 0.018,
                'generalization_gap': 0.003,
                'generalization_ratio': 1.2,
                'generalizes_well': True
            }

            test_results = {}
            for freq_idx, freq in enumerate(frequencies):
                signal_pure = np.sin(2 * np.pi * freq * time)
                noise = np.random.randn(len(time)) * 0.1
                signal_noisy = signal_pure + noise
                predictions = signal_pure + np.random.randn(len(time)) * 0.05

                test_results[freq_idx] = {
                    'predictions': predictions,
                    'targets': signal_pure,
                    'signal': signal_noisy,
                    'mse': 0.01 + freq_idx * 0.002
                }

            create_all_visualizations(
                history,
                train_metrics,
                test_results,
                time,
                frequencies,
                output_dir
            )

            # Check all files were created
            assert os.path.exists(os.path.join(output_dir, 'training_curve.png'))
            assert os.path.exists(os.path.join(output_dir, 'metrics_comparison.png'))
            assert os.path.exists(os.path.join(output_dir, 'freq_comparison.png'))
            assert os.path.exists(os.path.join(output_dir, 'all_frequencies.png'))

    def test_create_all_visualizations_creates_directory(self):
        """Test that function creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'new_plots_dir')

            time = np.linspace(0, 10, 1000)
            frequencies = [1, 3, 5, 7]

            history = {
                'epochs': [1, 2, 3],
                'train_loss': [0.1, 0.08, 0.06],
                'test_loss': [0.12, 0.1, 0.08]
            }

            train_metrics = {
                'train_mse': 0.02,
                'test_mse': 0.025,
                'generalization_gap': 0.005,
                'generalization_ratio': 1.25,
                'generalizes_well': True
            }

            test_results = {}
            for freq_idx in range(4):
                test_results[freq_idx] = {
                    'predictions': np.random.randn(1000),
                    'targets': np.random.randn(1000),
                    'signal': np.random.randn(1000),
                    'mse': 0.01
                }

            create_all_visualizations(
                history,
                train_metrics,
                test_results,
                time,
                frequencies,
                output_dir
            )

            # Directory should be created
            assert os.path.exists(output_dir)


class TestPlotFileFormats:
    """Test that plots are saved in correct format."""

    def test_png_format(self):
        """Test that plots are saved as PNG files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.png')

            history = {
                'epochs': [1, 2, 3],
                'train_loss': [0.1, 0.08, 0.06],
                'test_loss': [0.12, 0.1, 0.08]
            }

            plot_training_curve(history, save_path)

            # Check file extension
            assert save_path.endswith('.png')
            assert os.path.exists(save_path)

    def test_file_not_empty(self):
        """Test that saved plot files are not empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.png')

            history = {
                'epochs': [1, 2, 3],
                'train_loss': [0.1, 0.08, 0.06],
                'test_loss': [0.12, 0.1, 0.08]
            }

            plot_training_curve(history, save_path)

            # File should have content
            assert os.path.getsize(save_path) > 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
