"""
Unit tests for evaluator module.

Tests cover:
- Model evaluation
- Frequency-separated evaluation
- Metrics computation
- Saving/loading predictions
"""

import torch
import numpy as np
import pytest
import tempfile
import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluator import (
    evaluate_model,
    evaluate_by_frequency,
    compute_metrics,
    print_metrics,
    save_metrics,
    save_predictions
)
from model import FrequencyExtractorLSTM, create_dataloader


class TestEvaluateModel:
    """Test model evaluation functionality."""

    def test_evaluate_model_basic(self):
        """Test basic model evaluation."""
        S = np.random.randn(200)
        C = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
        targets = np.random.randn(200)

        dataloader = create_dataloader(S, C, targets, batch_size=32, shuffle=False)

        model = FrequencyExtractorLSTM(hidden_size=16)

        mse, predictions, targets_out = evaluate_model(model, dataloader, device='cpu')

        assert isinstance(mse, float)
        assert mse > 0
        assert predictions.shape == (200,)
        assert targets_out.shape == (200,)

    def test_evaluate_model_predictions_shape(self):
        """Test that predictions have correct shape."""
        S = np.random.randn(500)
        C = np.random.randint(0, 2, size=(500, 4)).astype(np.float32)
        targets = np.random.randn(500)

        dataloader = create_dataloader(S, C, targets, batch_size=64, shuffle=False)

        model = FrequencyExtractorLSTM(hidden_size=32)

        _, predictions, targets_out = evaluate_model(model, dataloader, device='cpu')

        assert len(predictions) == 500
        assert len(targets_out) == 500

    def test_evaluate_model_no_gradients(self):
        """Test that evaluation doesn't compute gradients."""
        S = np.random.randn(100)
        C = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
        targets = np.random.randn(100)

        dataloader = create_dataloader(S, C, targets, batch_size=32, shuffle=False)

        model = FrequencyExtractorLSTM(hidden_size=16)

        # Get initial weights
        initial_weight = model.fc.weight.clone()

        # Evaluate
        evaluate_model(model, dataloader, device='cpu')

        # Weights should not change
        assert torch.allclose(initial_weight, model.fc.weight)

    def test_evaluate_model_eval_mode(self):
        """Test that model is in eval mode during evaluation."""
        S = np.random.randn(100)
        C = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
        targets = np.random.randn(100)

        dataloader = create_dataloader(S, C, targets, batch_size=32, shuffle=False)

        model = FrequencyExtractorLSTM(hidden_size=16)
        model.train()  # Set to train mode

        evaluate_model(model, dataloader, device='cpu')

        # Should be in eval mode after
        assert model.training == False

    def test_evaluate_model_mse_calculation(self):
        """Test MSE calculation correctness."""
        # Create simple test case
        S = np.ones(100)
        C = np.eye(4, 4)[np.random.randint(0, 4, 100)]
        targets = np.zeros(100)

        dataloader = create_dataloader(S, C, targets, batch_size=32, shuffle=False)

        model = FrequencyExtractorLSTM(hidden_size=16)

        mse, predictions, targets_out = evaluate_model(model, dataloader, device='cpu')

        # Manual MSE calculation
        manual_mse = np.mean((predictions - targets_out) ** 2)

        assert np.isclose(mse, manual_mse, rtol=1e-5)


class TestEvaluateByFrequency:
    """Test frequency-separated evaluation."""

    def test_evaluate_by_frequency_structure(self):
        """Test that results have correct structure."""
        # Create dataset with 40,000 samples (10,000 per frequency)
        S = np.random.randn(40000)
        C = np.zeros((40000, 4))
        targets = np.random.randn(40000)

        # Set one-hot encoding for each frequency
        for i in range(4):
            C[i*10000:(i+1)*10000, i] = 1

        model = FrequencyExtractorLSTM(hidden_size=16)

        results = evaluate_by_frequency(
            model, S, C, targets,
            batch_size=64,
            device='cpu'
        )

        # Check structure
        assert len(results) == 4
        for freq_idx in range(4):
            assert freq_idx in results
            assert 'predictions' in results[freq_idx]
            assert 'targets' in results[freq_idx]
            assert 'signal' in results[freq_idx]
            assert 'mse' in results[freq_idx]

    def test_evaluate_by_frequency_shapes(self):
        """Test that frequency-separated results have correct shapes."""
        S = np.random.randn(40000)
        C = np.zeros((40000, 4))
        targets = np.random.randn(40000)

        for i in range(4):
            C[i*10000:(i+1)*10000, i] = 1

        model = FrequencyExtractorLSTM(hidden_size=16)

        results = evaluate_by_frequency(model, S, C, targets, batch_size=64, device='cpu')

        # Each frequency should have 10,000 samples
        for freq_idx in range(4):
            assert results[freq_idx]['predictions'].shape == (10000,)
            assert results[freq_idx]['targets'].shape == (10000,)
            assert results[freq_idx]['signal'].shape == (10000,)

    def test_evaluate_by_frequency_mse_values(self):
        """Test that MSE values are computed for each frequency."""
        S = np.random.randn(40000)
        C = np.zeros((40000, 4))
        targets = np.random.randn(40000)

        for i in range(4):
            C[i*10000:(i+1)*10000, i] = 1

        model = FrequencyExtractorLSTM(hidden_size=16)

        results = evaluate_by_frequency(model, S, C, targets, batch_size=64, device='cpu')

        for freq_idx in range(4):
            mse = results[freq_idx]['mse']
            assert isinstance(mse, (float, np.floating))
            assert mse >= 0

    def test_evaluate_by_frequency_mse_correctness(self):
        """Test that per-frequency MSE is calculated correctly."""
        S = np.random.randn(40000)
        C = np.zeros((40000, 4))
        targets = np.random.randn(40000)

        for i in range(4):
            C[i*10000:(i+1)*10000, i] = 1

        model = FrequencyExtractorLSTM(hidden_size=16)

        results = evaluate_by_frequency(model, S, C, targets, batch_size=64, device='cpu')

        # Check MSE calculation for first frequency
        freq_0_predictions = results[0]['predictions']
        freq_0_targets = results[0]['targets']
        freq_0_mse = results[0]['mse']

        manual_mse = np.mean((freq_0_predictions - freq_0_targets) ** 2)

        assert np.isclose(freq_0_mse, manual_mse, rtol=1e-5)


class TestComputeMetrics:
    """Test metrics computation."""

    def test_compute_metrics_basic(self):
        """Test basic metrics computation."""
        train_mse = 0.02
        test_mse = 0.025

        metrics = compute_metrics(train_mse, test_mse)

        assert 'train_mse' in metrics
        assert 'test_mse' in metrics
        assert 'generalization_gap' in metrics
        assert 'generalization_ratio' in metrics
        assert 'generalizes_well' in metrics

    def test_compute_metrics_values(self):
        """Test metrics calculation correctness."""
        train_mse = 0.01
        test_mse = 0.015

        metrics = compute_metrics(train_mse, test_mse)

        assert metrics['train_mse'] == 0.01
        assert metrics['test_mse'] == 0.015
        assert metrics['generalization_gap'] == 0.005
        assert np.isclose(metrics['generalization_ratio'], 1.5)

    def test_compute_metrics_generalizes_well_true(self):
        """Test generalization check when gap is small."""
        train_mse = 0.02
        test_mse = 0.025  # Gap = 0.005 < 0.01

        metrics = compute_metrics(train_mse, test_mse)

        assert metrics['generalizes_well'] == True

    def test_compute_metrics_generalizes_well_false(self):
        """Test generalization check when gap is large."""
        train_mse = 0.01
        test_mse = 0.05  # Gap = 0.04 > 0.01

        metrics = compute_metrics(train_mse, test_mse)

        assert metrics['generalizes_well'] == False

    def test_compute_metrics_threshold(self):
        """Test generalization threshold at boundary."""
        train_mse = 0.02
        test_mse = 0.03  # Gap = 0.01 (exactly at threshold)

        metrics = compute_metrics(train_mse, test_mse)

        # Gap = 0.01 is not < 0.01, so should be False
        assert metrics['generalizes_well'] == False

    def test_compute_metrics_ratio_calculation(self):
        """Test generalization ratio calculation."""
        train_mse = 0.04
        test_mse = 0.08

        metrics = compute_metrics(train_mse, test_mse)

        assert metrics['generalization_ratio'] == 2.0

    def test_compute_metrics_zero_train_mse(self):
        """Test handling of zero train MSE."""
        train_mse = 0.0
        test_mse = 0.01

        metrics = compute_metrics(train_mse, test_mse)

        # Should handle division by zero
        assert metrics['generalization_ratio'] == float('inf')


class TestPrintMetrics:
    """Test metrics printing."""

    def test_print_metrics_runs(self):
        """Test that print_metrics runs without errors."""
        metrics = {
            'train_mse': 0.02,
            'test_mse': 0.025,
            'generalization_gap': 0.005,
            'generalization_ratio': 1.25,
            'generalizes_well': True
        }

        # Should not raise any errors
        print_metrics(metrics)


class TestSaveMetrics:
    """Test metrics saving."""

    def test_save_metrics_creates_file(self):
        """Test that save_metrics creates a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, 'metrics.json')

            metrics = {
                'train_mse': 0.02,
                'test_mse': 0.025,
                'generalization_gap': 0.005,
                'generalization_ratio': 1.25,
                'generalizes_well': True
            }

            save_metrics(metrics, metrics_path)

            assert os.path.exists(metrics_path)

    def test_save_metrics_correct_content(self):
        """Test that saved metrics have correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, 'metrics.json')

            metrics = {
                'train_mse': 0.02,
                'test_mse': 0.025,
                'generalization_gap': 0.005,
                'generalization_ratio': 1.25,
                'generalizes_well': True
            }

            save_metrics(metrics, metrics_path)

            # Load and verify
            with open(metrics_path, 'r') as f:
                loaded = json.load(f)

            assert loaded['train_mse'] == 0.02
            assert loaded['test_mse'] == 0.025
            assert loaded['generalizes_well'] == True

    def test_save_metrics_numpy_types(self):
        """Test saving metrics with numpy types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, 'metrics.json')

            # Use numpy types
            metrics = {
                'train_mse': np.float64(0.02),
                'test_mse': np.float32(0.025),
                'generalizes_well': np.bool_(True)
            }

            save_metrics(metrics, metrics_path)

            # Should be able to load as JSON
            with open(metrics_path, 'r') as f:
                loaded = json.load(f)

            assert isinstance(loaded['train_mse'], float)
            assert isinstance(loaded['generalizes_well'], bool)


class TestSavePredictions:
    """Test predictions saving."""

    def test_save_predictions_creates_file(self):
        """Test that save_predictions creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_path = os.path.join(tmpdir, 'predictions.npz')

            predictions = np.random.randn(100)
            targets = np.random.randn(100)
            S = np.random.randn(100)
            time = np.linspace(0, 10, 100)

            save_predictions(predictions, targets, S, time, predictions_path)

            assert os.path.exists(predictions_path)

    def test_save_predictions_correct_content(self):
        """Test that saved predictions have correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_path = os.path.join(tmpdir, 'predictions.npz')

            predictions = np.array([1, 2, 3, 4, 5])
            targets = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
            S = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
            time = np.array([0, 1, 2, 3, 4])

            save_predictions(predictions, targets, S, time, predictions_path)

            # Load and verify
            data = np.load(predictions_path)

            np.testing.assert_array_equal(data['predictions'], predictions)
            np.testing.assert_array_equal(data['targets'], targets)
            np.testing.assert_array_equal(data['S'], S)
            np.testing.assert_array_equal(data['time'], time)

    def test_save_predictions_large_arrays(self):
        """Test saving large prediction arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_path = os.path.join(tmpdir, 'predictions.npz')

            predictions = np.random.randn(40000)
            targets = np.random.randn(40000)
            S = np.random.randn(40000)
            time = np.linspace(0, 10, 40000)

            save_predictions(predictions, targets, S, time, predictions_path)

            # Load and verify shapes
            data = np.load(predictions_path)

            assert data['predictions'].shape == (40000,)
            assert data['targets'].shape == (40000,)


class TestIntegration:
    """Integration tests for evaluation workflow."""

    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy dataset
            S_train = np.random.randn(400)
            C_train = np.random.randint(0, 2, size=(400, 4)).astype(np.float32)
            targets_train = np.random.randn(400)

            S_test = np.random.randn(200)
            C_test = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
            targets_test = np.random.randn(200)

            # Create dataloaders
            train_loader = create_dataloader(S_train, C_train, targets_train, batch_size=32, shuffle=False)
            test_loader = create_dataloader(S_test, C_test, targets_test, batch_size=32, shuffle=False)

            # Create model
            model = FrequencyExtractorLSTM(hidden_size=16)

            # Evaluate on both sets
            train_mse, _, _ = evaluate_model(model, train_loader, device='cpu')
            test_mse, test_preds, test_targets = evaluate_model(model, test_loader, device='cpu')

            # Compute metrics
            metrics = compute_metrics(train_mse, test_mse)

            # Save metrics
            metrics_path = os.path.join(tmpdir, 'metrics.json')
            save_metrics(metrics, metrics_path)

            # Save predictions
            predictions_path = os.path.join(tmpdir, 'predictions.npz')
            time = np.linspace(0, 10, len(test_preds))
            save_predictions(test_preds, test_targets, S_test, time, predictions_path)

            # Verify files exist
            assert os.path.exists(metrics_path)
            assert os.path.exists(predictions_path)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
