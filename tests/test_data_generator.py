"""
Unit tests for data_generator module.

Tests cover:
- Noisy sinusoid generation
- Mixed signal generation
- Ground truth generation
- Dataset creation
- File I/O operations
- Seed reproducibility
"""

import numpy as np
import pytest
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import (
    generate_noisy_sinusoid,
    generate_mixed_signal,
    generate_ground_truth,
    create_frequency_dataset,
    save_dataset,
    load_dataset,
    create_train_and_test_datasets,
    FREQUENCIES,
    NUM_SAMPLES,
    AMPLITUDE_MIN,
    AMPLITUDE_MAX,
    PHASE_MIN,
    PHASE_MAX,
    SAMPLING_RATE,
    TIME_START,
    TIME_END
)


class TestGenerateNoisySinusoid:
    """Test noisy sinusoid generation."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        time_array = np.linspace(0, 10, 100)
        signal = generate_noisy_sinusoid(1.0, time_array, seed=42)
        assert signal.shape == time_array.shape
        assert len(signal) == 100

    def test_amplitude_range(self):
        """Test that signal amplitudes are within expected range."""
        time_array = np.linspace(0, 10, 1000)
        frequency = 5.0
        signal = generate_noisy_sinusoid(frequency, time_array, seed=42)

        # Since amplitude is uniform(0.8, 1.2) and sin is [-1, 1],
        # signal should be roughly in range [-1.2, 1.2]
        assert np.min(signal) >= -1.5  # Allow some margin
        assert np.max(signal) <= 1.5

    def test_seed_reproducibility(self):
        """Test that same seed produces same output."""
        time_array = np.linspace(0, 10, 100)
        signal1 = generate_noisy_sinusoid(3.0, time_array, seed=123)
        signal2 = generate_noisy_sinusoid(3.0, time_array, seed=123)
        np.testing.assert_array_equal(signal1, signal2)

    def test_different_seeds_differ(self):
        """Test that different seeds produce different outputs."""
        time_array = np.linspace(0, 10, 100)
        signal1 = generate_noisy_sinusoid(3.0, time_array, seed=1)
        signal2 = generate_noisy_sinusoid(3.0, time_array, seed=2)
        assert not np.allclose(signal1, signal2)

    def test_different_frequencies(self):
        """Test signals with different frequencies are different."""
        time_array = np.linspace(0, 10, 100)
        signal1 = generate_noisy_sinusoid(1.0, time_array, seed=42)
        signal2 = generate_noisy_sinusoid(7.0, time_array, seed=42)
        # Even with same seed, different frequencies should produce different signals
        assert not np.allclose(signal1, signal2)


class TestGenerateMixedSignal:
    """Test mixed signal generation."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        time_array = np.linspace(0, 10, 100)
        mixed = generate_mixed_signal([1, 3, 5, 7], time_array, seed=42)
        assert mixed.shape == time_array.shape

    def test_averaging(self):
        """Test that signal is properly averaged."""
        time_array = np.linspace(0, 10, 100)
        mixed = generate_mixed_signal([1, 3, 5, 7], time_array, seed=42)

        # Mixed signal should be roughly in range [-1.2, 1.2] after averaging
        assert np.min(mixed) >= -2.0
        assert np.max(mixed) <= 2.0

    def test_seed_reproducibility(self):
        """Test that same seed produces same output."""
        time_array = np.linspace(0, 10, 100)
        mixed1 = generate_mixed_signal([1, 3, 5, 7], time_array, seed=42)
        mixed2 = generate_mixed_signal([1, 3, 5, 7], time_array, seed=42)
        np.testing.assert_array_equal(mixed1, mixed2)

    def test_different_seeds_differ(self):
        """Test that different seeds produce different outputs."""
        time_array = np.linspace(0, 10, 100)
        mixed1 = generate_mixed_signal([1, 3, 5, 7], time_array, seed=1)
        mixed2 = generate_mixed_signal([1, 3, 5, 7], time_array, seed=2)
        assert not np.allclose(mixed1, mixed2)


class TestGenerateGroundTruth:
    """Test ground truth generation."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        time_array = np.linspace(0, 10, 100)
        gt = generate_ground_truth(5.0, time_array)
        assert gt.shape == time_array.shape

    def test_amplitude_range(self):
        """Test that pure sinusoid is in [-1, 1]."""
        time_array = np.linspace(0, 10, 1000)
        gt = generate_ground_truth(3.0, time_array)
        assert np.min(gt) >= -1.0
        assert np.max(gt) <= 1.0
        # Should actually reach close to -1 and 1
        assert np.min(gt) < -0.99
        assert np.max(gt) > 0.99

    def test_deterministic(self):
        """Test that ground truth is deterministic (no randomness)."""
        time_array = np.linspace(0, 10, 100)
        gt1 = generate_ground_truth(7.0, time_array)
        gt2 = generate_ground_truth(7.0, time_array)
        np.testing.assert_array_equal(gt1, gt2)

    def test_correct_frequency(self):
        """Test that signal has correct frequency."""
        # For a 1 Hz signal over 1 second, should complete 1 cycle
        time_array = np.linspace(0, 1, 1000, endpoint=False)
        gt = generate_ground_truth(1.0, time_array)
        # At t=0, sin(0) = 0
        assert abs(gt[0]) < 0.01
        # At t=0.25, sin(π/2) = 1
        idx_quarter = int(0.25 * 1000)
        assert abs(gt[idx_quarter] - 1.0) < 0.01


class TestCreateFrequencyDataset:
    """Test complete dataset creation."""

    def test_output_shapes(self):
        """Test that outputs have correct shapes."""
        time_array = np.linspace(0, 10, 100)
        frequencies = [1, 3, 5, 7]
        S, C, targets = create_frequency_dataset(frequencies, time_array, seed=42)

        # 100 samples × 4 frequencies = 400 total rows
        assert S.shape == (400,)
        assert C.shape == (400, 4)
        assert targets.shape == (400,)

    def test_one_hot_encoding(self):
        """Test that one-hot encoding is correct."""
        time_array = np.linspace(0, 10, 100)
        frequencies = [1, 3, 5, 7]
        S, C, targets = create_frequency_dataset(frequencies, time_array, seed=42)

        # Each row should have exactly one 1 and three 0s
        assert np.all(np.sum(C, axis=1) == 1)

        # First 100 rows should have C[:, 0] = 1
        assert np.all(C[0:100, 0] == 1)
        assert np.all(C[0:100, 1:] == 0)

        # Second 100 rows should have C[:, 1] = 1
        assert np.all(C[100:200, 1] == 1)
        assert np.all(C[100:200, [0, 2, 3]] == 0)

    def test_signal_consistency(self):
        """Test that S is the same for all frequency selections."""
        time_array = np.linspace(0, 10, 100)
        frequencies = [1, 3, 5, 7]
        S, C, targets = create_frequency_dataset(frequencies, time_array, seed=42)

        # All rows should have the same mixed signal values (repeated 4 times)
        np.testing.assert_array_equal(S[0:100], S[100:200])
        np.testing.assert_array_equal(S[0:100], S[200:300])
        np.testing.assert_array_equal(S[0:100], S[300:400])

    def test_target_values(self):
        """Test that targets match ground truth for each frequency."""
        time_array = np.linspace(0, 10, 100)
        frequencies = [1, 3, 5, 7]
        S, C, targets = create_frequency_dataset(frequencies, time_array, seed=42)

        # Generate expected ground truths
        for i, freq in enumerate(frequencies):
            expected_gt = generate_ground_truth(freq, time_array)
            start_idx = i * 100
            end_idx = (i + 1) * 100
            np.testing.assert_array_almost_equal(
                targets[start_idx:end_idx],
                expected_gt
            )

    def test_full_size_dataset(self):
        """Test dataset with actual parameters (10000 samples, 4 frequencies)."""
        time_array = np.linspace(TIME_START, TIME_END, NUM_SAMPLES)
        S, C, targets = create_frequency_dataset(FREQUENCIES, time_array, seed=1)

        assert S.shape == (40000,)
        assert C.shape == (40000, 4)
        assert targets.shape == (40000,)


class TestDatasetIO:
    """Test dataset saving and loading."""

    def test_save_and_load(self):
        """Test that save and load preserve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create small dataset
            time_array = np.linspace(0, 10, 100)
            frequencies = [1, 3, 5, 7]
            S, C, targets = create_frequency_dataset(frequencies, time_array, seed=42)

            # Save
            filepath = os.path.join(tmpdir, 'test_dataset.npz')
            save_dataset(S, C, targets, time_array, seed=42, filename=filepath)

            # Load
            loaded = load_dataset(filepath)

            # Verify
            np.testing.assert_array_equal(loaded['S'], S)
            np.testing.assert_array_equal(loaded['C'], C)
            np.testing.assert_array_equal(loaded['targets'], targets)
            np.testing.assert_array_equal(loaded['time'], time_array)
            assert loaded['seed'] == 42

    def test_saved_metadata(self):
        """Test that metadata is correctly saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            time_array = np.linspace(0, 10, 100)
            frequencies = [1, 3, 5, 7]
            S, C, targets = create_frequency_dataset(frequencies, time_array, seed=123)

            filepath = os.path.join(tmpdir, 'test_dataset.npz')
            save_dataset(S, C, targets, time_array, seed=123, filename=filepath)

            loaded = load_dataset(filepath)

            np.testing.assert_array_equal(loaded['frequencies'], FREQUENCIES)
            assert loaded['sampling_rate'] == SAMPLING_RATE
            assert loaded['seed'] == 123


class TestCreateTrainAndTestDatasets:
    """Test creation of both train and test datasets."""

    def test_datasets_created(self):
        """Test that both datasets are created with correct shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_data, test_data = create_train_and_test_datasets(save_dir=tmpdir)

            # Check shapes
            assert train_data['S'].shape == (40000,)
            assert test_data['S'].shape == (40000,)
            assert train_data['C'].shape == (40000, 4)
            assert test_data['C'].shape == (40000, 4)

    def test_different_noise(self):
        """Test that train and test have different noise (different seeds)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_data, test_data = create_train_and_test_datasets(save_dir=tmpdir)

            # Signals should be different (different noise)
            assert not np.allclose(train_data['S'], test_data['S'])

            # But targets should be identical (same pure frequencies)
            np.testing.assert_array_almost_equal(
                train_data['targets'],
                test_data['targets']
            )

    def test_correct_seeds(self):
        """Test that train uses seed 1 and test uses seed 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_data, test_data = create_train_and_test_datasets(save_dir=tmpdir)

            assert train_data['seed'] == 1
            assert test_data['seed'] == 2

    def test_files_created(self):
        """Test that .npz files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_train_and_test_datasets(save_dir=tmpdir)

            assert os.path.exists(os.path.join(tmpdir, 'train_dataset.npz'))
            assert os.path.exists(os.path.join(tmpdir, 'test_dataset.npz'))


class TestConstants:
    """Test that constants have correct values."""

    def test_frequencies(self):
        """Test frequency values."""
        assert FREQUENCIES == [1, 3, 5, 7]

    def test_time_parameters(self):
        """Test time parameters."""
        assert TIME_START == 0.0
        assert TIME_END == 10.0
        assert NUM_SAMPLES == 10000
        assert SAMPLING_RATE == 1000

    def test_amplitude_range(self):
        """Test amplitude range."""
        assert AMPLITUDE_MIN == 0.8
        assert AMPLITUDE_MAX == 1.2
        assert AMPLITUDE_MIN < AMPLITUDE_MAX

    def test_phase_range(self):
        """Test phase range."""
        assert PHASE_MIN == 0.0
        assert PHASE_MAX == 2 * np.pi
        assert PHASE_MIN < PHASE_MAX


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
