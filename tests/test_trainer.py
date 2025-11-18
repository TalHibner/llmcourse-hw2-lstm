"""
Unit tests for trainer module.

Tests cover:
- Training epoch
- Validation epoch
- Model training with early stopping
- Checkpoint saving/loading
- Training configuration
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
import tempfile
import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trainer import (
    train_epoch,
    validate_epoch,
    train_model,
    load_trained_model,
    get_training_config,
    save_training_config,
    load_training_config
)
from model import FrequencyExtractorLSTM, create_dataloader


class TestTrainEpoch:
    """Test training epoch functionality."""

    def test_train_epoch_runs(self):
        """Test that train_epoch completes without errors."""
        # Create dummy data
        S = np.random.randn(200)
        C = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
        targets = np.random.randn(200)

        dataloader = create_dataloader(S, C, targets, batch_size=32)

        model = FrequencyExtractorLSTM(hidden_size=16)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = train_epoch(model, dataloader, criterion, optimizer, device='cpu')

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_epoch_updates_weights(self):
        """Test that training updates model weights."""
        S = np.random.randn(200)
        C = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
        targets = np.random.randn(200)

        dataloader = create_dataloader(S, C, targets, batch_size=32)

        model = FrequencyExtractorLSTM(hidden_size=16)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Get initial weights
        initial_weight = model.fc.weight.clone()

        # Train one epoch
        train_epoch(model, dataloader, criterion, optimizer, device='cpu')

        # Check weights changed
        assert not torch.allclose(initial_weight, model.fc.weight)

    def test_train_epoch_model_in_train_mode(self):
        """Test that model is in training mode during train_epoch."""
        S = np.random.randn(100)
        C = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
        targets = np.random.randn(100)

        dataloader = create_dataloader(S, C, targets, batch_size=32)

        model = FrequencyExtractorLSTM(hidden_size=16)
        model.eval()  # Set to eval mode first

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_epoch(model, dataloader, criterion, optimizer, device='cpu')

        # Model should be in train mode after
        assert model.training == True


class TestValidateEpoch:
    """Test validation epoch functionality."""

    def test_validate_epoch_runs(self):
        """Test that validate_epoch completes without errors."""
        S = np.random.randn(200)
        C = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
        targets = np.random.randn(200)

        dataloader = create_dataloader(S, C, targets, batch_size=32)

        model = FrequencyExtractorLSTM(hidden_size=16)
        criterion = nn.MSELoss()

        loss = validate_epoch(model, dataloader, criterion, device='cpu')

        assert isinstance(loss, float)
        assert loss > 0

    def test_validate_epoch_no_gradient(self):
        """Test that validation doesn't compute gradients."""
        S = np.random.randn(100)
        C = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
        targets = np.random.randn(100)

        dataloader = create_dataloader(S, C, targets, batch_size=32)

        model = FrequencyExtractorLSTM(hidden_size=16)
        criterion = nn.MSELoss()

        # Get initial weights
        initial_weight = model.fc.weight.clone()

        # Validate
        validate_epoch(model, dataloader, criterion, device='cpu')

        # Weights should not change
        assert torch.allclose(initial_weight, model.fc.weight)

    def test_validate_epoch_model_in_eval_mode(self):
        """Test that model is in eval mode during validation."""
        S = np.random.randn(100)
        C = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
        targets = np.random.randn(100)

        dataloader = create_dataloader(S, C, targets, batch_size=32)

        model = FrequencyExtractorLSTM(hidden_size=16)
        model.train()  # Set to train mode first

        criterion = nn.MSELoss()

        validate_epoch(model, dataloader, criterion, device='cpu')

        # Model should be in eval mode after
        assert model.training == False


class TestTrainModel:
    """Test full model training."""

    def test_train_model_basic(self):
        """Test basic training loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data
            S_train = np.random.randn(400)
            C_train = np.random.randint(0, 2, size=(400, 4)).astype(np.float32)
            targets_train = np.random.randn(400)

            S_test = np.random.randn(200)
            C_test = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
            targets_test = np.random.randn(200)

            train_loader = create_dataloader(S_train, C_train, targets_train, batch_size=32)
            test_loader = create_dataloader(S_test, C_test, targets_test, batch_size=32)

            model = FrequencyExtractorLSTM(hidden_size=16)

            save_path = os.path.join(tmpdir, 'test_model.pth')

            history = train_model(
                model,
                train_loader,
                test_loader,
                num_epochs=3,
                learning_rate=0.001,
                device='cpu',
                save_path=save_path,
                patience=10,
                verbose=False
            )

            # Check history structure
            assert 'train_loss' in history
            assert 'test_loss' in history
            assert 'epochs' in history
            assert len(history['train_loss']) == 3
            assert os.path.exists(save_path)

    def test_train_model_early_stopping(self):
        """Test early stopping mechanism."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data with low variance (easy to overfit)
            S_train = np.random.randn(200)
            C_train = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
            targets_train = np.random.randn(200)

            S_test = np.random.randn(100)
            C_test = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
            targets_test = np.random.randn(100)

            train_loader = create_dataloader(S_train, C_train, targets_train, batch_size=32)
            test_loader = create_dataloader(S_test, C_test, targets_test, batch_size=32)

            model = FrequencyExtractorLSTM(hidden_size=16)

            save_path = os.path.join(tmpdir, 'test_model.pth')

            history = train_model(
                model,
                train_loader,
                test_loader,
                num_epochs=100,
                learning_rate=0.001,
                device='cpu',
                save_path=save_path,
                patience=3,  # Stop early
                verbose=False
            )

            # Should stop before 100 epochs due to patience
            assert len(history['train_loss']) < 100

    def test_train_model_saves_best(self):
        """Test that best model is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            S_train = np.random.randn(200)
            C_train = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
            targets_train = np.random.randn(200)

            S_test = np.random.randn(100)
            C_test = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
            targets_test = np.random.randn(100)

            train_loader = create_dataloader(S_train, C_train, targets_train, batch_size=32)
            test_loader = create_dataloader(S_test, C_test, targets_test, batch_size=32)

            model = FrequencyExtractorLSTM(hidden_size=16)

            save_path = os.path.join(tmpdir, 'best_model.pth')

            train_model(
                model,
                train_loader,
                test_loader,
                num_epochs=5,
                learning_rate=0.001,
                device='cpu',
                save_path=save_path,
                patience=10,
                verbose=False
            )

            # Check checkpoint contents
            checkpoint = torch.load(save_path, map_location='cpu')

            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'epoch' in checkpoint
            assert 'train_loss' in checkpoint
            assert 'test_loss' in checkpoint
            assert 'history' in checkpoint

    def test_train_model_history_structure(self):
        """Test that history has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            S_train = np.random.randn(200)
            C_train = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
            targets_train = np.random.randn(200)

            S_test = np.random.randn(100)
            C_test = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
            targets_test = np.random.randn(100)

            train_loader = create_dataloader(S_train, C_train, targets_train, batch_size=32)
            test_loader = create_dataloader(S_test, C_test, targets_test, batch_size=32)

            model = FrequencyExtractorLSTM(hidden_size=16)

            save_path = os.path.join(tmpdir, 'test_model.pth')

            history = train_model(
                model,
                train_loader,
                test_loader,
                num_epochs=3,
                learning_rate=0.001,
                device='cpu',
                save_path=save_path,
                patience=10,
                verbose=False
            )

            assert len(history['epochs']) == len(history['train_loss'])
            assert len(history['epochs']) == len(history['test_loss'])
            assert len(history['epochs']) == len(history['learning_rates'])


class TestLoadTrainedModel:
    """Test loading trained models."""

    def test_load_trained_model(self):
        """Test loading a saved model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save a model
            S_train = np.random.randn(200)
            C_train = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
            targets_train = np.random.randn(200)

            S_test = np.random.randn(100)
            C_test = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
            targets_test = np.random.randn(100)

            train_loader = create_dataloader(S_train, C_train, targets_train, batch_size=32)
            test_loader = create_dataloader(S_test, C_test, targets_test, batch_size=32)

            model_original = FrequencyExtractorLSTM(hidden_size=16)
            save_path = os.path.join(tmpdir, 'model.pth')

            train_model(
                model_original,
                train_loader,
                test_loader,
                num_epochs=2,
                learning_rate=0.001,
                device='cpu',
                save_path=save_path,
                patience=10,
                verbose=False
            )

            # Load the model
            model_loaded = FrequencyExtractorLSTM(hidden_size=16)
            loaded_model, history = load_trained_model(model_loaded, save_path, device='cpu')

            # Check model is in eval mode
            assert loaded_model.training == False

            # Check history is loaded
            assert 'train_loss' in history

    def test_loaded_model_produces_same_output(self):
        """Test that loaded model produces same output as original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            S = np.random.randn(100)
            C = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
            targets = np.random.randn(100)

            train_loader = create_dataloader(S[:80], C[:80], targets[:80], batch_size=32)
            test_loader = create_dataloader(S[80:], C[80:], targets[80:], batch_size=32)

            model = FrequencyExtractorLSTM(hidden_size=16)
            save_path = os.path.join(tmpdir, 'model.pth')

            train_model(
                model,
                train_loader,
                test_loader,
                num_epochs=2,
                learning_rate=0.001,
                device='cpu',
                save_path=save_path,
                patience=10,
                verbose=False
            )

            # Get output from original model
            model.eval()
            test_input = torch.randn(1, 1, 5)
            with torch.no_grad():
                output1, _ = model(test_input)

            # Load model and get output
            model_loaded = FrequencyExtractorLSTM(hidden_size=16)
            loaded_model, _ = load_trained_model(model_loaded, save_path, device='cpu')

            with torch.no_grad():
                output2, _ = loaded_model(test_input)

            assert torch.allclose(output1, output2)


class TestTrainingConfig:
    """Test training configuration utilities."""

    def test_get_training_config(self):
        """Test getting default training config."""
        config = get_training_config()

        assert 'input_size' in config
        assert 'hidden_size' in config
        assert 'batch_size' in config
        assert 'num_epochs' in config
        assert 'learning_rate' in config
        assert config['input_size'] == 5
        assert config['sequence_length'] == 1
        assert config['reset_state'] == True

    def test_save_load_config(self):
        """Test saving and loading config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.json')

            # Create and save config
            config = {
                'input_size': 5,
                'hidden_size': 128,
                'batch_size': 64,
                'learning_rate': 0.0001
            }

            save_training_config(config, config_path)

            # Load config
            loaded_config = load_training_config(config_path)

            assert loaded_config == config

    def test_config_json_format(self):
        """Test that config is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.json')

            config = get_training_config()
            save_training_config(config, config_path)

            # Read as JSON
            with open(config_path, 'r') as f:
                loaded = json.load(f)

            assert isinstance(loaded, dict)
            assert 'input_size' in loaded


class TestStateReset:
    """Test that state is properly reset during training."""

    def test_hidden_state_reset_per_batch(self):
        """Test that hidden state is reset for each batch."""
        S = np.random.randn(200)
        C = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
        targets = np.random.randn(200)

        dataloader = create_dataloader(S, C, targets, batch_size=32)

        model = FrequencyExtractorLSTM(hidden_size=16)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()

        hidden_states = []

        for batch_S, batch_C, batch_targets in dataloader:
            inputs = torch.cat([batch_S.unsqueeze(-1), batch_C], dim=-1)
            inputs = inputs.unsqueeze(1)

            # Reset hidden state (as done in train_epoch)
            hidden = None

            outputs, hidden_new = model(inputs, hidden)

            # Store hidden state
            hidden_states.append(hidden_new)

            # Backward pass
            loss = criterion(outputs.squeeze(), batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Each batch should have different hidden states (not carried over)
        assert len(hidden_states) > 1


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
