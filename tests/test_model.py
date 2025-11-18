"""
Unit tests for model module.

Tests cover:
- LSTM model architecture
- Forward pass
- Hidden state management
- FrequencyDataset
- DataLoader creation
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import (
    FrequencyExtractorLSTM,
    FrequencyDataset,
    create_dataloader
)


class TestFrequencyExtractorLSTM:
    """Test LSTM model architecture."""

    def test_model_initialization(self):
        """Test model can be initialized with default parameters."""
        model = FrequencyExtractorLSTM()
        assert model.input_size == 5
        assert model.hidden_size == 64
        assert model.num_layers == 1

    def test_model_custom_parameters(self):
        """Test model with custom parameters."""
        model = FrequencyExtractorLSTM(
            input_size=10,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        assert model.input_size == 10
        assert model.hidden_size == 128
        assert model.num_layers == 2

    def test_forward_pass_shape(self):
        """Test forward pass output shapes."""
        model = FrequencyExtractorLSTM()
        batch_size = 32
        seq_len = 1
        input_size = 5

        x = torch.randn(batch_size, seq_len, input_size)
        output, hidden_state = model(x)

        assert output.shape == (batch_size, 1)
        assert hidden_state[0].shape == (1, batch_size, 64)  # h_n
        assert hidden_state[1].shape == (1, batch_size, 64)  # c_n

    def test_forward_pass_with_hidden(self):
        """Test forward pass with provided hidden state."""
        model = FrequencyExtractorLSTM()
        batch_size = 16
        x = torch.randn(batch_size, 1, 5)

        # First forward pass
        output1, hidden1 = model(x, hidden=None)

        # Second forward pass with previous hidden state
        output2, hidden2 = model(x, hidden=hidden1)

        assert output1.shape == output2.shape
        assert not torch.allclose(output1, output2)  # Different outputs

    def test_reset_state(self):
        """Test state reset functionality."""
        model = FrequencyExtractorLSTM(hidden_size=32)
        batch_size = 8

        h_0, c_0 = model.reset_state(batch_size)

        assert h_0.shape == (1, batch_size, 32)
        assert c_0.shape == (1, batch_size, 32)
        assert torch.all(h_0 == 0)
        assert torch.all(c_0 == 0)

    def test_get_num_parameters(self):
        """Test parameter counting."""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=64, num_layers=1)
        num_params = model.get_num_parameters()

        # Should have parameters > 0
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_multi_layer_lstm(self):
        """Test model with multiple LSTM layers."""
        model = FrequencyExtractorLSTM(num_layers=2, dropout=0.1)
        batch_size = 10
        x = torch.randn(batch_size, 1, 5)

        output, hidden = model(x)

        # Hidden state should have num_layers dimension
        assert hidden[0].shape == (2, batch_size, 64)
        assert hidden[1].shape == (2, batch_size, 64)

    def test_model_device_transfer(self):
        """Test model can be moved to different devices."""
        model = FrequencyExtractorLSTM()

        # CPU
        model_cpu = model.to('cpu')
        assert next(model_cpu.parameters()).device.type == 'cpu'

        # CUDA (if available)
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            assert next(model_cuda.parameters()).device.type == 'cuda'

    def test_model_eval_train_modes(self):
        """Test model can switch between eval and train modes."""
        model = FrequencyExtractorLSTM()

        model.train()
        assert model.training == True

        model.eval()
        assert model.training == False


class TestFrequencyDataset:
    """Test FrequencyDataset class."""

    def test_dataset_initialization(self):
        """Test dataset can be initialized."""
        S = np.random.randn(100)
        C = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
        targets = np.random.randn(100)

        dataset = FrequencyDataset(S, C, targets)

        assert len(dataset) == 100
        assert isinstance(dataset.S, torch.Tensor)
        assert isinstance(dataset.C, torch.Tensor)
        assert isinstance(dataset.targets, torch.Tensor)

    def test_dataset_getitem(self):
        """Test dataset indexing."""
        S = np.array([1.0, 2.0, 3.0])
        C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        targets = np.array([0.5, 0.6, 0.7])

        dataset = FrequencyDataset(S, C, targets)

        s, c, target = dataset[0]

        assert s.item() == 1.0
        assert torch.allclose(c, torch.tensor([1, 0, 0, 0], dtype=torch.float32))
        assert target.item() == 0.5

    def test_dataset_length(self):
        """Test dataset length."""
        S = np.random.randn(500)
        C = np.random.randint(0, 2, size=(500, 4)).astype(np.float32)
        targets = np.random.randn(500)

        dataset = FrequencyDataset(S, C, targets)

        assert len(dataset) == 500

    def test_dataset_shape_mismatch(self):
        """Test dataset raises error on shape mismatch."""
        S = np.random.randn(100)
        C = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
        targets = np.random.randn(90)  # Different length

        with pytest.raises(AssertionError):
            FrequencyDataset(S, C, targets)

    def test_dataset_tensor_types(self):
        """Test dataset converts to float tensors."""
        S = np.random.randn(50)
        C = np.random.randint(0, 2, size=(50, 4)).astype(np.float32)
        targets = np.random.randn(50)

        dataset = FrequencyDataset(S, C, targets)

        assert dataset.S.dtype == torch.float32
        assert dataset.C.dtype == torch.float32
        assert dataset.targets.dtype == torch.float32


class TestCreateDataloader:
    """Test dataloader creation."""

    def test_create_dataloader(self):
        """Test basic dataloader creation."""
        S = np.random.randn(1000)
        C = np.random.randint(0, 2, size=(1000, 4)).astype(np.float32)
        targets = np.random.randn(1000)

        dataloader = create_dataloader(S, C, targets, batch_size=64)

        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 64

    def test_dataloader_batch_iteration(self):
        """Test iterating through batches."""
        S = np.random.randn(200)
        C = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
        targets = np.random.randn(200)

        dataloader = create_dataloader(S, C, targets, batch_size=32, shuffle=False)

        batches = list(dataloader)
        assert len(batches) == 7  # 200 / 32 = 6.25, rounds up to 7

        # Check first batch
        batch_S, batch_C, batch_targets = batches[0]
        assert batch_S.shape == (32,)
        assert batch_C.shape == (32, 4)
        assert batch_targets.shape == (32,)

    def test_dataloader_shuffle(self):
        """Test dataloader shuffle functionality."""
        S = np.arange(100, dtype=np.float32)
        C = np.random.randint(0, 2, size=(100, 4)).astype(np.float32)
        targets = np.random.randn(100)

        # Without shuffle
        dataloader_no_shuffle = create_dataloader(S, C, targets, batch_size=10, shuffle=False)
        first_batch_no_shuffle, _, _ = next(iter(dataloader_no_shuffle))

        # With shuffle (check multiple times)
        shuffled = False
        for _ in range(5):
            dataloader_shuffle = create_dataloader(S, C, targets, batch_size=10, shuffle=True)
            first_batch_shuffle, _, _ = next(iter(dataloader_shuffle))
            if not torch.allclose(first_batch_shuffle, first_batch_no_shuffle):
                shuffled = True
                break

        # At least one shuffle should be different
        assert shuffled

    def test_dataloader_batch_sizes(self):
        """Test different batch sizes."""
        S = np.random.randn(1000)
        C = np.random.randint(0, 2, size=(1000, 4)).astype(np.float32)
        targets = np.random.randn(1000)

        for batch_size in [16, 32, 64, 128]:
            dataloader = create_dataloader(S, C, targets, batch_size=batch_size)
            assert dataloader.batch_size == batch_size


class TestModelIntegration:
    """Integration tests for model with dataloader."""

    def test_model_with_dataloader(self):
        """Test model forward pass with dataloader."""
        # Create dummy dataset
        S = np.random.randn(200)
        C = np.random.randint(0, 2, size=(200, 4)).astype(np.float32)
        targets = np.random.randn(200)

        dataloader = create_dataloader(S, C, targets, batch_size=32)

        model = FrequencyExtractorLSTM()

        # Process one batch
        for batch_S, batch_C, batch_targets in dataloader:
            # Prepare input
            inputs = torch.cat([batch_S.unsqueeze(-1), batch_C], dim=-1)
            inputs = inputs.unsqueeze(1)

            # Forward pass
            outputs, hidden = model(inputs, hidden=None)

            assert outputs.shape == (batch_S.shape[0], 1)
            assert hidden[0].shape[1] == batch_S.shape[0]
            break

    def test_full_epoch_simulation(self):
        """Test simulating a full epoch."""
        # Create dummy dataset
        S = np.random.randn(1000)
        C = np.random.randint(0, 2, size=(1000, 4)).astype(np.float32)
        targets = np.random.randn(1000)

        dataloader = create_dataloader(S, C, targets, batch_size=64)

        model = FrequencyExtractorLSTM()
        criterion = torch.nn.MSELoss()

        total_loss = 0
        num_batches = 0

        model.eval()
        with torch.no_grad():
            for batch_S, batch_C, batch_targets in dataloader:
                # Prepare input
                inputs = torch.cat([batch_S.unsqueeze(-1), batch_C], dim=-1)
                inputs = inputs.unsqueeze(1)

                # Forward pass with reset state
                outputs, _ = model(inputs, hidden=None)

                # Compute loss
                loss = criterion(outputs.squeeze(), batch_targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        assert avg_loss > 0  # Loss should be positive


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
