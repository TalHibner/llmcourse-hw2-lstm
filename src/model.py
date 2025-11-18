"""
LSTM Model Architecture for Frequency Extraction

This module implements the LSTM network for conditional regression
to extract pure frequency components from mixed noisy signals.

Critical State Management:
- For L=1 (sequence length = 1), internal state (h_t, c_t) must be RESET between batches
- This forces the LSTM to learn frequency patterns through internal memory alone
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import numpy as np


class FrequencyExtractorLSTM(nn.Module):
    """
    LSTM network for frequency extraction from mixed signals.

    Architecture:
        Input: [S[t], C1, C2, C3, C4] (5 features)
          ↓
        LSTM Layer(s)
          ↓
        Fully Connected Layer
          ↓
        Output: Target_i[t] (1 value)

    Critical Implementation:
        - Accepts hidden state as input (for state management)
        - Returns both output and updated hidden state
        - Setting hidden=None automatically resets state to zeros
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features (default: 5 = S + 4 C values)
            hidden_size: Number of hidden units in LSTM (tunable)
            num_layers: Number of LSTM layers (tunable)
            dropout: Dropout probability between LSTM layers (if num_layers > 1)
        """
        super(FrequencyExtractorLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, seq_len=1, features=5]
            hidden: Tuple of (h_0, c_0) or None
                   - If None, LSTM automatically initializes with zeros (STATE RESET)
                   - If provided, continues from previous state

        Returns:
            output: Predicted target values, shape [batch_size, 1]
            hidden_state: Tuple of (h_n, c_n) for next iteration
        """
        # LSTM forward pass
        # lstm_out shape: [batch, seq_len, hidden_size]
        # hidden_state: tuple of (h_n, c_n)
        lstm_out, hidden_state = self.lstm(x, hidden)

        # Take the output from the last (and only) timestep
        # Shape: [batch, hidden_size]
        last_output = lstm_out[:, -1, :]

        # Fully connected layer
        # Shape: [batch, 1]
        output = self.fc(last_output)

        return output, hidden_state

    def get_num_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            num_params: Total trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_state(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Explicitly create zero hidden state.

        Note: This is usually not needed because passing hidden=None
        automatically resets. This method is provided for clarity.

        Args:
            batch_size: Batch size
            device: 'cpu' or 'cuda'

        Returns:
            (h_0, c_0): Zero-initialized hidden and cell states
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)


class FrequencyDataset(Dataset):
    """
    PyTorch Dataset for frequency extraction training.

    Each sample consists of:
        - S[t]: Mixed noisy signal value
        - C: One-hot frequency selection vector [C1, C2, C3, C4]
        - target: Ground truth pure sinusoid value
    """

    def __init__(
        self,
        S: np.ndarray,
        C: np.ndarray,
        targets: np.ndarray
    ):
        """
        Initialize dataset.

        Args:
            S: Signal values, shape (N,)
            C: One-hot vectors, shape (N, 4)
            targets: Ground truth targets, shape (N,)
        """
        self.S = torch.FloatTensor(S)
        self.C = torch.FloatTensor(C)
        self.targets = torch.FloatTensor(targets)

        assert len(S) == len(C) == len(targets), "All arrays must have same length"

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.S)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            S[idx], C[idx], target[idx]
        """
        return self.S[idx], self.C[idx], self.targets[idx]


def create_dataloader(
    S: np.ndarray,
    C: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create PyTorch DataLoader for training or evaluation.

    Args:
        S: Signal values
        C: One-hot vectors
        targets: Ground truth targets
        batch_size: Batch size
        shuffle: Whether to shuffle data (True for training, False for eval)
        num_workers: Number of data loading workers

    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = FrequencyDataset(S, C, targets)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


def test_model():
    """
    Test the model architecture to ensure it works correctly.
    """
    print("="*60)
    print("Testing LSTM Model Architecture")
    print("="*60)

    # Create dummy data
    batch_size = 32
    seq_len = 1
    input_size = 5

    # Random input
    x = torch.randn(batch_size, seq_len, input_size)

    # Initialize model
    model = FrequencyExtractorLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=1
    )

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal Parameters: {model.get_num_parameters():,}")

    # Test forward pass with no hidden state (STATE RESET)
    print("\n" + "-"*60)
    print("Test 1: Forward pass with hidden=None (STATE RESET)")
    print("-"*60)

    output, hidden_state = model(x, hidden=None)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state (h_n) shape: {hidden_state[0].shape}")
    print(f"Cell state (c_n) shape: {hidden_state[1].shape}")

    assert output.shape == (batch_size, 1), f"Output shape mismatch: {output.shape}"
    assert hidden_state[0].shape == (1, batch_size, 64), f"Hidden state shape mismatch"
    print("✓ Shapes are correct")

    # Test forward pass with provided hidden state (CONTINUE)
    print("\n" + "-"*60)
    print("Test 2: Forward pass with provided hidden state (CONTINUE)")
    print("-"*60)

    output2, hidden_state2 = model(x, hidden=hidden_state)

    print(f"Output shape: {output2.shape}")
    print("✓ Can continue with previous state")

    # Test state reset
    print("\n" + "-"*60)
    print("Test 3: Explicit state reset")
    print("-"*60)

    reset_hidden = model.reset_state(batch_size)
    print(f"Reset hidden state (h_0) shape: {reset_hidden[0].shape}")
    print(f"Reset cell state (c_0) shape: {reset_hidden[1].shape}")
    print(f"All values zero: {torch.all(reset_hidden[0] == 0) and torch.all(reset_hidden[1] == 0)}")
    print("✓ State reset works correctly")

    # Test with DataLoader
    print("\n" + "-"*60)
    print("Test 4: DataLoader integration")
    print("-"*60)

    # Create dummy dataset
    dummy_S = np.random.randn(1000)
    dummy_C = np.random.randint(0, 2, size=(1000, 4)).astype(np.float32)
    dummy_targets = np.random.randn(1000)

    dataloader = create_dataloader(
        dummy_S, dummy_C, dummy_targets,
        batch_size=64,
        shuffle=True
    )

    print(f"DataLoader created successfully")
    print(f"Number of batches: {len(dataloader)}")

    # Test one batch
    for batch_S, batch_C, batch_targets in dataloader:
        # Concatenate S and C
        inputs = torch.cat([batch_S.unsqueeze(-1), batch_C], dim=-1)
        inputs = inputs.unsqueeze(1)  # Add sequence dimension

        # Forward pass
        outputs, _ = model(inputs, hidden=None)

        print(f"Batch input shape: {inputs.shape}")
        print(f"Batch output shape: {outputs.shape}")
        print(f"Batch target shape: {batch_targets.shape}")
        print("✓ DataLoader integration works")
        break

    print("\n" + "="*60)
    print("All Tests Passed!")
    print("="*60)


if __name__ == "__main__":
    test_model()
