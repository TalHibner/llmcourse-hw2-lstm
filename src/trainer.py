"""
Training Module for LSTM Frequency Extraction

This module handles training the LSTM model with CRITICAL state management.

State Reset Strategy (L=1):
    For each batch, hidden state MUST be reset to zeros:
        hidden = None  # PyTorch automatically initializes to zeros

This is pedagogically important to demonstrate that the LSTM can learn
frequency patterns through internal memory management alone, without
relying on sequential dependencies between samples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import numpy as np
import json
import time


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = 'cpu'
) -> float:
    """
    Train model for one epoch.

    CRITICAL: Hidden state is reset for each batch (hidden=None)

    Args:
        model: LSTM model
        dataloader: Training data loader
        criterion: Loss function (MSE)
        optimizer: Optimizer (Adam)
        device: 'cpu' or 'cuda'

    Returns:
        avg_loss: Average loss over all batches
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (S, C, targets) in enumerate(dataloader):
        # Move data to device
        S = S.to(device)
        C = C.to(device)
        targets = targets.to(device)

        # Prepare input: concatenate S and C
        # S shape: [batch]
        # C shape: [batch, 4]
        # inputs shape: [batch, 5]
        inputs = torch.cat([S.unsqueeze(-1), C], dim=-1)

        # Add sequence dimension: [batch, seq_len=1, features=5]
        inputs = inputs.unsqueeze(1)

        # ✅ CRITICAL: Reset hidden state for each batch
        # This ensures L=1 behavior - no temporal dependencies between batches
        hidden = None

        # Forward pass
        outputs, hidden = model(inputs, hidden)

        # Compute loss
        # outputs shape: [batch, 1]
        # targets shape: [batch]
        loss = criterion(outputs.squeeze(), targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional, helps with stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = 'cpu'
) -> float:
    """
    Validate model on validation/test set.

    CRITICAL: Hidden state is reset for each batch (hidden=None)

    Args:
        model: LSTM model
        dataloader: Validation data loader
        criterion: Loss function (MSE)
        device: 'cpu' or 'cuda'

    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

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

            # Compute loss
            loss = criterion(outputs.squeeze(), targets)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    save_path: str = 'models/best_model.pth',
    patience: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Train LSTM model with early stopping.

    Args:
        model: LSTM model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for Adam optimizer
        device: 'cpu' or 'cuda'
        save_path: Path to save best model
        patience: Early stopping patience
        verbose: Print training progress

    Returns:
        history: Dictionary containing training history
    """
    # Move model to device
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'learning_rates': [],
        'epochs': []
    }

    best_test_loss = float('inf')
    epochs_without_improvement = 0

    if verbose:
        print("="*70)
        print("Starting Training")
        print("="*70)
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print("="*70)
        print()

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        test_loss = validate_epoch(model, test_loader, criterion, device)

        # Update scheduler
        scheduler.step(test_loss)

        # Save history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['learning_rates'].append(current_lr)
        history['epochs'].append(epoch + 1)

        epoch_time = time.time() - epoch_start_time

        # Print progress
        if verbose:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Test Loss: {test_loss:.6f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.2f}s")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'history': history
            }, save_path)
            if verbose:
                print(f"  → Best model saved! (Test Loss: {best_test_loss:.6f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best test loss: {best_test_loss:.6f}")
            break

    total_time = time.time() - start_time

    # Add computational metrics to history
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history['computational_metrics'] = {
        'total_training_time_seconds': round(total_time, 2),
        'total_training_time_minutes': round(total_time / 60, 2),
        'average_time_per_epoch_seconds': round(total_time / len(history['epochs']), 2),
        'device_used': str(device),
        'num_model_parameters': num_params,
        'epochs_completed': len(history['epochs']),
        'early_stopped': len(history['epochs']) < num_epochs,
        'batch_size': train_loader.batch_size,
        'learning_rate_initial': learning_rate,
        'learning_rate_final': history['learning_rates'][-1] if history['learning_rates'] else learning_rate
    }

    if verbose:
        print()
        print("="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Average time per epoch: {history['computational_metrics']['average_time_per_epoch_seconds']:.2f}s")
        print(f"Best test loss: {best_test_loss:.6f}")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"Model saved to: {save_path}")
        print("="*70)

    return history


def load_trained_model(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cpu'
) -> tuple:
    """
    Load trained model from checkpoint.

    Args:
        model: Model instance (architecture)
        checkpoint_path: Path to checkpoint file
        device: 'cpu' or 'cuda'

    Returns:
        model: Loaded model
        history: Training history
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Train Loss: {checkpoint['train_loss']:.6f}")
    print(f"  - Test Loss: {checkpoint['test_loss']:.6f}")

    return model, checkpoint.get('history', {})


def get_training_config() -> Dict:
    """
    Get default training configuration.

    Returns:
        config: Configuration dictionary
    """
    config = {
        # Model
        'input_size': 5,
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.0,

        # Training
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,

        # State Management
        'sequence_length': 1,
        'reset_state': True,  # Critical for L=1

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    return config


def save_training_config(config: Dict, filename: str = 'models/config.json'):
    """Save training configuration to JSON file."""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filename}")


def load_training_config(filename: str = 'models/config.json') -> Dict:
    """Load training configuration from JSON file."""
    with open(filename, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {filename}")
    return config


if __name__ == "__main__":
    """
    Test the training module with dummy data
    """
    from src.model import FrequencyExtractorLSTM, create_dataloader
    import os

    print("="*70)
    print("Testing Training Module")
    print("="*70)

    # Create dummy data
    np.random.seed(42)
    dummy_S = np.random.randn(1000)
    dummy_C = np.random.randint(0, 2, size=(1000, 4)).astype(np.float32)
    dummy_targets = np.random.randn(1000)

    # Create dataloaders
    train_loader = create_dataloader(dummy_S[:800], dummy_C[:800], dummy_targets[:800],
                                     batch_size=32, shuffle=True)
    test_loader = create_dataloader(dummy_S[800:], dummy_C[800:], dummy_targets[800:],
                                    batch_size=32, shuffle=False)

    # Create model
    model = FrequencyExtractorLSTM(input_size=5, hidden_size=32, num_layers=1)

    # Get config
    config = get_training_config()
    config['num_epochs'] = 5  # Just for testing
    config['hidden_size'] = 32
    print(f"\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Train
    history = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device'],
        save_path='models/test_model.pth',
        patience=5,
        verbose=True
    )

    print("\n" + "="*70)
    print("Training Test Complete!")
    print("="*70)
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final test loss: {history['test_loss'][-1]:.6f}")
