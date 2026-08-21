"""Deep learning training helpers for CNN/RNN-style demos.

This module keeps a config-driven shape similar to the sklearn workflow,
while handling tensor datasets and PyTorch models for classroom demos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    classes: list[int]
    framing: str
    input_shape: tuple[int, ...]


class CNN2DClassifier(nn.Module):
    """Compact 2D CNN for small grayscale images (e.g., Digits 8x8)."""

    def __init__(
        self,
        num_classes: int,
        channels_1: int = 16,
        channels_2: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels_1, channels_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels_2 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class RecurrentClassifier(nn.Module):
    """Unified RNN/LSTM/GRU classifier head for sequence classification."""

    def __init__(
        self,
        cell_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if cell_type == "rnn":
            rnn_cls = nn.RNN
        elif cell_type == "lstm":
            rnn_cls = nn.LSTM
        elif cell_type == "gru":
            rnn_cls = nn.GRU
        else:
            raise ValueError(f"Unsupported recurrent cell_type: {cell_type}")

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.backbone = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.backbone(x)
        out = out[:, -1, :]
        return self.head(out)


def _generate_timeseries_dataset(n_samples: int, seq_len: int, n_classes: int = 4, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time series patterns naturally suited for RNN.
    
    Patterns:
    - Class 0: Linear increasing
    - Class 1: Linear decreasing  
    - Class 2: Sine wave
    - Class 3: Random walk
    """
    rng = np.random.RandomState(seed)
    x = np.zeros((n_samples, seq_len, 1), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    
    samples_per_class = n_samples // n_classes
    for cls in range(n_classes):
        start_idx = cls * samples_per_class
        end_idx = start_idx + samples_per_class
        
        if cls == 0:  # Linear increasing
            for i in range(end_idx - start_idx):
                x[start_idx + i, :, 0] = np.linspace(0, 1, seq_len) + rng.normal(0, 0.05, seq_len)
        elif cls == 1:  # Linear decreasing
            for i in range(end_idx - start_idx):
                x[start_idx + i, :, 0] = np.linspace(1, 0, seq_len) + rng.normal(0, 0.05, seq_len)
        elif cls == 2:  # Sine wave
            for i in range(end_idx - start_idx):
                t = np.linspace(0, 4 * np.pi, seq_len)
                x[start_idx + i, :, 0] = 0.5 * (1 + np.sin(t + rng.uniform(0, 2*np.pi))) + rng.normal(0, 0.05, seq_len)
        elif cls == 3:  # Random walk
            for i in range(end_idx - start_idx):
                x[start_idx + i, :, 0] = np.cumsum(rng.normal(0, 0.1, seq_len)) / seq_len + 0.5
        
        y[start_idx:end_idx] = cls
    
    # Shuffle
    shuffle_idx = rng.permutation(n_samples)
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    
    return x, y


def load_demo_dataset(data_cfg: dict, seed: int) -> DatasetBundle:
    """Load a famous image dataset or time series data and frame it for CNN or RNN-family models."""
    dataset_name = data_cfg.get("dataset_name", "digits")
    framing = data_cfg.get("framing", "image")
    test_size = float(data_cfg.get("test_size", 0.2))

    if dataset_name == "digits":
        digits = load_digits()
        x_img = digits.images.astype(np.float32) / 16.0  # normalize to [0,1]
        y = digits.target.astype(np.int64)

        if framing == "image":
            x = x_img  # (N, 8, 8)
            input_shape = (8, 8)
        elif framing == "sequence":
            # Sequential MNIST-style: each 8x8 image is a sequence of 8 timesteps
            # with 8 features each.
            x = x_img  # (N, seq_len=8, input_size=8)
            input_shape = (8, 8)
        else:
            raise ValueError("framing must be 'image' or 'sequence'.")
    
    elif dataset_name == "timeseries":
        # Generate synthetic temporal patterns naturally suited for RNN
        n_samples = int(data_cfg.get("n_samples", 800))
        seq_len = int(data_cfg.get("seq_len", 64))
        x, y = _generate_timeseries_dataset(n_samples=n_samples, seq_len=seq_len, seed=seed)
        input_shape = (seq_len, 1)
        framing = "sequence"
    
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Supported: 'digits', 'timeseries'")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return DatasetBundle(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        classes=sorted(np.unique(y).tolist()),
        framing=framing,
        input_shape=input_shape,
    )


def _make_model(model_type: str, model_params: dict, n_classes: int) -> nn.Module:
    if model_type == "cnn2d":
        return CNN2DClassifier(num_classes=n_classes, **model_params)

    if model_type in {"rnn", "lstm", "gru"}:
        return RecurrentClassifier(
            cell_type=model_type,
            input_size=int(model_params.get("input_size", 8)),
            hidden_size=int(model_params.get("hidden_size", 64)),
            num_layers=int(model_params.get("num_layers", 1)),
            num_classes=n_classes,
            dropout=float(model_params.get("dropout", 0.0)),
        )

    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        "Available: cnn2d, rnn, lstm, gru"
    )


def _make_loader(x: np.ndarray, y: np.ndarray, model_type: str, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    if model_type == "cnn2d":
        # (N, H, W) -> (N, 1, H, W)
        x_t = x_t.unsqueeze(1)

    dataset = TensorDataset(x_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_deep_model(
    dataset: DatasetBundle,
    model_type: str,
    model_params: dict,
    train_params: dict,
    seed: int,
) -> tuple[nn.Module, dict, np.ndarray, np.ndarray]:
    """Train deep model and return model, history, predictions, and probabilities."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(train_params.get("batch_size", 64))
    epochs = int(train_params.get("epochs", 20))
    learning_rate = float(train_params.get("learning_rate", 1e-3))
    weight_decay = float(train_params.get("weight_decay", 0.0))

    train_loader = _make_loader(dataset.x_train, dataset.y_train, model_type, batch_size, shuffle=True)
    test_loader = _make_loader(dataset.x_test, dataset.y_test, model_type, batch_size=256, shuffle=False)

    model = _make_model(model_type, model_params, n_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "test_accuracy": [],
        "device": str(device),
        "epochs": epochs,
    }

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        history["train_loss"].append(float(epoch_loss))
        history["test_accuracy"].append(float(correct / total if total else 0.0))

    model.eval()
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_pred.extend(np.argmax(probs, axis=1).tolist())

    return model, history, np.array(y_pred), np.array(y_prob)
