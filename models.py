from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModelRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., nn.Module]] = {}

    def register(self, name: str):
        key = name.lower()

        def decorator(fn: Callable[..., nn.Module]):
            if key in self._builders:
                raise KeyError(f"Model already registered: {name}")
            self._builders[key] = fn
            return fn

        return decorator

    def build(self, name: str, **kwargs) -> nn.Module:
        key = name.lower()
        if key not in self._builders:
            raise KeyError(f"Unknown model {name!r}. Available: {sorted(self._builders)}")
        return self._builders[key](**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))


MODELS = ModelRegistry()


# -----------------------------------------------------------------------------
# CIFAR-10: Modified ResNet-18 with 2-group GroupNorm
# -----------------------------------------------------------------------------


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(2, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(2, out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CIFARResNet18GN(nn.Module):
    """ResNet-18 adapted for 32x32 CIFAR images using two-group GroupNorm."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(2, 64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [CIFARBasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(CIFARBasicBlock(self.in_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


@MODELS.register("cifar10")
def build_cifar10_model(num_classes: int = 10, **_: object) -> nn.Module:
    return CIFARResNet18GN(num_classes=num_classes)


# -----------------------------------------------------------------------------
# FEMNIST: two-convolution benchmark CNN, ~1.20M parameters
# -----------------------------------------------------------------------------


class FEMNISTCNN(nn.Module):
    """Two-convolution FEMNIST CNN with a 2048-unit hidden layer."""

    def __init__(self, num_classes: int = 62) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        return self.fc2(x)


@MODELS.register("femnist")
def build_femnist_model(num_classes: int = 62, **_: object) -> nn.Module:
    return FEMNISTCNN(num_classes=num_classes)


# -----------------------------------------------------------------------------
# Shakespeare: character embedding -> 2-layer LSTM -> character projection
# Target ~2M parameters.
# -----------------------------------------------------------------------------


class ShakespeareLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 86,
        embedding_dim: int = 96,
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(tokens)
        x, state = self.lstm(x, state)
        logits = self.projection(x)
        return logits, state


@MODELS.register("shakespeare")
def build_shakespeare_model(
    vocab_size: int = 86,
    embedding_dim: int = 96,
    hidden_size: int = 384,
    num_layers: int = 2,
    **_: object,
) -> nn.Module:
    return ShakespeareLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )


# -----------------------------------------------------------------------------
# N-BaIoT: four-block residual tabular MLP, ~595k parameters
# -----------------------------------------------------------------------------


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.norm2(self.fc2(x))
        return F.gelu(x + residual)


class NBAIoTResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 115,
        width: int = 264,
        blocks: int = 4,
        num_classes: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ResidualMLPBlock(width, dropout=dropout) for _ in range(blocks))
        )
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.blocks(x)
        return self.head(x)


@MODELS.register("nbaiot")
def build_nbaiot_model(
    input_dim: int = 115,
    width: int = 264,
    blocks: int = 4,
    num_classes: int = 2,
    **_: object,
) -> nn.Module:
    return NBAIoTResidualMLP(
        input_dim=input_dim,
        width=width,
        blocks=blocks,
        num_classes=num_classes,
    )


# -----------------------------------------------------------------------------
# Convenience API
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSummary:
    name: str
    parameters: int


def build_model(name: str, **kwargs) -> nn.Module:
    return MODELS.build(name, **kwargs)


def summarize_default_models() -> list[ModelSummary]:
    configs = {
        "cifar10": {},
        "femnist": {},
        "shakespeare": {},
        "nbaiot": {},
    }
    summaries: list[ModelSummary] = []
    for name, kwargs in configs.items():
        model = build_model(name, **kwargs)
        summaries.append(ModelSummary(name=name, parameters=count_trainable_parameters(model)))
    return summaries


if __name__ == "__main__":
    for summary in summarize_default_models():
        print(f"{summary.name:12s} {summary.parameters:,} trainable parameters")
