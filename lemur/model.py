from __future__ import annotations

import math

import torch
import torch.nn as nn


def _validate_activation(activation: str) -> str:
    allowed = {"relu", "gelu", "silu", "mish"}
    if activation not in allowed:
        raise ValueError(f"activation must be one of: {', '.join(allowed)}")
    return activation


def _activation_cls(activation: str) -> type[nn.Module]:
    classes = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
    }
    return classes[activation]


def _activation_fn(activation: str):
    functions = {
        "relu": torch.relu,
        "gelu": torch.nn.functional.gelu,
        "silu": torch.nn.functional.silu,
        "mish": torch.nn.functional.mish,
    }
    return functions[activation]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=1024,
        final_hidden_dim=None,
        num_layers=2,
        activation="relu",
    ):
        super().__init__()

        if final_hidden_dim is None:
            final_hidden_dim = hidden_dim

        self.config = {
            "model_type": "mlp",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "final_hidden_dim": final_hidden_dim,
            "num_layers": num_layers,
            "activation": activation,
        }

        activation = _validate_activation(activation)
        activation_cls = _activation_cls(activation)

        modules = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for i, in_dim in enumerate(dims):
            is_last = i == len(dims) - 1
            out_dim = final_hidden_dim if is_last else hidden_dim

            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.LayerNorm(out_dim))
            modules.append(activation_cls())

        self.feature_extractor = nn.Sequential(*modules)

        self.output_layer = nn.Linear(final_hidden_dim, output_dim, bias=False)

    def forward(self, x):
        feats = self.feature_extractor(x)
        return self.output_layer(feats)


class RandomActivationFeatures(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str = "gelu"):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0 for random activation features")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0 for random activation features")

        activation = _validate_activation(activation)
        self.register_buffer("weight", torch.randn(input_dim, output_dim))
        self.activation = activation
        self._activation_fn = _activation_fn(activation)
        self.scale = math.sqrt(2.0 / output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self._activation_fn(x @ self.weight)


class ELM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        final_hidden_dim: int,
        activation: str = "gelu",
    ):
        super().__init__()
        if final_hidden_dim <= 0:
            raise ValueError("final_hidden_dim must be > 0")
        activation = _validate_activation(activation)
        self.config = {
            "model_type": "elm",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "final_hidden_dim": final_hidden_dim,
            "activation": activation,
        }
        self.feature_extractor = RandomActivationFeatures(
            input_dim=input_dim,
            output_dim=final_hidden_dim,
            activation=activation,
        )
        self.output_layer = nn.Linear(final_hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return self.output_layer(feats)
