from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features).to(self.weight_mu.device)
        epsilon_out = self._scale_noise(self.out_features).to(self.weight_mu.device)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(inputs, weight, bias)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()


class RainbowNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        noisy_std: float,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim, noisy_std)
        self.value_output = NoisyLinear(hidden_dim, num_atoms, noisy_std)

        self.advantage_hidden = NoisyLinear(hidden_dim, hidden_dim, noisy_std)
        self.advantage_output = NoisyLinear(hidden_dim, action_dim * num_atoms, noisy_std)

        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        distribution = self.dist(inputs)
        return torch.sum(distribution * self.support, dim=-1)

    def dist(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(inputs)

        value = F.relu(self.value_hidden(features))
        value = self.value_output(value).view(-1, 1, self.num_atoms)

        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_output(advantage).view(-1, self.action_dim, self.num_atoms)

        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        distribution = F.softmax(logits, dim=-1)
        distribution = distribution.clamp(min=1e-6)
        return distribution / distribution.sum(dim=-1, keepdim=True)

    def reset_noise(self) -> None:
        self.value_hidden.reset_noise()
        self.value_output.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_output.reset_noise()
