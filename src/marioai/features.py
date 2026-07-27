"""Stable-Baselines3 visual feature extractors for Mario observations."""

import torch
from torch import nn
from stable_baselines3.common.preprocessing import (
    is_image_space_channels_first,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImpalaResidualBlock(nn.Module):
    """Pre-activation residual block used by the IMPALA visual encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class ImpalaCnnFeaturesExtractor(BaseFeaturesExtractor):
    """IMPALA residual CNN for channel-last Mario frame stacks."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        channels: tuple[int, ...] = (16, 32, 32),
    ):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 3:
            raise ValueError(
                "IMPALA observations must have three dimensions, got "
                f"{observation_space.shape}"
            )
        self._channels_first = is_image_space_channels_first(
            observation_space
        )
        channel_axis = 0 if self._channels_first else -1
        self._input_channels = observation_space.shape[channel_axis]
        self._channels = tuple(channels)

        stages: list[nn.Module] = []
        input_channels = self._input_channels
        for output_channels in self._channels:
            stages.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    ImpalaResidualBlock(output_channels),
                    ImpalaResidualBlock(output_channels),
                ]
            )
            input_channels = output_channels
        self.cnn = nn.Sequential(*stages)

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()).unsqueeze(0).float()
            if not self._channels_first:
                sample = sample.permute(0, 3, 1, 2)
            flattened_dim = torch.flatten(self.cnn(sample), start_dim=1).shape[1]
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(flattened_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()
        if not self._channels_first:
            x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        return self.projection(torch.flatten(x, start_dim=1))
