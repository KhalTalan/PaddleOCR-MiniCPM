# Copyright Lornatang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import Tensor, nn

__all__ = [
    "ResidualDenseBlock", "ResidualResidualDenseBlock", "ResidualFeatureDistillationBlock",
]


class ResidualDenseBlock(nn.Module):
    r"""Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv_1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, 3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, 3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, 3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(channels + growth_channels * 4, channels, 3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out_1 = self.leaky_relu(self.conv_1(x))
        out_2 = self.leaky_relu(self.conv_2(torch.cat([x, out_1], 1)))
        out_3 = self.leaky_relu(self.conv_3(torch.cat([x, out_1, out_2], 1)))
        out_4 = self.leaky_relu(self.conv_4(torch.cat([x, out_1, out_2, out_3], 1)))
        out_5 = self.identity(self.conv_5(torch.cat([x, out_1, out_2, out_3, out_4], 1)))
        out = torch.mul(out_5, 0.2)
        return torch.add(out, identity)


class ResidualResidualDenseBlock(nn.Module):
    r"""Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb_1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb_2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb_3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        out = torch.mul(out, 0.2)
        return torch.add(out, identity)


class ResidualFeatureDistillationBlock(nn.Module):
    r"""Residual feature distillation block.
    `Residual Feature Distillation Network for Lightweight Image Super-Resolution` https://arxiv.org/abs/2009.11551v1 paper.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualFeatureDistillationBlock, self).__init__()
        self.distilled_channels = self.distilled_channels = channels // 2
        self.remaining_channels = self.remaining_channels = channels

        self.conv_1_distilled = nn.Conv2d(channels, self.distilled_channels, 1, stride=1, padding=0)
        self.conv_1_remaining = nn.Conv2d(channels, self.remaining_channels, 3, stride=1, padding=1)
        self.conv_2_distilled = nn.Conv2d(self.remaining_channels, self.distilled_channels, 1, stride=1, padding=0)
        self.conv_2_remaining = nn.Conv2d(self.remaining_channels, self.remaining_channels, 3, stride=1, padding=1)
        self.conv_3_distilled = nn.Conv2d(self.remaining_channels, self.distilled_channels, 1, stride=1, padding=0)
        self.conv_3_remaining = nn.Conv2d(self.remaining_channels, self.remaining_channels, 3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(self.distilled_channels * 4, channels, 1, stride=1, padding=0)

        self.esa = EnhancedSpatialAttention(channels)
        self.leaky_relu = nn.LeakyReLU(0.05, True)

    def forward(self, x: Tensor) -> Tensor:
        distilled_conv_1 = self.conv_1_distilled(x)
        distilled_conv_1 = self.leaky_relu(distilled_conv_1)
        remaining_conv_1 = self.conv_1_remaining(x)
        remaining_conv_1 = torch.add(remaining_conv_1, x)
        remaining_conv_1 = self.leaky_relu(remaining_conv_1)

        distilled_conv_2 = self.conv_2_distilled(remaining_conv_1)
        distilled_conv_2 = self.leaky_relu(distilled_conv_2)
        remaining_conv_2 = self.conv_2_remaining(remaining_conv_1)
        remaining_conv_2 = torch.add(remaining_conv_2, remaining_conv_1)
        remaining_conv_2 = self.leaky_relu(remaining_conv_2)

        distilled_conv_3 = self.conv_3_distilled(remaining_conv_2)
        distilled_conv_3 = self.leaky_relu(distilled_conv_3)
        remaining_conv_3 = self.conv_3_remaining(remaining_conv_2)
        remaining_conv_3 = torch.add(remaining_conv_3, remaining_conv_2)
        remaining_conv_3 = self.leaky_relu(remaining_conv_3)

        remaining_conv_4 = self.conv_4(remaining_conv_3)
        remaining_conv_4 = self.leaky_relu(remaining_conv_4)

        out = torch.cat([distilled_conv_1, distilled_conv_2, distilled_conv_3, remaining_conv_4], 1)
        out = self.conv_5(out)

        return self.esa(out)
