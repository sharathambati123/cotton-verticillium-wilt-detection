"""
Spatial CNN Backbone for High-Resolution Multi-View UAV Aerial Imagery.
Extracts spatial foliage degradation, canopy structure, and leaf necrosis patterns.
"""

from typing import Tuple, Optional, Any, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module:
            pass


class ResidualBlock(nn.Module if HAS_TORCH else object):
    """Residual convolutional block with batch normalization and LeakyReLU."""
    def __init__(self, channels: int):
        if HAS_TORCH:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: Any) -> Any:
        if HAS_TORCH:
            residual = x
            out = self.act(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return self.act(out)
        return x


class SpatialCNNBackbone(nn.Module if HAS_TORCH else object):
    """
    Multi-scale spatial foliage feature extractor.
    Accepts high-res UAV images (B, 3, H, W) or UAV feature descriptors (B, 10).
    Outputs a rich spatial foliage embedding vector of dimension feature_dim (default 256).
    """

    def __init__(self, in_features: int = 10, feature_dim: int = 256, dropout: float = 0.2):
        if HAS_TORCH:
            super().__init__()
            self.in_features = in_features
            self.feature_dim = feature_dim

            # Image feature extraction path (for full RGB orthophotos)
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 112x112
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 56x56
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.res1 = ResidualBlock(64)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.img_proj = nn.Linear(64, feature_dim)

            # Spatial descriptor projection path (for engineered UAV canopy metrics)
            self.descriptor_mlp = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, feature_dim),
                nn.LayerNorm(feature_dim)
            )

    def forward(self, x: Any) -> Any:
        if HAS_TORCH:
            # Check if input is image tensor (4D) or descriptor tensor (2D)
            if x.dim() == 4:
                feat = self.stem(x)
                feat = self.res1(feat)
                feat = self.pool(feat)
                feat = torch.flatten(feat, 1)
                return self.img_proj(feat)
            else:
                return self.descriptor_mlp(x)
        # Fallback simulation
        return [0.1] * 256
