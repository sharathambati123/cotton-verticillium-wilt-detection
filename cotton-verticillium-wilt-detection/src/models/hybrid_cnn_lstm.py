"""
Hybrid CNN-LSTM Architecture for Cotton Verticillium Wilt Detection.
Unifies spatial foliage degradation patterns from UAV aerial views with
temporal spectral variations from multi-spectral satellite time-series.
"""

from typing import Tuple, Optional, Any, Dict
from .cnn_backbone import SpatialCNNBackbone
from .temporal_lstm import TemporalSpectralLSTM

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


class HybridCNNLSTM(nn.Module if HAS_TORCH else object):
    """
    Hybrid CNN-LSTM model:
    - Spatial branch: SpatialCNNBackbone extracts spatial foliage degradation (texture, canopy coverage)
    - Temporal branch: TemporalSpectralLSTM captures temporal spectral variation (NDRE, PSRI, CWSI)
    - Cross-modal fusion layer with multi-task prediction heads:
      1. Diagnostic Wilt Classification (4 classes: Healthy, Early-Stage, Moderate, Severe)
      2. Lead-Time to Visual Scouting Threshold (continuous days regression)
      3. Foliar Degradation Severity Index (continuous [0, 1] regression)
    """

    def __init__(
        self,
        uav_features: int = 10,
        sat_seq_len: int = 12,
        sat_features: int = 16,
        feature_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.2
    ):
        if HAS_TORCH:
            super().__init__()
            self.spatial_cnn = SpatialCNNBackbone(in_features=uav_features, feature_dim=feature_dim, dropout=dropout)
            self.temporal_lstm = TemporalSpectralLSTM(
                input_dim=sat_features,
                hidden_dim=128,
                num_layers=2,
                output_dim=feature_dim,
                dropout=dropout
            )

            # Multimodal Fusion Layer
            fusion_in = feature_dim * 2
            self.fusion_fc = nn.Sequential(
                nn.Linear(fusion_in, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

            # Classification Head
            self.classifier = nn.Linear(feature_dim, num_classes)

            # Regression Head 1: Days to visual scouting threshold
            self.lead_time_head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            )

            # Regression Head 2: Foliar severity score [0, 1]
            self.severity_head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def forward(self, sat_series: Any, uav_input: Any) -> Dict[str, Any]:
        """
        Forward pass.
        Args:
            sat_series: (B, 12, 16)
            uav_input: (B, 10) or (B, 3, 224, 224)
        Returns:
            dict containing:
                - logits: (B, 4)
                - days_to_visual: (B, 1)
                - foliar_severity: (B, 1)
                - temporal_attention: (B, 12, 1)
        """
        if HAS_TORCH:
            spatial_emb = self.spatial_cnn(uav_input)
            temporal_emb, temp_attn = self.temporal_lstm(sat_series)

            # Concatenate features
            fused = torch.cat([spatial_emb, temporal_emb], dim=1)
            latent = self.fusion_fc(fused)

            logits = self.classifier(latent)
            days = self.lead_time_head(latent)
            severity = self.severity_head(latent)

            return {
                "logits": logits,
                "days_to_visual": days.squeeze(-1),
                "foliar_severity": severity.squeeze(-1),
                "temporal_attention": temp_attn,
                "latent_features": latent
            }
        return {"logits": None}
