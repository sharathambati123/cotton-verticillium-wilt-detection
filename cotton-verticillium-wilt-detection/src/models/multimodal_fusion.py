"""
Gated Multi-Modal Fusion Engine and Model Factory.
Dynamically balances confidence between satellite spectral time-series and UAV high-resolution spatial views.
"""

from typing import Tuple, Optional, Any, Dict
from .hybrid_cnn_lstm import HybridCNNLSTM
from .spatio_temporal_transformer import SpatioTemporalTransformer

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


class GatedMultiModalFusion(nn.Module if HAS_TORCH else object):
    """
    Gated Attention Multi-Modal Network.
    Computes modality gating coefficients (alpha_sat, alpha_uav) to dynamically
    downweight satellite modality during partial cloud occlusion or downweight UAV
    modality if wind-blur/motion occurs.
    """

    def __init__(
        self,
        sat_features: int = 16,
        uav_features: int = 10,
        hidden_dim: int = 256,
        num_classes: int = 4
    ):
        if HAS_TORCH:
            super().__init__()
            self.sat_encoder = nn.Sequential(
                nn.Linear(sat_features * 12, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            self.uav_encoder = nn.Sequential(
                nn.Linear(uav_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            )

            self.classifier = nn.Linear(hidden_dim, num_classes)
            self.lead_time_head = nn.Linear(hidden_dim, 1)
            self.severity_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, sat_series: Any, uav_input: Any) -> Dict[str, Any]:
        if HAS_TORCH:
            b = sat_series.size(0)
            sat_flat = sat_series.view(b, -1)
            h_sat = self.sat_encoder(sat_flat)
            h_uav = self.uav_encoder(uav_input)

            gates = self.gate(torch.cat([h_sat, h_uav], dim=-1))
            g_sat = gates[:, 0:1]
            g_uav = gates[:, 1:2]

            fused = g_sat * h_sat + g_uav * h_uav

            logits = self.classifier(fused)
            days = self.lead_time_head(fused).squeeze(-1)
            severity = self.severity_head(fused).squeeze(-1)

            return {
                "logits": logits,
                "days_to_visual": days,
                "foliar_severity": severity,
                "modality_gates": gates,
                "latent_features": fused
            }
        return {"logits": None}


def build_model(model_type: str = "cnn_lstm", **kwargs) -> Any:
    """
    Factory function to construct model architectures:
      - 'cnn_lstm': Hybrid CNN-LSTM
      - 'transformer': Spatio-Temporal Transformer
      - 'gated_fusion': Gated Multi-Modal Network
    """
    model_type = model_type.lower()
    if model_type in ["cnn_lstm", "hybrid"]:
        return HybridCNNLSTM(**kwargs)
    elif model_type in ["transformer", "spatio_temporal"]:
        return SpatioTemporalTransformer(**kwargs)
    elif model_type in ["gated", "gated_fusion"]:
        return GatedMultiModalFusion(**kwargs)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'cnn_lstm', 'transformer', or 'gated_fusion'.")
