"""
Spatio-Temporal Transformer for Multi-Spectral Sequence Modeling & UAV Cross-Attention.
Applies Multi-Head Self-Attention across temporal revisit timestamps and Cross-Modal Attention
between UAV spatial foliage tokens and satellite spectral tokens.
"""

from typing import Tuple, Optional, Any, Dict
import math

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


class PositionalEncoding(nn.Module if HAS_TORCH else object):
    """Sinusoidal positional encoding for temporal revisit timestamps."""
    def __init__(self, d_model: int, max_len: int = 50):
        if HAS_TORCH:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Any) -> Any:
        if HAS_TORCH:
            return x + self.pe[:, :x.size(1)]
        return x


class SpatioTemporalTransformer(nn.Module if HAS_TORCH else object):
    """
    Spatio-Temporal Transformer:
    - Input 1: Multi-spectral satellite time-series (B, T=12, F=16)
    - Input 2: UAV spatial foliage descriptor (B, 10)
    - Multi-Head Self-Attention across time sequence
    - Cross-modal attention between spatial UAV tokens and temporal spectral tokens
    - Multi-task heads for wilt classification, lead-time estimation, and severity scoring.
    """

    def __init__(
        self,
        sat_features: int = 16,
        uav_features: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        num_classes: int = 4,
        dropout: float = 0.15
    ):
        if HAS_TORCH:
            super().__init__()
            self.d_model = d_model

            # Projections to embedding dimension
            self.sat_proj = nn.Linear(sat_features, d_model)
            self.uav_proj = nn.Linear(uav_features, d_model)
            self.pos_encoder = PositionalEncoding(d_model)

            # Temporal Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu"
            )
            self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Cross-Modal Attention: UAV queries attend to satellite spectral keys/values
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.cross_norm = nn.LayerNorm(d_model)

            # Pooling & Classification heads
            self.classifier = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )

            self.lead_time_head = nn.Sequential(
                nn.Linear(d_model * 2, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

            self.severity_head = nn.Sequential(
                nn.Linear(d_model * 2, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def forward(self, sat_series: Any, uav_input: Any) -> Dict[str, Any]:
        """
        Forward pass.
        sat_series: (B, 12, 16)
        uav_input: (B, 10)
        """
        if HAS_TORCH:
            # 1. Project and encode temporal satellite sequence
            sat_tokens = self.sat_proj(sat_series) # (B, 12, d_model)
            sat_tokens = self.pos_encoder(sat_tokens)
            sat_encoded = self.temporal_transformer(sat_tokens) # (B, 12, d_model)

            # Temporal global representation via mean pooling
            sat_global = torch.mean(sat_encoded, dim=1) # (B, d_model)

            # 2. Project UAV spatial descriptor
            uav_token = self.uav_proj(uav_input).unsqueeze(1) # (B, 1, d_model)

            # 3. Cross-modal attention (UAV query attends to temporal satellite tokens)
            cross_out, cross_weights = self.cross_attention(
                query=uav_token,
                key=sat_encoded,
                value=sat_encoded
            )
            uav_refined = self.cross_norm(uav_token + cross_out).squeeze(1) # (B, d_model)

            # 4. Joint representation
            fused = torch.cat([sat_global, uav_refined], dim=1) # (B, d_model * 2)

            logits = self.classifier(fused)
            days = self.lead_time_head(fused)
            severity = self.severity_head(fused)

            return {
                "logits": logits,
                "days_to_visual": days.squeeze(-1),
                "foliar_severity": severity.squeeze(-1),
                "cross_attention_weights": cross_weights,
                "latent_features": fused
            }
        return {"logits": None}
