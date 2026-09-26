"""
Bidirectional LSTM with Temporal Attention Pooling for Multi-Spectral Satellite Time-Series.
Captures temporal spectral variations and physiological stress dynamics across revisit dates.
"""

from typing import Tuple, Optional, Any, Dict

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


class TemporalAttention(nn.Module if HAS_TORCH else object):
    """
    Self-attention layer across temporal observation timesteps.
    Highlights crucial disease inflection dates where NDRE, PSRI, and CWSI shift.
    """
    def __init__(self, hidden_dim: int):
        if HAS_TORCH:
            super().__init__()
            self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: Any) -> Tuple[Any, Any]:
        """
        lstm_outputs: (B, T, hidden_dim)
        Returns:
            context_vector: (B, hidden_dim)
            attention_weights: (B, T, 1)
        """
        if HAS_TORCH:
            scores = self.query(lstm_outputs) # (B, T, 1)
            weights = F.softmax(scores, dim=1) # (B, T, 1)
            context = torch.sum(lstm_outputs * weights, dim=1) # (B, hidden_dim)
            return context, weights
        return lstm_outputs, None


class TemporalSpectralLSTM(nn.Module if HAS_TORCH else object):
    """
    Bidirectional LSTM with Attention for 12-timestep multi-spectral sequences.
    Input shape: (Batch, Timesteps=12, Features=16)
    Output shape: (Batch, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 256,
        dropout: float = 0.25
    ):
        if HAS_TORCH:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            # Feature projection & normalization
            self.in_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

            lstm_out_dim = hidden_dim * 2
            self.attention = TemporalAttention(lstm_out_dim)
            self.out_proj = nn.Sequential(
                nn.Linear(lstm_out_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

    def forward(self, x: Any) -> Tuple[Any, Any]:
        """
        Forward pass.
        Args:
            x: (B, 12, 16)
        Returns:
            temporal_embedding: (B, output_dim)
            attention_weights: (B, 12, 1)
        """
        if HAS_TORCH:
            proj = self.in_proj(x)
            lstm_out, _ = self.lstm(proj)
            context, attn_weights = self.attention(lstm_out)
            embedding = self.out_proj(context)
            return embedding, attn_weights
        return x, None
