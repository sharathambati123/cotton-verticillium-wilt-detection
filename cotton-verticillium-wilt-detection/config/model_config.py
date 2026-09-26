"""
Configuration schemas and loaders for Cotton Verticillium Wilt Detection.
Supports yaml loading with dataclass validation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import os
import json


@dataclass
class SatelliteConfig:
    num_samples: int = 5200
    sequence_length: int = 12
    raw_bands: List[str] = field(default_factory=lambda: [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"
    ])
    computed_indices: List[str] = field(default_factory=lambda: [
        "NDVI", "NDRE", "GNDVI", "PSRI", "CWSI", "OSAVI"
    ])
    time_interval_days: int = 6

    @property
    def total_features_per_step(self) -> int:
        return len(self.raw_bands) + len(self.computed_indices)


@dataclass
class UAVConfig:
    image_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    views: List[str] = field(default_factory=lambda: ["nadir", "oblique_30", "oblique_45"])
    feature_matching: str = "ORB"


@dataclass
class ModelArchitectureConfig:
    cnn_feature_dim: int = 256
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    transformer_d_model: int = 256
    transformer_nhead: int = 8
    transformer_layers: int = 4
    fusion_dim: int = 256
    num_classes: int = 4


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    focal_gamma: float = 2.0
    focal_alpha: List[float] = field(default_factory=lambda: [0.15, 0.40, 0.25, 0.20])


@dataclass
class AgronomicImpactConfig:
    early_detection_lead_days: int = 12
    baseline_unmitigated_loss_pct: float = 35.0
    standard_scouting_loss_pct: float = 26.0
    early_intervention_loss_pct: float = 7.0
    yield_salvaged_pct: float = 28.0
    average_cotton_yield_kg_ha: float = 2800.0
    cotton_price_per_kg_usd: float = 1.85
    targeted_spray_saving_pct: float = 64.0


@dataclass
class SystemConfig:
    project_name: str = "cotton-verticillium-wilt-detection"
    random_seed: int = 42
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    models: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    agronomic: AgronomicImpactConfig = field(default_factory=AgronomicImpactConfig)


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration from YAML file or return default dataclass."""
    cfg = SystemConfig()
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data:
                if "dataset" in data:
                    sat_data = data["dataset"].get("satellite", {})
                    if sat_data:
                        cfg.satellite.num_samples = sat_data.get("num_samples", cfg.satellite.num_samples)
                        cfg.satellite.sequence_length = sat_data.get("sequence_length", cfg.satellite.sequence_length)
                        cfg.satellite.raw_bands = sat_data.get("raw_bands", cfg.satellite.raw_bands)
                        cfg.satellite.computed_indices = sat_data.get("computed_indices", cfg.satellite.computed_indices)
                if "models" in data:
                    m = data["models"]
                    cfg.models.cnn_feature_dim = m.get("cnn_backbone", {}).get("feature_dim", cfg.models.cnn_feature_dim)
                    cfg.models.lstm_hidden_dim = m.get("temporal_lstm", {}).get("hidden_dim", cfg.models.lstm_hidden_dim)
                    cfg.models.transformer_d_model = m.get("spatio_temporal_transformer", {}).get("d_model", cfg.models.transformer_d_model)
                if "agronomic_impact" in data:
                    agro = data["agronomic_impact"]
                    cfg.agronomic.early_detection_lead_days = agro.get("early_detection_lead_days", 12)
                    cfg.agronomic.yield_salvaged_pct = agro.get("yield_salvaged_pct", 28.0)
        except Exception:
            pass  # Fall back to defaults
    return cfg
