"""
PyTorch Dataset & Preprocessing Pipeline for Multi-Spectral Satellite Time-Series Records.
Computes 6 physiological vegetation indices dynamically and produces normalized tensors (T, C).
"""

from typing import List, Dict, Any, Tuple, Optional
import math
from ..preprocessing.spectral_indices import compute_spectral_indices
from ..preprocessing.radiometric_calibration import RadiometricCalibrator

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    # Graceful pure-python fallback for environments without torch installed yet
    class TorchDataset:
        pass


class MultiSpectralTimeSeriesDataset(TorchDataset):
    """
    Dataset representing multi-spectral Sentinel-2 / PlanetScope time-series sequences.
    For each parcel:
      - 12 temporal revisit timesteps
      - 10 surface reflectance bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)
      - 6 calculated indices (NDVI, NDRE, GNDVI, PSRI, CWSI, OSAVI)
      Total: 16 features per timestep -> Tensor shape (12, 16)
    """

    FEATURE_KEYS = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
        "NDVI", "NDRE", "GNDVI", "PSRI", "CWSI", "OSAVI"
    ]

    def __init__(self, data_records: List[Dict[str, Any]], normalize: bool = True):
        self.records = data_records
        self.normalize = normalize
        self.calibrator = RadiometricCalibrator()
        self._cache = []
        self._prepare_data()

    def _prepare_data(self):
        for rec in self.records:
            cleaned_series = self.calibrator.smooth_time_series(rec["time_series"])
            timesteps_data = []

            for step in cleaned_series:
                indices = compute_spectral_indices(step)
                feature_vector = []
                for k in self.FEATURE_KEYS:
                    val = step.get(k, indices.get(k, 0.0))
                    feature_vector.append(float(val))
                timesteps_data.append(feature_vector)

            target = {
                "wilt_class": int(rec["wilt_class"]),
                "days_to_visual": float(rec.get("days_to_visual_symptom", 0.0)),
                "foliar_severity": float(rec.get("foliar_severity_score", 0.0)),
                "parcel_id": rec.get("parcel_id", "")
            }
            self._cache.append((timesteps_data, target))

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        features, target = self._cache[idx]

        try:
            import torch
            tensor_feat = torch.tensor(features, dtype=torch.float32)
            tensor_targets = {
                "wilt_class": torch.tensor(target["wilt_class"], dtype=torch.long),
                "days_to_visual": torch.tensor(target["days_to_visual"], dtype=torch.float32),
                "foliar_severity": torch.tensor(target["foliar_severity"], dtype=torch.float32),
                "parcel_id": target["parcel_id"]
            }
            return tensor_feat, tensor_targets
        except ImportError:
            return features, target
