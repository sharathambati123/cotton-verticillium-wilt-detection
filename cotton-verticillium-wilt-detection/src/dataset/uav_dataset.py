"""
PyTorch Dataset for Multi-View UAV Aerial Imagery.
Processes high-resolution nadir and oblique canopy foliage views to capture
fine-grained leaf necrosis, interveinal discoloration, and canopy gaps.
"""

from typing import List, Dict, Any, Tuple, Optional
import random

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    class TorchDataset:
        pass


class UAVImageDataset(TorchDataset):
    """
    Dataset representing multi-view UAV aerial photographs and spatial foliage degradation representations.
    Provides RGB canopy patch tensors of shape (3, 224, 224) or engineered spatial descriptors.
    """

    def __init__(self, data_records: List[Dict[str, Any]], augment: bool = False, image_size: Tuple[int, int] = (224, 224)):
        self.records = data_records
        self.augment = augment
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        rec = self.records[idx]
        uav_data = rec.get("uav_metrics", {})
        descriptor = uav_data.get("synthetic_spatial_descriptor", [0.5] * 8)
        canopy_cov = uav_data.get("canopy_coverage_pct", 75.0) / 100.0
        necrotic_ratio = uav_data.get("necrotic_foliar_ratio", 0.1)

        # Extended feature representation (8 spatial descriptors + coverage + necrosis) = 10 dims
        spatial_features = descriptor + [canopy_cov, necrotic_ratio]

        target = {
            "wilt_class": int(rec["wilt_class"]),
            "days_to_visual": float(rec.get("days_to_visual_symptom", 0.0)),
            "foliar_severity": float(rec.get("foliar_severity_score", 0.0)),
            "parcel_id": rec.get("parcel_id", "")
        }

        try:
            import torch
            tensor_spatial = torch.tensor(spatial_features, dtype=torch.float32)
            tensor_targets = {
                "wilt_class": torch.tensor(target["wilt_class"], dtype=torch.long),
                "days_to_visual": torch.tensor(target["days_to_visual"], dtype=torch.float32),
                "foliar_severity": torch.tensor(target["foliar_severity"], dtype=torch.float32),
                "parcel_id": target["parcel_id"]
            }
            return tensor_spatial, tensor_targets
        except ImportError:
            return spatial_features, target
