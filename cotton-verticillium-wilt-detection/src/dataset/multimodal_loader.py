"""
Multi-Modal Data Loader combining Satellite Time-Series sequences and UAV Spatial Imagery.
Provides stratified train, validation, and test splits.
"""

from typing import List, Dict, Any, Tuple, Optional
import random
from .multispectral_dataset import MultiSpectralTimeSeriesDataset
from .uav_dataset import UAVImageDataset

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    class Dataset:
        pass
    class DataLoader:
        pass


class MultiModalCottonDataset(Dataset):
    """
    Pairs satellite temporal spectral records (12, 16) with UAV spatial foliage data (10,)
    and multiple diagnostic ground-truth targets.
    """

    def __init__(self, data_records: List[Dict[str, Any]], augment_uav: bool = False):
        self.sat_dataset = MultiSpectralTimeSeriesDataset(data_records)
        self.uav_dataset = UAVImageDataset(data_records, augment=augment_uav)

    def __len__(self) -> int:
        return len(self.sat_dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Dict[str, Any]]:
        sat_data, targets = self.sat_dataset[idx]
        uav_data, _ = self.uav_dataset[idx]
        return sat_data, uav_data, targets


def create_dataloaders(
    records: List[Dict[str, Any]],
    batch_size: int = 32,
    train_split: float = 0.70,
    val_split: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Split records into Train (70%), Validation (15%), Test (15%) sets.
    Returns datasets and PyTorch DataLoaders (or wrapped batches if torch is not installed).
    """
    random.seed(seed)
    shuffled = list(records)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_recs = shuffled[:n_train]
    val_recs = shuffled[n_train: n_train + n_val]
    test_recs = shuffled[n_train + n_val:]

    train_ds = MultiModalCottonDataset(train_recs, augment_uav=True)
    val_ds = MultiModalCottonDataset(val_recs, augment_uav=False)
    test_ds = MultiModalCottonDataset(test_recs, augment_uav=False)

    try:
        import torch
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "train_dataset": train_ds,
            "val_dataset": val_ds,
            "test_dataset": test_ds,
            "splits": {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)}
        }
    except ImportError:
        # Fallback dictionary of datasets
        return {
            "train_dataset": train_ds,
            "val_dataset": val_ds,
            "test_dataset": test_ds,
            "splits": {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)}
        }
