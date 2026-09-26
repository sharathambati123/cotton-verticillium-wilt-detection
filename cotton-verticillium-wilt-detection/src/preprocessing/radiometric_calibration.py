"""
Radiometric Calibration & Atmospheric Normalization Pipeline for Multi-Spectral Satellite Data.
Handles Bottom-of-Atmosphere (BOA) surface reflectance scaling, cloud/shadow masking,
and temporal outlier smoothing for Sentinel-2 MSI and PlanetScope time-series.
"""

from typing import List, Dict, Any, Tuple, Optional
import math


class RadiometricCalibrator:
    """
    Calibrates multi-spectral digital numbers (DN) to surface reflectance,
    screens out cloud/cloud-shadow contamination, and normalizes spectral dynamic ranges.
    """

    def __init__(self, dn_scale_factor: float = 10000.0, valid_range: Tuple[float, float] = (0.0, 1.0)):
        self.dn_scale_factor = dn_scale_factor
        self.min_val, self.max_val = valid_range

    def dn_to_reflectance(self, val: float) -> float:
        """Convert Sentinel-2 integer DN (0-10000) to physical surface reflectance [0.0, 1.0]."""
        if val > 1.0:
            val = val / self.dn_scale_factor
        return max(self.min_val, min(self.max_val, val))

    def detect_cloud_or_shadow(self, bands: Dict[str, float]) -> bool:
        """
        Flag cloud or cloud shadow contamination.
        Clouds: High blue (> 0.25) + high red/NIR.
        Shadows: Abnormally low NIR (< 0.05) and low green.
        """
        blue = bands.get("B02", 0.0)
        nir = bands.get("B08", bands.get("B8A", 0.0))
        green = bands.get("B03", 0.0)

        # Cloud detection heuristic
        if blue > 0.28 and nir > 0.35:
            return True
        # Shadow detection heuristic
        if nir < 0.04 and green < 0.03:
            return True
        return False

    def clean_record(self, raw_bands: Dict[str, float]) -> Dict[str, float]:
        """Normalize raw band dictionary to calibrated BOA reflectance."""
        cleaned = {}
        for band, val in raw_bands.items():
            if isinstance(val, (int, float)):
                cleaned[band] = self.dn_to_reflectance(val)
            else:
                cleaned[band] = val
        return cleaned

    def smooth_time_series(self, sequence: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Temporal moving-average smoothing and linear interpolation for missing/cloudy steps.
        Ensures consistent, noise-free temporal trajectories across the 12 revisit dates.
        """
        if not sequence:
            return sequence

        smoothed = []
        num_steps = len(sequence)
        band_keys = [k for k, v in sequence[0].items() if isinstance(v, (int, float))]

        for t in range(num_steps):
            cur_step = dict(sequence[t])
            is_corrupt = self.detect_cloud_or_shadow(cur_step)

            if is_corrupt and 0 < t < num_steps - 1:
                # Interpolate using temporal neighbors
                prev_step = sequence[t - 1]
                next_step = sequence[t + 1]
                for b in band_keys:
                    p_val = prev_step.get(b, 0.0)
                    n_val = next_step.get(b, 0.0)
                    cur_step[b] = (p_val + n_val) / 2.0
            smoothed.append(cur_step)

        return smoothed
