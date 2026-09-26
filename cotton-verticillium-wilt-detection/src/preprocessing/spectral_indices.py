"""
Multi-spectral vegetation and physiological stress indices computation engine.
Optimized for Sentinel-2 MSI and MicaSense multi-spectral imagery for cotton canopy analysis.
"""

from typing import Dict, Any, Union, List
import math


def safe_div(num: float, den: float, eps: float = 1e-7) -> float:
    """Safely divide two numbers avoiding zero division and inf/nan."""
    if abs(den) < eps:
        return 0.0
    res = num / den
    if math.isnan(res) or math.isinf(res):
        return 0.0
    return res


def compute_spectral_indices(bands: Dict[str, float], eps: float = 1e-7) -> Dict[str, float]:
    """
    Compute key physiological and canopy stress indices from multi-spectral band reflectance.

    Expected bands in `bands`:
      - 'B02': Blue (490 nm)
      - 'B03': Green (560 nm)
      - 'B04': Red (665 nm)
      - 'B05': RedEdge 1 (705 nm)
      - 'B06': RedEdge 2 (740 nm)
      - 'B07': RedEdge 3 (783 nm)
      - 'B08': NIR (842 nm)
      - 'B8A': Narrow NIR (865 nm)
      - 'B11': SWIR 1 (1610 nm)
      - 'B12': SWIR 2 (2190 nm)

    Returns:
      Dict with NDVI, NDRE, GNDVI, PSRI, CWSI, OSAVI.
    """
    blue = bands.get("B02", 0.0)
    green = bands.get("B03", 0.0)
    red = bands.get("B04", 0.0)
    re1 = bands.get("B05", 0.0)
    re2 = bands.get("B06", 0.0)
    nir = bands.get("B08", bands.get("B8A", 0.0))
    swir1 = bands.get("B11", 0.0)
    swir2 = bands.get("B12", 0.0)

    # 1. NDVI (Normalized Difference Vegetation Index)
    # Measures general canopy greenness and biomass
    ndvi = safe_div(nir - red, nir + red + eps, eps)

    # 2. NDRE (Normalized Difference Red Edge)
    # Critical for early wilt: Red edge penetrates deeper into canopy and detects
    # subtle sub-visual chlorophyll loss before red channel saturates.
    ndre = safe_div(nir - re1, nir + re1 + eps, eps)

    # 3. GNDVI (Green Normalized Difference Vegetation Index)
    # Highly sensitive to wide variations in chlorophyll concentration.
    gndvi = safe_div(nir - green, nir + green + eps, eps)

    # 4. PSRI (Plant Senescence Reflectance Index)
    # (Red - Green) / RedEdge. Increases during senescence, carotenoid build-up,
    # and vascular pathogen attack (Verticillium wilt vascular occlusion).
    psri = safe_div(red - green, re1 + eps, eps)

    # 5. CWSI (Canopy Water Stress Index - spectral approximation using SWIR/NIR)
    # Verticillium dahliae blocks xylem vessels, causing acute hydraulic dysfunction.
    # Relative water stress derived from SWIR moisture absorption vs NIR structure.
    msi = safe_div(swir1, nir + eps, eps)  # Moisture Stress Index
    cwsi = min(1.0, max(0.0, (msi - 0.4) / 0.8))

    # 6. OSAVI (Optimized Soil-Adjusted Vegetation Index)
    # Eliminates bare soil background reflectance in young or defoliated cotton rows.
    # OSAVI = (1 + 0.16) * (NIR - Red) / (NIR + Red + 0.16)
    osavi = safe_div(1.16 * (nir - red), nir + red + 0.16 + eps, eps)

    return {
        "NDVI": ndvi,
        "NDRE": ndre,
        "GNDVI": gndvi,
        "PSRI": psri,
        "CWSI": cwsi,
        "OSAVI": osavi,
    }


def compute_time_series_indices(time_series: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Process an entire temporal sequence of multi-spectral observations.
    Appends the 6 computed indices to the raw bands for each time step.
    """
    processed = []
    for step in time_series:
        indices = compute_spectral_indices(step)
        combined = {**step, **indices}
        processed.append(combined)
    return processed
