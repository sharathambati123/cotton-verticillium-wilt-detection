"""
Unit tests for multi-spectral vegetation indices computation.
"""

import unittest
from src.preprocessing.spectral_indices import compute_spectral_indices, safe_div


class TestSpectralIndices(unittest.TestCase):
    def test_safe_div(self):
        self.assertEqual(safe_div(10.0, 2.0), 5.0)
        self.assertEqual(safe_div(5.0, 0.0), 0.0)
        self.assertEqual(safe_div(0.0, 0.0), 0.0)

    def test_healthy_cotton_indices(self):
        # Typical healthy cotton reflectance: low red, high NIR, moderate green & red-edge
        bands = {
            "B02": 0.04, # Blue
            "B03": 0.08, # Green
            "B04": 0.05, # Red
            "B05": 0.15, # RedEdge 1
            "B06": 0.30, # RedEdge 2
            "B07": 0.42, # RedEdge 3
            "B08": 0.52, # NIR
            "B8A": 0.50, # Narrow NIR
            "B11": 0.10, # SWIR 1
            "B12": 0.06, # SWIR 2
        }
        indices = compute_spectral_indices(bands)

        # In healthy cotton:
        # NDVI should be high (~0.8)
        self.assertGreater(indices["NDVI"], 0.70)
        # NDRE should be strong (> 0.50)
        self.assertGreater(indices["NDRE"], 0.50)
        # PSRI should be low (< 0.10)
        self.assertLess(indices["PSRI"], 0.10)
        # CWSI should be low (< 0.30)
        self.assertLess(indices["CWSI"], 0.30)

    def test_early_wilt_indices_divergence(self):
        # Early-stage wilt: red is still relatively low, but RedEdge reflectance shifts
        # and SWIR water stress rises due to vascular occlusion
        bands_early_wilt = {
            "B02": 0.04,
            "B03": 0.08,
            "B04": 0.06, # Still appears visually normal/green
            "B05": 0.22, # RedEdge shift (chlorophyll loss at red edge)
            "B06": 0.28,
            "B07": 0.35,
            "B08": 0.42, # Modest NIR drop
            "B8A": 0.40,
            "B11": 0.22, # Moisture stress index rises
            "B12": 0.14,
        }
        indices = compute_spectral_indices(bands_early_wilt)

        # NDRE drops sharply before visible red channel changes
        self.assertLess(indices["NDRE"], 0.40)
        # CWSI increases indicating water deficit from fungal xylem blockage
        self.assertGreater(indices["CWSI"], 0.10)


if __name__ == "__main__":
    unittest.main()
