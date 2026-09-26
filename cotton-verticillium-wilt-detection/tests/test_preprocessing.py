"""
Unit tests for radiometric calibration, cloud screening, and UAV alignment.
"""

import unittest
from src.preprocessing.radiometric_calibration import RadiometricCalibrator
from src.preprocessing.uav_alignment import UAVImageProcessor


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.calibrator = RadiometricCalibrator(dn_scale_factor=10000.0)
        self.uav_proc = UAVImageProcessor()

    def test_dn_to_reflectance(self):
        # 5000 DN should convert to 0.50 reflectance
        refl = self.calibrator.dn_to_reflectance(5000.0)
        self.assertAlmostEqual(refl, 0.50, places=4)

        # Clamping
        refl_clamped = self.calibrator.dn_to_reflectance(15000.0)
        self.assertAlmostEqual(refl_clamped, 1.0, places=4)

    def test_cloud_detection(self):
        cloudy_step = {"B02": 0.35, "B08": 0.45, "B03": 0.20}
        clean_step = {"B02": 0.04, "B08": 0.45, "B03": 0.08}
        self.assertTrue(self.calibrator.detect_cloud_or_shadow(cloudy_step))
        self.assertFalse(self.calibrator.detect_cloud_or_shadow(clean_step))

    def test_temporal_smoothing(self):
        seq = [
            {"B02": 0.04, "B08": 0.50, "B03": 0.08},
            {"B02": 0.35, "B08": 0.40, "B03": 0.25}, # Cloud contaminated
            {"B02": 0.04, "B08": 0.52, "B03": 0.08}
        ]
        smoothed = self.calibrator.smooth_time_series(seq)
        self.assertEqual(len(smoothed), 3)
        # The contaminated step should be interpolated between neighbors
        self.assertAlmostEqual(smoothed[1]["B02"], 0.04, places=2)

    def test_uav_processor_fallback(self):
        metrics = self.uav_proc.extract_canopy_metrics(None)
        self.assertIn("canopy_coverage_pct", metrics)
        self.assertIn("yellowing_foliar_ratio", metrics)


if __name__ == "__main__":
    unittest.main()
