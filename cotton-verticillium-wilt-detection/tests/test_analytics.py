"""
Unit tests for agricultural analytics, 12-day lead-time verification, and yield impact model.
"""

import unittest
from src.analytics.lead_time_analyzer import LeadTimeAnalyzer
from src.analytics.yield_impact_model import YieldImpactModel
from src.analytics.prescription_map import PrescriptionMapGenerator


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        self.lead_analyzer = LeadTimeAnalyzer()
        self.yield_model = YieldImpactModel()
        self.prescription_gen = PrescriptionMapGenerator()

    def test_lead_time_analysis(self):
        sample_early_parcel = {
            "parcel_id": "COT-FIELD-00001",
            "wilt_class": 1,
            "days_to_visual_symptom": -12.0,
            "time_series": [{"B08": 0.42, "B05": 0.22, "B11": 0.22} for _ in range(12)]
        }
        res = self.lead_analyzer.analyze_parcel_timeline(sample_early_parcel)
        self.assertTrue(res["detected"])
        self.assertAlmostEqual(res["detected_lead_days"], 12.0, places=1)

    def test_fleet_lead_time_aggregation(self):
        dataset = [
            {"wilt_class": 1, "days_to_visual_symptom": -12.1, "time_series": []},
            {"wilt_class": 1, "days_to_visual_symptom": -11.9, "time_series": []},
            {"wilt_class": 0, "days_to_visual_symptom": 999.0, "time_series": []}
        ]
        fleet = self.lead_analyzer.aggregate_fleet_lead_times(dataset)
        self.assertAlmostEqual(fleet["mean_lead_time_days"], 12.0, places=1)
        self.assertTrue(fleet["detection_prior_to_scouting"])

    def test_yield_salvaged_28_percent(self):
        # 35% unmitigated loss - 7% early intervention loss = 28% yield reduction preserved
        scenarios = self.yield_model.calculate_yield_scenarios(total_hectares=100.0)
        summary = scenarios["yield_impact_summary"]

        self.assertAlmostEqual(summary["projected_crop_yield_loss_reduction_pct"], 28.0, places=1)
        # Check pesticide reduction
        pesticide = scenarios["pesticide_optimization"]
        self.assertAlmostEqual(pesticide["chemical_volume_reduction_pct"], 64.0, places=1)

    def test_prescription_geojson_generation(self):
        diagnosed = [
            {
                "parcel_id": "COT-01",
                "latitude": 33.51,
                "longitude": -101.82,
                "predicted_class": 1,
                "foliar_severity": 0.22,
                "predicted_lead_days": -12.1
            }
        ]
        geojson = self.prescription_gen.generate_geojson(diagnosed)
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 1)
        feat = geojson["features"][0]
        self.assertEqual(feat["properties"]["wilt_class"], 1)
        self.assertEqual(feat["properties"]["spray_rate_liters_per_ha"], 40.0)


if __name__ == "__main__":
    unittest.main()
