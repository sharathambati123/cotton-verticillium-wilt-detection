"""
Variable-Rate Pesticide & Bio-Fungicide Prescription Map Generator.
Translates model predictions into geo-referenced application zones (GeoJSON)
compatible with agricultural spray drones and variable-rate tractor rigs.
"""

from typing import List, Dict, Any
import json


class PrescriptionMapGenerator:
    """
    Generates precision variable-rate treatment maps based on parcel diagnostic severity.
    """

    TREATMENT_PROTOCOLS = {
        0: {
            "action": "Monitor Only",
            "chemical_rate_l_ha": 0.0,
            "description": "Healthy canopy; maintain standard irrigation schedule."
        },
        1: {
            "action": "Targeted Early-Stage Bio-Fungicide & SAR Inducer",
            "chemical_rate_l_ha": 40.0,
            "description": "Pre-visual wilt detected 12 days early. Apply Bacillus subtilis / potassium silicate systemic acquired resistance (SAR) agent."
        },
        2: {
            "action": "Curative Chemigation",
            "chemical_rate_l_ha": 95.0,
            "description": "Visual symptoms active. Deploy systemic vascular fungicide and buffer zone barrier."
        },
        3: {
            "action": "Quarantine & Canopy Sanitation",
            "chemical_rate_l_ha": 140.0,
            "description": "Severe necrosis. Apply defoliant/biocide to prevent spore transmission to adjacent cotton rows."
        }
    }

    def generate_geojson(self, diagnosed_parcels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Produce a GeoJSON FeatureCollection of treatment parcels.
        """
        features = []

        for p in diagnosed_parcels:
            lat = p.get("latitude", 33.50)
            lon = p.get("longitude", -101.80)
            wilt_class = p.get("predicted_class", p.get("wilt_class", 0))
            severity = p.get("foliar_severity", p.get("foliar_severity_score", 0.0))
            lead_days = p.get("predicted_lead_days", p.get("days_to_visual_symptom", 0.0))

            protocol = self.TREATMENT_PROTOCOLS.get(wilt_class, self.TREATMENT_PROTOCOLS[0])

            # Generate parcel boundary polygon (~50m x 50m)
            delta = 0.00045
            polygon_coords = [[
                [round(lon - delta, 6), round(lat - delta, 6)],
                [round(lon + delta, 6), round(lat - delta, 6)],
                [round(lon + delta, 6), round(lat + delta, 6)],
                [round(lon - delta, 6), round(lat + delta, 6)],
                [round(lon - delta, 6), round(lat - delta, 6)],
            ]]

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon_coords
                },
                "properties": {
                    "parcel_id": p.get("parcel_id", "UNKNOWN"),
                    "wilt_class": wilt_class,
                    "class_name": ["Healthy", "Early-Stage Wilt (Pre-Visual)", "Moderate Wilt", "Severe Wilt"][wilt_class],
                    "foliar_severity": round(severity, 3),
                    "days_to_visual_scouting": round(lead_days, 1),
                    "treatment_action": protocol["action"],
                    "spray_rate_liters_per_ha": protocol["chemical_rate_l_ha"],
                    "prescription_notes": protocol["description"]
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "metadata": {
                "crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
                "target_crop": "Gossypium hirsutum (Cotton)",
                "pathogen": "Verticillium dahliae",
                "generation_standard": "Variable-Rate Application (VRA) Prescription Map"
            },
            "features": features
        }

    def save_prescription(self, geojson_data: Dict[str, Any], filepath: str):
        """Write prescription map to GeoJSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)
