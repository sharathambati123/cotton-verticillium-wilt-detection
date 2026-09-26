"""
Synthetic Data Generator for Cotton Verticillium Wilt Detection.
Generates 5,000+ realistic multi-spectral satellite time-series records (12 revisit timesteps)
and multi-view UAV aerial imagery representations based on plant pathology dynamics of
Verticillium dahliae vascular infection in Gossypium hirsutum (cotton).
"""

import os
import json
import random
import math
from typing import List, Dict, Any, Tuple


def generate_single_parcel(parcel_idx: int, wilt_class: int) -> Dict[str, Any]:
    """
    Simulate a single cotton field parcel over a 12-timestep temporal sequence
    (approx 72 days across squaring, flowering, and boll-opening stages).

    Class 0: Healthy
    Class 1: Early-Stage Wilt (Detected at sub-visual phase, ~12 days before scouting threshold)
    Class 2: Moderate Wilt (Visual scouting threshold reached, foliar chlorosis evident)
    Class 3: Severe Wilt (Extensive necrosis, vascular browning, canopy defoliation)
    """
    timesteps = 12
    records = []

    # Biological progression variables
    if wilt_class == 0:  # Healthy
        infection_onset_t = 999
        days_to_visual = 999.0
        severity_score = random.uniform(0.0, 0.08)
    elif wilt_class == 1:  # Early-stage wilt (detected ~12 days prior to visual symptoms)
        infection_onset_t = random.randint(4, 6)
        # Visual symptom would appear at timestep infection_onset_t + 2 (12 days later @ 6 days/timestep)
        days_to_visual = -12.0 + random.uniform(-1.5, 1.5)
        severity_score = random.uniform(0.12, 0.30)
    elif wilt_class == 2:  # Moderate wilt (at or near visual threshold)
        infection_onset_t = random.randint(2, 4)
        days_to_visual = random.uniform(-2.0, 4.0)
        severity_score = random.uniform(0.35, 0.65)
    else:  # Severe wilt
        infection_onset_t = random.randint(1, 2)
        days_to_visual = random.uniform(8.0, 20.0)
        severity_score = random.uniform(0.70, 0.98)

    for t in range(timesteps):
        # Baseline healthy cotton phenological canopy curve
        phenology_factor = math.sin((t / timesteps) * math.pi)  # Peaks around mid-season (flowering)
        base_nir = 0.42 + 0.14 * phenology_factor + random.gauss(0, 0.015)
        base_red = 0.05 - 0.02 * phenology_factor + random.gauss(0, 0.005)
        base_green = 0.08 - 0.01 * phenology_factor + random.gauss(0, 0.005)
        base_re1 = 0.14 + 0.04 * phenology_factor + random.gauss(0, 0.01)
        base_re2 = 0.28 + 0.08 * phenology_factor + random.gauss(0, 0.01)
        base_re3 = 0.38 + 0.10 * phenology_factor + random.gauss(0, 0.01)
        base_swir1 = 0.12 - 0.03 * phenology_factor + random.gauss(0, 0.01)
        base_swir2 = 0.07 - 0.02 * phenology_factor + random.gauss(0, 0.005)

        # Apply disease stress if beyond infection onset
        if t >= infection_onset_t:
            progress = (t - infection_onset_t + 1)
            if wilt_class == 1:
                # Sub-visual phase:
                # Xylem vascular clogging causes early water stress (SWIR rises)
                # Chlorophyll begins subtle decline at red edge (RE1 drops, NDRE drops)
                # But visible RED has NOT changed significantly yet!
                base_swir1 += 0.05 * progress
                base_re1 += 0.03 * progress
                base_nir -= 0.04 * progress
                # Minimal red change to simulate sub-visual condition
                base_red += 0.008 * progress
            elif wilt_class == 2:
                # Moderate wilt: visible chlorosis ("tiger-stripe")
                base_red += 0.04 * progress
                base_green += 0.03 * progress
                base_nir -= 0.08 * progress
                base_swir1 += 0.08 * progress
                base_re1 += 0.05 * progress
            elif wilt_class == 3:
                # Severe wilt: necrosis, leaf desiccation, canopy collapse
                base_red += 0.09 * progress
                base_nir -= 0.14 * progress
                base_swir1 += 0.12 * progress
                base_re1 += 0.08 * progress

        step_bands = {
            "B02": round(max(0.01, min(0.30, 0.04 + random.gauss(0, 0.005))), 4),
            "B03": round(max(0.01, min(0.35, base_green)), 4),
            "B04": round(max(0.01, min(0.40, base_red)), 4),
            "B05": round(max(0.01, min(0.50, base_re1)), 4),
            "B06": round(max(0.01, min(0.60, base_re2)), 4),
            "B07": round(max(0.01, min(0.65, base_re3)), 4),
            "B08": round(max(0.02, min(0.70, base_nir)), 4),
            "B8A": round(max(0.02, min(0.70, base_nir * 0.98)), 4),
            "B11": round(max(0.01, min(0.50, base_swir1)), 4),
            "B12": round(max(0.01, min(0.40, base_swir2)), 4),
        }
        records.append(step_bands)

    # Simulated UAV multi-view spatial features
    # Representing texture, canopy coverage %, and necrotic patch ratios
    if wilt_class == 0:
        canopy_coverage = random.uniform(85.0, 96.0)
        necrotic_ratio = random.uniform(0.0, 0.02)
        uav_features = [random.uniform(0.7, 0.9) for _ in range(8)]
    elif wilt_class == 1:
        canopy_coverage = random.uniform(78.0, 87.0)
        necrotic_ratio = random.uniform(0.03, 0.08) # Micro-spots
        uav_features = [random.uniform(0.4, 0.7) for _ in range(8)]
    elif wilt_class == 2:
        canopy_coverage = random.uniform(60.0, 75.0)
        necrotic_ratio = random.uniform(0.12, 0.28) # Discoloration
        uav_features = [random.uniform(0.2, 0.5) for _ in range(8)]
    else:
        canopy_coverage = random.uniform(35.0, 58.0)
        necrotic_ratio = random.uniform(0.35, 0.75) # Defoliation
        uav_features = [random.uniform(0.05, 0.3) for _ in range(8)]

    # Projected yield loss without intervention
    if wilt_class == 0:
        unmitigated_loss_pct = 0.0
    elif wilt_class == 1:
        unmitigated_loss_pct = round(random.uniform(32.0, 38.0), 1)  # ~35%
    elif wilt_class == 2:
        unmitigated_loss_pct = round(random.uniform(40.0, 52.0), 1)
    else:
        unmitigated_loss_pct = round(random.uniform(60.0, 80.0), 1)

    return {
        "parcel_id": f"COT-FIELD-{parcel_idx:05d}",
        "latitude": round(33.5 + random.uniform(-0.4, 0.4), 6),
        "longitude": round(-101.8 + random.uniform(-0.4, 0.4), 6),
        "wilt_class": wilt_class,
        "class_name": ["Healthy", "Early-Stage Wilt (Pre-Visual)", "Moderate Wilt", "Severe Wilt"][wilt_class],
        "days_to_visual_symptom": round(days_to_visual, 1),
        "foliar_severity_score": round(severity_score, 3),
        "unmitigated_yield_loss_pct": unmitigated_loss_pct,
        "uav_metrics": {
            "canopy_coverage_pct": round(canopy_coverage, 2),
            "necrotic_foliar_ratio": round(necrotic_ratio, 3),
            "synthetic_spatial_descriptor": [round(x, 4) for x in uav_features]
        },
        "time_series": records
    }


def generate_dataset(num_samples: int = 5200, output_path: str = "data/cotton_wilt_dataset.json") -> List[Dict[str, Any]]:
    """
    Generate balanced dataset of 5,000+ parcels with representative class splits:
    50% Healthy, 22% Early Wilt, 18% Moderate Wilt, 10% Severe Wilt.
    """
    random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating {num_samples} multi-spectral satellite & UAV time-series records...")
    dataset = []

    # Distribution weights
    class_distribution = [
        (0, int(num_samples * 0.50)),  # Healthy
        (1, int(num_samples * 0.22)),  # Early-stage pre-visual (~12 days lead)
        (2, int(num_samples * 0.18)),  # Moderate wilt
        (3, num_samples - int(num_samples * 0.50) - int(num_samples * 0.22) - int(num_samples * 0.18)), # Severe wilt
    ]

    parcel_idx = 1
    for w_class, count in class_distribution:
        for _ in range(count):
            parcel = generate_single_parcel(parcel_idx, w_class)
            dataset.append(parcel)
            parcel_idx += 1

    # Shuffle dataset
    random.shuffle(dataset)

    # Save to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Successfully generated and cached {len(dataset)} records at {output_path}")
    return dataset


if __name__ == "__main__":
    generate_dataset(5200, "/Users/sharath8/.gemini/antigravity/scratch/cotton-verticillium-wilt-detection/data/cotton_wilt_dataset.json")
