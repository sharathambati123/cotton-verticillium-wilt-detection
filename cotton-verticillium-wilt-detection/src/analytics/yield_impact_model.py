"""
Cotton Agronomic Yield Loss & Precision Economic Impact Model.
Quantifies yield preservation resulting from 12-day early detection of Verticillium wilt
and targeted variable-rate pesticide deployment.
"""

from typing import Dict, Any, List


class YieldImpactModel:
    """
    Simulates agricultural yield outcomes and economic returns based on
    Verticillium wilt intervention timing.
    """

    def __init__(
        self,
        base_yield_kg_ha: float = 2800.0,
        cotton_price_usd_kg: float = 1.85,
        cost_early_treatment_usd_ha: float = 85.0,
        cost_blanket_spray_usd_ha: float = 195.0
    ):
        self.base_yield_kg_ha = base_yield_kg_ha
        self.cotton_price_usd_kg = cotton_price_usd_kg
        self.cost_early_treatment = cost_early_treatment_usd_ha
        self.cost_blanket_spray = cost_blanket_spray_usd_ha

    def calculate_yield_scenarios(self, total_hectares: float = 500.0) -> Dict[str, Any]:
        """
        Compare three agricultural management regimes across total_hectares:
        1. Scenario A: Unmitigated Wilt (No detection / no effective treatment)
        2. Scenario B: Delayed Intervention (Standard visual scouting threshold @ Day 0)
        3. Scenario C: Early AI Intervention (Detected 12 days prior @ Day -12)
        """
        total_potential_yield_kg = total_hectares * self.base_yield_kg_ha

        # Yield Loss Percentages
        loss_unmitigated_pct = 35.0
        loss_delayed_pct = 26.0
        loss_early_pct = 7.0

        # Salvaged Yield Percentage
        # Net yield reduction preserved: 35.0% - 7.0% = 28.0%
        salvaged_pct = loss_unmitigated_pct - loss_early_pct

        # Yield Calculations (kg)
        yield_unmitigated = total_potential_yield_kg * (1.0 - loss_unmitigated_pct / 100.0)
        yield_delayed = total_potential_yield_kg * (1.0 - loss_delayed_pct / 100.0)
        yield_early = total_potential_yield_kg * (1.0 - loss_early_pct / 100.0)

        salvaged_kg = yield_early - yield_unmitigated
        salvaged_kg_per_ha = salvaged_kg / total_hectares

        # Revenue Calculations (USD)
        revenue_potential = total_potential_yield_kg * self.cotton_price_usd_kg
        revenue_unmitigated = yield_unmitigated * self.cotton_price_usd_kg
        revenue_delayed = yield_delayed * self.cotton_price_usd_kg
        revenue_early = yield_early * self.cotton_price_usd_kg

        gross_value_salvaged_usd = salvaged_kg * self.cotton_price_usd_kg
        gross_value_salvaged_per_ha = gross_value_salvaged_usd / total_hectares

        # Pesticide & Treatment Volume Savings
        # Standard visual scouting requires blanket curative chemical spraying across 100% of acreage
        # Precision early detection applies targeted variable-rate treatment to affected spots (~36% of acreage)
        acreage_treated_targeted_pct = 36.0
        pesticide_volume_saving_pct = 100.0 - acreage_treated_targeted_pct  # 64.0% reduction in chemical volume

        targeted_treatment_cost_total = (total_hectares * (acreage_treated_targeted_pct / 100.0)) * self.cost_early_treatment
        blanket_spray_cost_total = total_hectares * self.cost_blanket_spray
        chemical_cost_savings_usd = blanket_spray_cost_total - targeted_treatment_cost_total

        net_economic_benefit_usd = gross_value_salvaged_usd + chemical_cost_savings_usd

        return {
            "farm_area_hectares": total_hectares,
            "base_yield_kg_ha": self.base_yield_kg_ha,
            "cotton_price_per_kg_usd": self.cotton_price_usd_kg,
            "scenarios": {
                "unmitigated_wilt": {
                    "yield_loss_pct": loss_unmitigated_pct,
                    "harvested_yield_kg_ha": round(yield_unmitigated / total_hectares, 1),
                    "total_revenue_usd": round(revenue_unmitigated, 2)
                },
                "standard_visual_scouting": {
                    "yield_loss_pct": loss_delayed_pct,
                    "harvested_yield_kg_ha": round(yield_delayed / total_hectares, 1),
                    "total_revenue_usd": round(revenue_delayed, 2)
                },
                "early_ai_intervention_day_minus_12": {
                    "yield_loss_pct": loss_early_pct,
                    "harvested_yield_kg_ha": round(yield_early / total_hectares, 1),
                    "total_revenue_usd": round(revenue_early, 2)
                }
            },
            "yield_impact_summary": {
                "projected_crop_yield_loss_reduction_pct": salvaged_pct,
                "salvaged_cotton_lint_kg_per_ha": round(salvaged_kg_per_ha, 1),
                "total_salvaged_cotton_kg": round(salvaged_kg, 1),
                "gross_revenue_preserved_usd": round(gross_value_salvaged_usd, 2),
                "gross_revenue_preserved_per_ha_usd": round(gross_value_salvaged_per_ha, 2)
            },
            "pesticide_optimization": {
                "targeted_deployment_area_pct": acreage_treated_targeted_pct,
                "chemical_volume_reduction_pct": pesticide_volume_saving_pct,
                "chemical_cost_savings_usd": round(chemical_cost_savings_usd, 2)
            },
            "net_roi_total_usd": round(net_economic_benefit_usd, 2)
        }
