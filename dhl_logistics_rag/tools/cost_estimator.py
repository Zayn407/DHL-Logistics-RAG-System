"""Shipping cost estimation tool."""

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.tools import tool

from ..common.constants import (
    DEFAULT_BASE_RATE,
    DEFAULT_WEIGHT_RATE,
    DEFAULT_INTERNATIONAL_SURCHARGE
)

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Breakdown of shipping cost."""
    origin: str
    destination: str
    weight_kg: float
    base_rate: float
    weight_charge: float
    international_surcharge: float
    total: float

    def __str__(self) -> str:
        return f"""💰 Shipping Cost Estimate:
Route: {self.origin} → {self.destination}
Weight: {self.weight_kg} kg
Base Rate: ${self.base_rate:.2f}
Weight Charge: ${self.weight_charge:.2f}
International: ${self.international_surcharge:.2f}
━━━━━━━━━━━━━━━━━━
Total: ${self.total:.2f}"""


class CostEstimator:
    """Service for estimating DHL shipping costs."""

    def __init__(
        self,
        base_rate: float = DEFAULT_BASE_RATE,
        weight_rate: float = DEFAULT_WEIGHT_RATE,
        international_surcharge: float = DEFAULT_INTERNATIONAL_SURCHARGE
    ):
        self.base_rate = base_rate
        self.weight_rate = weight_rate
        self.international_surcharge = international_surcharge

    def estimate(
        self,
        origin: str,
        destination: str,
        weight_kg: float
    ) -> CostBreakdown:
        """Estimate shipping cost.

        Args:
            origin: Origin city/country
            destination: Destination city/country
            weight_kg: Package weight in kilograms

        Returns:
            CostBreakdown with detailed pricing
        """
        weight_charge = weight_kg * self.weight_rate

        # Apply international surcharge if different locations
        is_international = origin.lower() != destination.lower()
        surcharge = self.international_surcharge if is_international else 0.0

        total = self.base_rate + weight_charge + surcharge

        breakdown = CostBreakdown(
            origin=origin,
            destination=destination,
            weight_kg=weight_kg,
            base_rate=self.base_rate,
            weight_charge=weight_charge,
            international_surcharge=surcharge,
            total=total
        )

        logger.info(f"Estimated cost: {origin} → {destination}, {weight_kg}kg = ${total:.2f}")
        return breakdown

    def estimate_string(
        self,
        origin: str,
        destination: str,
        weight_kg: float
    ) -> str:
        """Estimate shipping cost and return formatted string."""
        breakdown = self.estimate(origin, destination, weight_kg)
        return str(breakdown)

    def get_tool(self):
        """Get LangChain tool wrapper."""
        estimator = self

        @tool
        def estimate_shipping_cost(origin: str, destination: str, weight_kg: float) -> str:
            """Estimate DHL shipping cost based on origin, destination and weight."""
            return estimator.estimate_string(origin, destination, weight_kg)

        return estimate_shipping_cost
