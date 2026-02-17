"""Shipment tracking tool."""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from langchain_core.tools import tool

from ..common.constants import ShipmentStatus

logger = logging.getLogger(__name__)


@dataclass
class Shipment:
    """Shipment data model."""
    tracking_id: str
    origin: str
    destination: str
    status: str
    weight_kg: float
    ship_date: datetime
    eta: datetime

    def to_dict(self) -> Dict:
        return {
            "tracking_id": self.tracking_id,
            "origin": self.origin,
            "destination": self.destination,
            "status": self.status,
            "weight_kg": self.weight_kg,
            "ship_date": self.ship_date.strftime("%Y-%m-%d"),
            "eta": self.eta.strftime("%Y-%m-%d")
        }

    def __str__(self) -> str:
        return f"""📦 Shipment: {self.tracking_id}
Route: {self.origin} → {self.destination}
Status: {self.status}
Weight: {self.weight_kg} kg
Shipped: {self.ship_date.strftime('%Y-%m-%d')}
ETA: {self.eta.strftime('%Y-%m-%d')}"""


class ShipmentTracker:
    """Service for tracking DHL shipments."""

    def __init__(self):
        self._shipments: Dict[str, Shipment] = {}
        self._init_sample_data()

    def _init_sample_data(self):
        """Initialize with sample shipment data."""
        sample_data = [
            Shipment(
                tracking_id="DHL001",
                origin="New York",
                destination="London",
                status=ShipmentStatus.IN_TRANSIT,
                weight_kg=2.5,
                ship_date=datetime.now() - timedelta(days=2),
                eta=datetime.now() + timedelta(days=1)
            ),
            Shipment(
                tracking_id="DHL002",
                origin="Los Angeles",
                destination="Tokyo",
                status=ShipmentStatus.OUT_FOR_DELIVERY,
                weight_kg=5.0,
                ship_date=datetime.now() - timedelta(days=3),
                eta=datetime.now() + timedelta(days=2)
            ),
            Shipment(
                tracking_id="DHL003",
                origin="Chicago",
                destination="Berlin",
                status=ShipmentStatus.DELIVERED,
                weight_kg=1.2,
                ship_date=datetime.now() - timedelta(days=4),
                eta=datetime.now() + timedelta(days=3)
            ),
            Shipment(
                tracking_id="DHL004",
                origin="Houston",
                destination="Paris",
                status=ShipmentStatus.CUSTOMS_HOLD,
                weight_kg=8.3,
                ship_date=datetime.now() - timedelta(days=5),
                eta=datetime.now() + timedelta(days=4)
            ),
            Shipment(
                tracking_id="DHL005",
                origin="Miami",
                destination="Sydney",
                status=ShipmentStatus.EXCEPTION,
                weight_kg=3.7,
                ship_date=datetime.now() - timedelta(days=6),
                eta=datetime.now() + timedelta(days=5)
            ),
        ]

        for shipment in sample_data:
            self._shipments[shipment.tracking_id] = shipment

        logger.info(f"Initialized {len(self._shipments)} sample shipments")

    def track(self, tracking_id: str) -> Optional[Shipment]:
        """Get shipment by tracking ID.

        Args:
            tracking_id: DHL tracking number

        Returns:
            Shipment if found, None otherwise
        """
        return self._shipments.get(tracking_id.upper())

    def check_status(self, tracking_id: str) -> str:
        """Check shipment status and return formatted string.

        Args:
            tracking_id: DHL tracking number

        Returns:
            Formatted status string
        """
        shipment = self.track(tracking_id)

        if shipment is None:
            return f"Shipment {tracking_id} not found."

        return str(shipment)

    def get_all_shipments(self) -> List[Shipment]:
        """Get all shipments."""
        return list(self._shipments.values())

    def add_shipment(self, shipment: Shipment):
        """Add a new shipment."""
        self._shipments[shipment.tracking_id] = shipment
        logger.info(f"Added shipment: {shipment.tracking_id}")

    def update_status(self, tracking_id: str, status: str) -> bool:
        """Update shipment status.

        Args:
            tracking_id: DHL tracking number
            status: New status

        Returns:
            True if updated, False if not found
        """
        shipment = self.track(tracking_id)

        if shipment is None:
            return False

        shipment.status = status
        logger.info(f"Updated {tracking_id} status to {status}")
        return True

    def get_tool(self):
        """Get LangChain tool wrapper."""
        tracker = self

        @tool
        def check_shipment(tracking_id: str) -> str:
            """Check the status of a DHL shipment by tracking ID."""
            return tracker.check_status(tracking_id)

        return check_shipment
