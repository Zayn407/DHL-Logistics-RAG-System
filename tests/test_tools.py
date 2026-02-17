"""Tests for tools."""

import pytest
from dhl_logistics_rag.tools.shipment_tracker import ShipmentTracker
from dhl_logistics_rag.tools.cost_estimator import CostEstimator


class TestShipmentTracker:
    """Tests for ShipmentTracker."""

    def setup_method(self):
        self.tracker = ShipmentTracker()

    def test_track_existing_shipment(self):
        """Test tracking an existing shipment."""
        shipment = self.tracker.track("DHL001")
        assert shipment is not None
        assert shipment.tracking_id == "DHL001"
        assert shipment.origin == "New York"

    def test_track_nonexistent_shipment(self):
        """Test tracking a non-existent shipment."""
        shipment = self.tracker.track("INVALID123")
        assert shipment is None

    def test_check_status_format(self):
        """Test status string format."""
        status = self.tracker.check_status("DHL001")
        assert "DHL001" in status
        assert "New York" in status
        assert "London" in status

    def test_case_insensitive_tracking(self):
        """Test that tracking ID is case insensitive."""
        shipment1 = self.tracker.track("DHL001")
        shipment2 = self.tracker.track("dhl001")
        assert shipment1 == shipment2


class TestCostEstimator:
    """Tests for CostEstimator."""

    def setup_method(self):
        self.estimator = CostEstimator(
            base_rate=25.0,
            weight_rate=5.0,
            international_surcharge=15.0
        )

    def test_domestic_shipping(self):
        """Test domestic shipping cost."""
        breakdown = self.estimator.estimate("New York", "New York", 2.0)
        assert breakdown.international_surcharge == 0.0
        assert breakdown.total == 25.0 + (2.0 * 5.0)

    def test_international_shipping(self):
        """Test international shipping cost."""
        breakdown = self.estimator.estimate("New York", "London", 2.0)
        assert breakdown.international_surcharge == 15.0
        assert breakdown.total == 25.0 + (2.0 * 5.0) + 15.0

    def test_weight_calculation(self):
        """Test weight-based calculation."""
        breakdown = self.estimator.estimate("A", "B", 10.0)
        assert breakdown.weight_charge == 50.0  # 10 * 5

    def test_estimate_string_format(self):
        """Test string output format."""
        result = self.estimator.estimate_string("New York", "London", 5.0)
        assert "New York" in result
        assert "London" in result
        assert "$" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
