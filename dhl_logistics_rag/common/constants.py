"""Constants used throughout the application."""

# Document sources
PDF_SOURCES = {
    "dhl_express_terms": "dhl_express_terms.pdf",
    "dhl_customs_guide": "dhl_customs_guide.pdf",
    "dhl_ecommerce_terms": "dhl_ecommerce_terms.pdf",
}

# Shipment statuses
class ShipmentStatus:
    IN_TRANSIT = "IN_TRANSIT"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    CUSTOMS_HOLD = "CUSTOMS_HOLD"
    EXCEPTION = "EXCEPTION"

# Evaluation categories
EVAL_CATEGORIES = [
    "prohibited_items",
    "liability",
    "claims",
    "customs",
    "delivery",
    "packaging",
    "privacy",
]

# Default rate configuration
DEFAULT_BASE_RATE = 25.0
DEFAULT_WEIGHT_RATE = 5.0
DEFAULT_INTERNATIONAL_SURCHARGE = 15.0
