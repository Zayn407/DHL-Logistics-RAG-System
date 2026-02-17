"""Utility functions."""

import logging
import requests
from typing import Dict, Optional


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def check_ollama_status(base_url: str = "http://localhost:11434") -> Dict:
    """Check if Ollama server is running.

    Args:
        base_url: Ollama server URL

    Returns:
        Dict with status and available models
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)

        if response.status_code == 200:
            models = response.json()
            return {
                "status": "running",
                "models": [m["name"] for m in models.get("models", [])]
            }
        else:
            return {"status": "error", "message": "Server not responding"}

    except requests.exceptions.ConnectionError:
        return {
            "status": "not_running",
            "message": "Ollama is not running. Start with: ollama serve"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def validate_tracking_id(tracking_id: str) -> bool:
    """Validate DHL tracking ID format.

    Args:
        tracking_id: Tracking ID to validate

    Returns:
        True if valid format
    """
    if not tracking_id:
        return False

    # Simple validation: starts with DHL and has digits
    return tracking_id.upper().startswith("DHL") and len(tracking_id) >= 4


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency string.

    Args:
        amount: Amount to format
        currency: Currency code

    Returns:
        Formatted currency string
    """
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, currency + " ")
    return f"{symbol}{amount:.2f}"
