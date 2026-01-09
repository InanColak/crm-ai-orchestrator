"""
Email Delivery Adapters (Phase 4.3)
===================================
Pluggable email delivery adapters for different providers.

MVP: HubSpot Email API
Future: Salesforce, SendGrid, direct SMTP
"""

from backend.services.email.adapters.base import (
    EmailDeliveryAdapter,
    EmailAdapterError,
    EmailAuthenticationError,
    EmailRateLimitError,
    EmailDeliveryError,
)
from backend.services.email.adapters.hubspot import HubSpotEmailAdapter

# Adapter registry
_adapter_registry: dict[str, type[EmailDeliveryAdapter]] = {
    "hubspot": HubSpotEmailAdapter,
}

# Singleton instances
_adapter_instances: dict[str, EmailDeliveryAdapter] = {}


def get_email_adapter(provider: str = "hubspot") -> EmailDeliveryAdapter:
    """
    Get or create an email adapter instance.

    Args:
        provider: Email provider name (hubspot, salesforce, etc.)

    Returns:
        EmailDeliveryAdapter: Adapter instance for the provider

    Raises:
        ValueError: If provider is not supported
    """
    provider = provider.lower()

    if provider not in _adapter_registry:
        supported = ", ".join(_adapter_registry.keys())
        raise ValueError(
            f"Unsupported email provider: {provider}. Supported: {supported}"
        )

    if provider not in _adapter_instances:
        adapter_class = _adapter_registry[provider]
        _adapter_instances[provider] = adapter_class()

    return _adapter_instances[provider]


def register_adapter(provider: str, adapter_class: type[EmailDeliveryAdapter]) -> None:
    """
    Register a custom email adapter.

    Args:
        provider: Provider name
        adapter_class: Adapter class to register
    """
    _adapter_registry[provider.lower()] = adapter_class


__all__ = [
    # Base
    "EmailDeliveryAdapter",
    "EmailAdapterError",
    "EmailAuthenticationError",
    "EmailRateLimitError",
    "EmailDeliveryError",
    # Adapters
    "HubSpotEmailAdapter",
    # Factory
    "get_email_adapter",
    "register_adapter",
]
