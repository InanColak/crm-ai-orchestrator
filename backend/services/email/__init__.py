"""
Email Services Module (Phase 4.3)
=================================
Email delivery and context building services for Email Copilot.

This module provides:
- EmailDeliveryAdapter: Abstract base class for email delivery
- HubSpotEmailAdapter: HubSpot email API adapter (MVP)
- EmailContextBuilder: Context aggregation for email generation (RAG-ready)
"""

from backend.services.email.adapters import (
    EmailDeliveryAdapter,
    HubSpotEmailAdapter,
    get_email_adapter,
)
from backend.services.email.context_builder import (
    EmailContextBuilder,
    get_email_context_builder,
)

__all__ = [
    # Adapters
    "EmailDeliveryAdapter",
    "HubSpotEmailAdapter",
    "get_email_adapter",
    # Context Builder
    "EmailContextBuilder",
    "get_email_context_builder",
]
