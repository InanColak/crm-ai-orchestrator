"""
HubSpot Email Adapter (Phase 4.3)
=================================
Email delivery adapter for HubSpot Sales Email API.

Uses HubSpot's engagement API to:
- Send emails from HubSpot
- Associate emails with contacts, deals, companies
- Track opens and clicks
- Log to CRM timeline
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from backend.app.schemas.email import (
    EmailDeliveryPayload,
    EmailDeliveryResult,
    DeliveryProvider,
)
from backend.services.email.adapters.base import (
    EmailDeliveryAdapter,
    EmailAdapterError,
    EmailAuthenticationError,
    EmailRateLimitError,
    EmailDeliveryError,
    EmailValidationError,
)
from backend.app.core.config import Settings

logger = logging.getLogger(__name__)


class HubSpotEmailAdapter(EmailDeliveryAdapter):
    """
    HubSpot email delivery adapter.

    Sends emails via HubSpot's Sales Email API and logs
    them as engagements on the CRM timeline.

    Note: In production, this would use the HubSpot SDK.
    For MVP, we implement the interface and mock the actual API calls.
    """

    def __init__(self, access_token: str | None = None):
        """
        Initialize HubSpot email adapter.

        Args:
            access_token: HubSpot API access token (optional, uses config if not provided)
        """
        self._settings = Settings()
        self._access_token = access_token or getattr(
            self._settings, "hubspot_access_token", None
        )
        self._api_base = "https://api.hubapi.com"

    @property
    def provider(self) -> DeliveryProvider:
        return DeliveryProvider.HUBSPOT

    @property
    def is_configured(self) -> bool:
        """Check if HubSpot API is configured."""
        return bool(self._access_token)

    async def send(self, payload: EmailDeliveryPayload) -> EmailDeliveryResult:
        """
        Send email via HubSpot.

        Creates an email engagement in HubSpot associated with
        the relevant contact, deal, and company.

        Args:
            payload: Email delivery payload

        Returns:
            EmailDeliveryResult: Result with HubSpot engagement ID

        Raises:
            EmailAuthenticationError: If API token is invalid
            EmailRateLimitError: If HubSpot rate limit exceeded
            EmailDeliveryError: If email send fails
        """
        if not self.is_configured:
            raise EmailAuthenticationError(
                "HubSpot API token not configured",
                provider=self.provider,
            )

        # Validate payload
        errors = await self.validate_payload(payload)
        if errors:
            raise EmailValidationError(
                f"Validation failed: {', '.join(errors)}",
                provider=self.provider,
            )

        try:
            # Build HubSpot engagement payload
            engagement_data = self._build_engagement_payload(payload)

            # In production: Make actual API call to HubSpot
            # response = await self._hubspot_client.crm.engagements.create(engagement_data)

            # For MVP: Simulate successful send
            logger.info(
                f"[HubSpotEmailAdapter] Would send email to {payload.to_email} "
                f"with subject '{payload.subject[:50]}...'"
            )

            # Simulate HubSpot response
            engagement_id = f"hs-eng-{datetime.now(timezone.utc).timestamp()}"
            message_id = f"hs-msg-{datetime.now(timezone.utc).timestamp()}"

            return self._create_result(
                success=True,
                message_id=message_id,
                thread_id=None,  # HubSpot doesn't provide thread ID in same way
                crm_activity_id=engagement_id,
            )

        except EmailAdapterError:
            # Re-raise our own errors
            raise

        except Exception as e:
            # Wrap unexpected errors
            logger.exception(f"[HubSpotEmailAdapter] Unexpected error: {e}")
            raise EmailDeliveryError(
                f"Failed to send email via HubSpot: {str(e)}",
                provider=self.provider,
                retry_possible=True,
            )

    async def health_check(self) -> bool:
        """
        Check HubSpot API connectivity.

        Returns:
            bool: True if API is accessible
        """
        if not self.is_configured:
            return False

        try:
            # In production: Make a simple API call to verify token
            # response = await self._hubspot_client.crm.contacts.get_all(limit=1)
            # return True

            # For MVP: Return True if configured
            return True

        except Exception as e:
            logger.warning(f"[HubSpotEmailAdapter] Health check failed: {e}")
            return False

    def _build_engagement_payload(self, payload: EmailDeliveryPayload) -> dict[str, Any]:
        """
        Build HubSpot engagement API payload.

        Args:
            payload: Our email payload

        Returns:
            dict: HubSpot engagement creation payload
        """
        # Current timestamp in milliseconds (HubSpot format)
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Build associations
        associations = {}
        if payload.contact_id:
            associations["contactIds"] = [payload.contact_id]
        if payload.deal_id:
            associations["dealIds"] = [payload.deal_id]
        if payload.company_id:
            associations["companyIds"] = [payload.company_id]

        return {
            "engagement": {
                "active": True,
                "type": "EMAIL",
                "timestamp": timestamp_ms,
            },
            "associations": associations,
            "metadata": {
                "from": {
                    "email": payload.from_email or "noreply@example.com",
                    "firstName": payload.from_name.split()[0] if payload.from_name else None,
                    "lastName": " ".join(payload.from_name.split()[1:]) if payload.from_name and len(payload.from_name.split()) > 1 else None,
                },
                "to": [
                    {
                        "email": payload.to_email,
                        "firstName": payload.to_name.split()[0] if payload.to_name else None,
                        "lastName": " ".join(payload.to_name.split()[1:]) if payload.to_name and len(payload.to_name.split()) > 1 else None,
                    }
                ],
                "subject": payload.subject,
                "html": payload.body_html,
                "text": payload.body_plain,
            },
        }

    async def validate_payload(self, payload: EmailDeliveryPayload) -> list[str]:
        """
        HubSpot-specific payload validation.

        Args:
            payload: Email delivery payload

        Returns:
            list[str]: Validation errors
        """
        # Start with base validation
        errors = await super().validate_payload(payload)

        # HubSpot-specific validations
        if not payload.contact_id and not payload.deal_id and not payload.company_id:
            errors.append(
                "HubSpot requires at least one association (contact_id, deal_id, or company_id)"
            )

        # HubSpot subject length limit
        if payload.subject and len(payload.subject) > 255:
            errors.append("HubSpot subject line max is 255 characters")

        return errors

    async def get_tracking(self, engagement_id: str) -> dict[str, Any] | None:
        """
        Get email tracking data from HubSpot.

        Args:
            engagement_id: HubSpot engagement ID

        Returns:
            dict: Tracking data (opens, clicks) or None
        """
        if not self.is_configured:
            return None

        try:
            # In production: Fetch engagement details
            # response = await self._hubspot_client.crm.engagements.get(engagement_id)
            # return {
            #     "opens": response.metadata.get("opens", 0),
            #     "clicks": response.metadata.get("clicks", 0),
            #     "status": response.engagement.get("active", True),
            # }

            # For MVP: Return mock tracking
            return {
                "opens": 0,
                "clicks": 0,
                "status": "sent",
            }

        except Exception as e:
            logger.warning(f"[HubSpotEmailAdapter] Failed to get tracking: {e}")
            return None
