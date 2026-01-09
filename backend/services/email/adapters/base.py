"""
Email Delivery Adapter Base (Phase 4.3)
=======================================
Abstract base class for email delivery adapters.

All email providers (HubSpot, Salesforce, SMTP) implement this interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from backend.app.schemas.email import (
    EmailDeliveryPayload,
    EmailDeliveryResult,
    DeliveryProvider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class EmailAdapterError(Exception):
    """Base exception for email adapter errors."""

    def __init__(
        self,
        message: str,
        provider: DeliveryProvider | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.provider = provider
        self.details = details or {}
        super().__init__(message)


class EmailAuthenticationError(EmailAdapterError):
    """Authentication failed with email provider."""
    pass


class EmailRateLimitError(EmailAdapterError):
    """Rate limit exceeded for email provider."""

    def __init__(
        self,
        message: str,
        provider: DeliveryProvider | None = None,
        retry_after: int | None = None,
    ):
        super().__init__(message, provider)
        self.retry_after = retry_after


class EmailDeliveryError(EmailAdapterError):
    """Email delivery failed."""

    def __init__(
        self,
        message: str,
        provider: DeliveryProvider | None = None,
        error_code: str | None = None,
        retry_possible: bool = False,
    ):
        super().__init__(message, provider)
        self.error_code = error_code
        self.retry_possible = retry_possible


class EmailValidationError(EmailAdapterError):
    """Email payload validation failed."""
    pass


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================


class EmailDeliveryAdapter(ABC):
    """
    Abstract base class for email delivery adapters.

    All email providers must implement:
    - send(): Send an email
    - health_check(): Verify connectivity
    - is_configured: Property to check if adapter is ready

    Optionally implement:
    - get_thread(): Get email thread
    - get_tracking(): Get email tracking data
    """

    @property
    @abstractmethod
    def provider(self) -> DeliveryProvider:
        """Return the provider identifier."""
        pass

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the adapter is properly configured."""
        pass

    @abstractmethod
    async def send(self, payload: EmailDeliveryPayload) -> EmailDeliveryResult:
        """
        Send an email.

        Args:
            payload: Email delivery payload with all required data

        Returns:
            EmailDeliveryResult: Result of the delivery attempt

        Raises:
            EmailAuthenticationError: If authentication fails
            EmailRateLimitError: If rate limit is exceeded
            EmailDeliveryError: If delivery fails
            EmailValidationError: If payload is invalid
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the adapter is healthy and can send emails.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass

    async def validate_payload(self, payload: EmailDeliveryPayload) -> list[str]:
        """
        Validate email payload before sending.

        Args:
            payload: Email delivery payload

        Returns:
            list[str]: List of validation errors (empty if valid)
        """
        errors = []

        # Basic validations
        if not payload.to_email:
            errors.append("Recipient email is required")
        elif "@" not in payload.to_email:
            errors.append("Invalid recipient email format")

        if not payload.subject:
            errors.append("Email subject is required")
        elif len(payload.subject) > 200:
            errors.append("Email subject too long (max 200 chars)")

        if not payload.body_html and not payload.body_plain:
            errors.append("Email body is required (HTML or plain text)")

        return errors

    def _create_result(
        self,
        success: bool,
        message_id: str | None = None,
        thread_id: str | None = None,
        crm_activity_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        retry_possible: bool = False,
    ) -> EmailDeliveryResult:
        """
        Create a standardized delivery result.

        Args:
            success: Whether delivery was successful
            message_id: Provider's message ID
            thread_id: Email thread ID
            crm_activity_id: CRM activity ID if logged
            error_code: Error code if failed
            error_message: Error message if failed
            retry_possible: Whether retry is possible

        Returns:
            EmailDeliveryResult: Standardized result
        """
        return EmailDeliveryResult(
            success=success,
            provider=self.provider,
            message_id=message_id,
            thread_id=thread_id,
            sent_at=datetime.now(timezone.utc).isoformat() if success else None,
            crm_activity_id=crm_activity_id,
            error_code=error_code,
            error_message=error_message,
            retry_possible=retry_possible,
        )

    # Optional methods for enhanced functionality

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """
        Get email thread by ID (optional).

        Args:
            thread_id: Email thread ID

        Returns:
            dict: Thread data or None if not supported
        """
        return None

    async def get_tracking(self, message_id: str) -> dict[str, Any] | None:
        """
        Get email tracking data (opens, clicks) (optional).

        Args:
            message_id: Email message ID

        Returns:
            dict: Tracking data or None if not supported
        """
        return None


# =============================================================================
# MOCK ADAPTER (for testing)
# =============================================================================


class MockEmailAdapter(EmailDeliveryAdapter):
    """
    Mock email adapter for testing.

    Does not actually send emails - logs and returns success.
    """

    def __init__(self):
        self._sent_emails: list[EmailDeliveryPayload] = []
        self._should_fail = False
        self._fail_error: str | None = None

    @property
    def provider(self) -> DeliveryProvider:
        return DeliveryProvider.HUBSPOT  # Pretend to be HubSpot

    @property
    def is_configured(self) -> bool:
        return True

    def set_should_fail(self, should_fail: bool, error: str | None = None):
        """Configure mock to fail on send."""
        self._should_fail = should_fail
        self._fail_error = error

    def get_sent_emails(self) -> list[EmailDeliveryPayload]:
        """Get list of emails that were 'sent'."""
        return self._sent_emails.copy()

    def clear_sent_emails(self):
        """Clear sent emails list."""
        self._sent_emails.clear()

    async def send(self, payload: EmailDeliveryPayload) -> EmailDeliveryResult:
        """Mock send - stores payload and returns success."""
        if self._should_fail:
            raise EmailDeliveryError(
                self._fail_error or "Mock delivery failure",
                provider=self.provider,
                retry_possible=True,
            )

        # Validate first
        errors = await self.validate_payload(payload)
        if errors:
            raise EmailValidationError(
                f"Validation failed: {', '.join(errors)}",
                provider=self.provider,
            )

        # Store the email
        self._sent_emails.append(payload)
        logger.info(f"[MockEmailAdapter] Email 'sent' to {payload.to_email}")

        return self._create_result(
            success=True,
            message_id=f"mock-msg-{len(self._sent_emails)}",
            thread_id=f"mock-thread-{len(self._sent_emails)}",
            crm_activity_id=f"mock-activity-{len(self._sent_emails)}",
        )

    async def health_check(self) -> bool:
        """Mock health check - always healthy."""
        return True
