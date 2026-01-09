"""
Encryption Service
==================
AES-256-GCM encryption for sensitive data (OAuth tokens, credentials).
Provides secure encryption with authentication and key rotation support.
"""

import base64
import os
import secrets
from dataclasses import dataclass
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from backend.app.core.config import get_settings


# =============================================================================
# EXCEPTIONS
# =============================================================================

class EncryptionError(Exception):
    """Base exception for encryption operations."""
    pass


class DecryptionError(EncryptionError):
    """Failed to decrypt data (wrong key, corrupted data, etc.)."""
    pass


class KeyNotConfiguredError(EncryptionError):
    """Encryption key is not configured."""
    pass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EncryptedData:
    """Container for encrypted data with metadata."""
    ciphertext: bytes
    iv: bytes  # Initialization vector (nonce)
    key_version: int

    def to_storage_format(self) -> dict:
        """Convert to format suitable for database storage."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode("utf-8"),
            "iv": base64.b64encode(self.iv).decode("utf-8"),
            "key_version": self.key_version
        }

    @classmethod
    def from_storage_format(cls, data: dict) -> "EncryptedData":
        """Reconstruct from database storage format."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            iv=base64.b64decode(data["iv"]),
            key_version=data.get("key_version", 1)
        )


# =============================================================================
# ENCRYPTION SERVICE
# =============================================================================

class EncryptionService:
    """
    AES-256-GCM Encryption Service.

    Features:
    - AES-256-GCM: Strong encryption with authentication
    - Unique IV per encryption: Prevents pattern analysis
    - Key version tracking: Supports key rotation
    - Development mode: Works without key (plain base64)

    Usage:
        >>> service = EncryptionService()
        >>> encrypted = service.encrypt("my-secret-token")
        >>> decrypted = service.decrypt(encrypted.ciphertext, encrypted.iv)
        >>> assert decrypted == "my-secret-token"
    """

    # AES-GCM nonce size (96 bits = 12 bytes is recommended)
    NONCE_SIZE = 12

    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize encryption service.

        Args:
            encryption_key: Base64-encoded 32-byte key.
                           If None, uses CRM_ENCRYPTION_KEY from settings.
        """
        self._settings = get_settings()
        self._key: Optional[bytes] = None
        self._aesgcm: Optional[AESGCM] = None

        # Get key from parameter or settings
        key_str = encryption_key or self._settings.crm_encryption_key

        if key_str:
            self._key = base64.b64decode(key_str)
            if len(self._key) != 32:
                raise EncryptionError(
                    f"Encryption key must be 32 bytes, got {len(self._key)}"
                )
            self._aesgcm = AESGCM(self._key)

    @property
    def is_configured(self) -> bool:
        """Check if encryption is properly configured."""
        return self._key is not None

    @property
    def key_version(self) -> int:
        """Get current encryption key version."""
        return self._settings.encryption_key_version

    def encrypt(self, plaintext: str) -> EncryptedData:
        """
        Encrypt a string using AES-256-GCM.

        Args:
            plaintext: String to encrypt

        Returns:
            EncryptedData containing ciphertext, IV, and key version

        Raises:
            KeyNotConfiguredError: If encryption key is not set
            EncryptionError: If encryption fails
        """
        if not self.is_configured:
            # Development mode: use base64 encoding (NOT SECURE)
            if self._settings.is_development:
                return EncryptedData(
                    ciphertext=base64.b64encode(plaintext.encode("utf-8")),
                    iv=b"dev-mode-iv",  # Placeholder
                    key_version=0  # Version 0 = development mode
                )
            raise KeyNotConfiguredError(
                "CRM_ENCRYPTION_KEY is required for encryption"
            )

        try:
            # Generate random IV (nonce) for this encryption
            iv = secrets.token_bytes(self.NONCE_SIZE)

            # Encrypt with AES-GCM
            ciphertext = self._aesgcm.encrypt(
                iv,
                plaintext.encode("utf-8"),
                None  # No additional authenticated data
            )

            return EncryptedData(
                ciphertext=ciphertext,
                iv=iv,
                key_version=self.key_version
            )
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}")

    def decrypt(
        self,
        ciphertext: bytes,
        iv: bytes,
        key_version: Optional[int] = None
    ) -> str:
        """
        Decrypt AES-256-GCM encrypted data.

        Args:
            ciphertext: Encrypted bytes
            iv: Initialization vector used during encryption
            key_version: Version of key used for encryption

        Returns:
            Decrypted plaintext string

        Raises:
            KeyNotConfiguredError: If encryption key is not set
            DecryptionError: If decryption fails
        """
        # Development mode check
        if key_version == 0:
            try:
                return base64.b64decode(ciphertext).decode("utf-8")
            except Exception as e:
                raise DecryptionError(f"Development mode decryption failed: {e}")

        if not self.is_configured:
            raise KeyNotConfiguredError(
                "CRM_ENCRYPTION_KEY is required for decryption"
            )

        # Key version mismatch warning
        if key_version and key_version != self.key_version:
            # In production, you'd look up the old key from a key store
            # For now, we'll attempt with current key
            import logging
            logging.warning(
                f"Key version mismatch: data encrypted with v{key_version}, "
                f"current key is v{self.key_version}"
            )

        try:
            plaintext_bytes = self._aesgcm.decrypt(iv, ciphertext, None)
            return plaintext_bytes.decode("utf-8")
        except Exception as e:
            raise DecryptionError(f"Decryption failed: {e}")

    def decrypt_from_storage(self, encrypted_data: EncryptedData) -> str:
        """
        Decrypt from EncryptedData object.

        Args:
            encrypted_data: EncryptedData container

        Returns:
            Decrypted plaintext string
        """
        return self.decrypt(
            encrypted_data.ciphertext,
            encrypted_data.iv,
            encrypted_data.key_version
        )

    def re_encrypt(
        self,
        ciphertext: bytes,
        iv: bytes,
        old_key_version: int
    ) -> EncryptedData:
        """
        Re-encrypt data with current key (for key rotation).

        Args:
            ciphertext: Data encrypted with old key
            iv: IV from old encryption
            old_key_version: Version of old key

        Returns:
            Newly encrypted data with current key
        """
        # First decrypt with appropriate key
        plaintext = self.decrypt(ciphertext, iv, old_key_version)
        # Then encrypt with current key
        return self.encrypt(plaintext)


# =============================================================================
# SINGLETON & DEPENDENCY INJECTION
# =============================================================================

_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """
    Get singleton EncryptionService instance.

    Returns:
        EncryptionService instance
    """
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service


def reset_encryption_service() -> None:
    """Reset singleton (for testing)."""
    global _encryption_service
    _encryption_service = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_encryption_key() -> str:
    """
    Generate a new random encryption key.

    Returns:
        Base64-encoded 32-byte key suitable for CRM_ENCRYPTION_KEY

    Usage:
        >>> key = generate_encryption_key()
        >>> print(f"CRM_ENCRYPTION_KEY={key}")
    """
    key_bytes = secrets.token_bytes(32)
    return base64.b64encode(key_bytes).decode("utf-8")


def encrypt_token(token: str) -> tuple[bytes, bytes, int]:
    """
    Convenience function to encrypt a token.

    Args:
        token: The token string to encrypt

    Returns:
        Tuple of (ciphertext, iv, key_version)
    """
    service = get_encryption_service()
    encrypted = service.encrypt(token)
    return encrypted.ciphertext, encrypted.iv, encrypted.key_version


def decrypt_token(ciphertext: bytes, iv: bytes, key_version: int = 1) -> str:
    """
    Convenience function to decrypt a token.

    Args:
        ciphertext: Encrypted token bytes
        iv: Initialization vector
        key_version: Key version used for encryption

    Returns:
        Decrypted token string
    """
    service = get_encryption_service()
    return service.decrypt(ciphertext, iv, key_version)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "EncryptionError",
    "DecryptionError",
    "KeyNotConfiguredError",
    # Classes
    "EncryptedData",
    "EncryptionService",
    # DI
    "get_encryption_service",
    "reset_encryption_service",
    # Utilities
    "generate_encryption_key",
    "encrypt_token",
    "decrypt_token",
]
