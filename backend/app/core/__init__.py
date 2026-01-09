"""
Core application modules.
"""

from backend.app.core.config import Settings, get_settings, settings
from backend.app.core.dependencies import (
    # Settings
    get_settings_dep,
    SettingsDep,
    # Supabase
    get_supabase,
    get_supabase_admin,
    SupabaseDep,
    SupabaseAdminDep,
    SupabaseClientManager,
    # HTTP
    get_http_client,
    HTTPClientDep,
    HTTPClientManager,
    # Client Context
    ClientContext,
    get_client_context,
    ClientContextDep,
    # Rate Limiting
    RateLimiter,
    check_rate_limit,
    RateLimitDep,
    # Lifespan
    lifespan_manager,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "settings",
    # Settings Dep
    "get_settings_dep",
    "SettingsDep",
    # Supabase
    "get_supabase",
    "get_supabase_admin",
    "SupabaseDep",
    "SupabaseAdminDep",
    "SupabaseClientManager",
    # HTTP
    "get_http_client",
    "HTTPClientDep",
    "HTTPClientManager",
    # Client Context
    "ClientContext",
    "get_client_context",
    "ClientContextDep",
    # Rate Limiting
    "RateLimiter",
    "check_rate_limit",
    "RateLimitDep",
    # Lifespan
    "lifespan_manager",
]
