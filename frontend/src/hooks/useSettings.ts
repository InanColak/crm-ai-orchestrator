'use client';

import { useState, useEffect, useCallback } from 'react';
import { settingsApi, ApiError } from '@/services/api';
import type { SettingsSection, SettingsHealthResponse } from '@/services/api';

interface UseSettingsReturn {
  sections: SettingsSection[];
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

/**
 * Hook to fetch settings status from backend
 */
export function useSettings(): UseSettingsReturn {
  const [sections, setSections] = useState<SettingsSection[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSettings = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await settingsApi.getStatus();
      setSections(response.sections);
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : 'Failed to fetch settings';
      setError(message);
      console.error('[useSettings] Error:', err);

      // Set demo data on error for development
      setSections(getDemoSections());
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  return {
    sections,
    isLoading,
    error,
    refetch: fetchSettings,
  };
}

interface UseSettingsHealthReturn {
  health: SettingsHealthResponse | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

/**
 * Hook to fetch settings health check
 */
export function useSettingsHealth(): UseSettingsHealthReturn {
  const [health, setHealth] = useState<SettingsHealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealth = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await settingsApi.getHealth();
      setHealth(response);
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : 'Failed to fetch settings health';
      setError(message);
      console.error('[useSettingsHealth] Error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  return {
    health,
    isLoading,
    error,
    refetch: fetchHealth,
  };
}

/**
 * Demo data for when backend is unavailable
 */
function getDemoSections(): SettingsSection[] {
  return [
    {
      id: 'integrations',
      title: 'Integrations',
      description: 'Connect your CRM and other tools',
      icon: 'Link2',
      items: [
        { name: 'HubSpot', status: 'not_configured', status_color: 'gray', details: null },
        { name: 'Salesforce', status: 'not_configured', status_color: 'gray', details: null },
        { name: 'Supabase', status: 'not_configured', status_color: 'gray', details: null },
      ],
    },
    {
      id: 'api_keys',
      title: 'API Keys',
      description: 'Manage your API credentials',
      icon: 'Key',
      items: [
        { name: 'Anthropic API', status: 'not_configured', status_color: 'gray', details: null },
        { name: 'OpenAI API', status: 'not_configured', status_color: 'gray', details: null },
        { name: 'Tavily API', status: 'not_configured', status_color: 'gray', details: null },
        { name: 'LangSmith', status: 'not_configured', status_color: 'gray', details: null },
      ],
    },
    {
      id: 'security',
      title: 'Security',
      description: 'Encryption and security settings',
      icon: 'Shield',
      items: [
        { name: 'CRM Encryption', status: 'not_configured', status_color: 'gray', details: null },
      ],
    },
    {
      id: 'environment',
      title: 'Environment',
      description: 'Application environment settings',
      icon: 'Settings',
      items: [
        { name: 'Environment', status: 'development', status_color: 'yellow', details: null },
        { name: 'Debug Mode', status: 'enabled', status_color: 'yellow', details: null },
        { name: 'LLM Provider', status: 'none', status_color: 'gray', details: null },
      ],
    },
  ];
}
