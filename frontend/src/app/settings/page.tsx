'use client';

import { useState } from 'react';
import { Header } from '@/components/layout';
import { Card, CardHeader, Button } from '@/components/ui';
import { Link2, User, BarChart3, CreditCard, Loader2, AlertCircle, RefreshCw, ExternalLink } from 'lucide-react';
import { useSettings } from '@/hooks';

// Progress bar component
function ProgressBar({ value, max, label }: { value: number; max: number; label: string }) {
  const percentage = Math.min((value / max) * 100, 100);
  const isWarning = percentage > 80;
  const isCritical = percentage > 95;

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-[#D6D6D6]">{label}</span>
        <span className={`${isCritical ? 'text-red-400' : isWarning ? 'text-yellow-400' : 'text-[#9D9D9D]'}`}>
          {value.toLocaleString()} / {max.toLocaleString()}
        </span>
      </div>
      <div className="h-2 bg-[#414141] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${
            isCritical ? 'bg-red-500' : isWarning ? 'bg-yellow-500' : 'bg-[#00C0F0]'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="text-xs text-[#9D9D9D] text-right">{percentage.toFixed(0)}% used</div>
    </div>
  );
}

// Integration status component
function IntegrationItem({
  name,
  status,
  onAction
}: {
  name: string;
  status: 'connected' | 'not_configured';
  onAction: () => void;
}) {
  const isConnected = status === 'connected';

  return (
    <div className="flex items-center justify-between py-3 border-b border-[#414141] last:border-0">
      <div className="flex items-center gap-3">
        <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-[#9D9D9D]'}`} />
        <span className="text-sm text-[#D6D6D6]">{name}</span>
      </div>
      <div className="flex items-center gap-3">
        <span className={`text-sm ${isConnected ? 'text-green-400' : 'text-[#9D9D9D]'}`}>
          {isConnected ? 'Connected' : 'Not connected'}
        </span>
        <Button
          variant={isConnected ? 'secondary' : 'primary'}
          size="sm"
          onClick={onAction}
        >
          {isConnected ? 'Manage' : 'Connect'}
        </Button>
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const { sections, isLoading, error, refetch } = useSettings();

  // Extract integrations from sections
  const integrationsSection = sections.find(s => s.id === 'integrations');
  const integrations = integrationsSection?.items || [];

  // Demo usage data (will be replaced with real API)
  const usageData = {
    aiOperations: { used: 1234, limit: 10000 },
    documents: { used: 23, limit: 100 },
    searches: { used: 456, limit: 5000 },
  };

  const handleIntegrationAction = (name: string, isConnected: boolean) => {
    if (name === 'HubSpot') {
      if (isConnected) {
        // Show manage modal or redirect to HubSpot settings
        alert('HubSpot management coming soon. Contact support for changes.');
      } else {
        // Redirect to OAuth flow
        window.location.href = 'http://localhost:8080/api/v1/oauth/hubspot/authorize';
      }
    } else if (name === 'Salesforce') {
      alert('Salesforce integration coming soon.');
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen">
        <Header
          title="Settings"
          description="Configure your CRM AI Orchestrator"
        />
        <div className="p-6 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-[#00C0F0]" />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <Header
        title="Settings"
        description="Configure your CRM AI Orchestrator"
        actions={
          <Button
            variant="secondary"
            size="sm"
            onClick={() => refetch()}
            className="flex items-center gap-2"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </Button>
        }
      />

      <div className="p-6 space-y-6 max-w-4xl">
        {error && (
          <Card>
            <div className="flex items-center gap-3 text-yellow-400">
              <AlertCircle className="h-5 w-5" />
              <div>
                <p className="text-sm font-medium">Could not fetch settings from backend</p>
                <p className="text-xs text-[#9D9D9D]">Showing demo data. Error: {error}</p>
              </div>
            </div>
          </Card>
        )}

        {/* Integrations Section */}
        <Card>
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#414141]/50 shrink-0">
              <Link2 className="h-5 w-5 text-[#00C0F0]" />
            </div>
            <div className="flex-1">
              <CardHeader
                title="Integrations"
                description="Connect your CRM platforms"
              />
              <div className="mt-2">
                {integrations
                  .filter(item => item.name === 'HubSpot' || item.name === 'Salesforce')
                  .map((item) => (
                    <IntegrationItem
                      key={item.name}
                      name={item.name}
                      status={item.status === 'connected' ? 'connected' : 'not_configured'}
                      onAction={() => handleIntegrationAction(item.name, item.status === 'connected')}
                    />
                  ))}
                {integrations.filter(item => item.name === 'HubSpot' || item.name === 'Salesforce').length === 0 && (
                  <>
                    <IntegrationItem
                      name="HubSpot"
                      status="not_configured"
                      onAction={() => handleIntegrationAction('HubSpot', false)}
                    />
                    <IntegrationItem
                      name="Salesforce"
                      status="not_configured"
                      onAction={() => handleIntegrationAction('Salesforce', false)}
                    />
                  </>
                )}
              </div>
            </div>
          </div>
        </Card>

        {/* Usage Section */}
        <Card>
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#414141]/50 shrink-0">
              <BarChart3 className="h-5 w-5 text-[#00C0F0]" />
            </div>
            <div className="flex-1">
              <CardHeader
                title="Usage This Month"
                description="Track your AI operations and resource usage"
              />
              <div className="mt-4 space-y-6">
                <ProgressBar
                  value={usageData.aiOperations.used}
                  max={usageData.aiOperations.limit}
                  label="AI Operations"
                />
                <ProgressBar
                  value={usageData.documents.used}
                  max={usageData.documents.limit}
                  label="Documents"
                />
                <ProgressBar
                  value={usageData.searches.used}
                  max={usageData.searches.limit}
                  label="Semantic Searches"
                />
              </div>
              <div className="mt-4 pt-4 border-t border-[#414141]">
                <p className="text-xs text-[#9D9D9D]">
                  Usage resets on the 1st of each month. Need more capacity?{' '}
                  <a href="#" className="text-[#00C0F0] hover:underline">Contact us</a>
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Account Section */}
        <Card>
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#414141]/50 shrink-0">
              <User className="h-5 w-5 text-[#00C0F0]" />
            </div>
            <div className="flex-1">
              <CardHeader
                title="Account"
                description="Manage your account and subscription"
              />
              <div className="flex items-center gap-4 mt-2">
                <div className="h-12 w-12 rounded-full bg-[#00C0F0] flex items-center justify-center">
                  <span className="text-lg font-medium text-white">AC</span>
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-white">Acme Corporation</p>
                  <p className="text-sm text-[#9D9D9D]">admin@acme.com</p>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-[#414141] space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-[#9D9D9D]">Plan</span>
                  <span className="text-sm text-[#D6D6D6] font-medium">Professional</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-[#9D9D9D]">Next billing</span>
                  <span className="text-sm text-[#D6D6D6]">Feb 1, 2026</span>
                </div>
              </div>
              <div className="mt-4 flex gap-3">
                <Button variant="secondary" size="sm" className="flex items-center gap-2">
                  <CreditCard className="h-4 w-4" />
                  Billing
                </Button>
                <Button variant="secondary" size="sm" className="flex items-center gap-2">
                  <ExternalLink className="h-4 w-4" />
                  Upgrade Plan
                </Button>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
