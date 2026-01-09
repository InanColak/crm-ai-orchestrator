import { Header } from '@/components/layout';
import { Card, CardHeader, Button } from '@/components/ui';
import { Link2, Key, Bell, User, Shield } from 'lucide-react';

const settingsSections = [
  {
    id: 'integrations',
    title: 'Integrations',
    description: 'Connect your CRM and other tools',
    icon: Link2,
    items: [
      { name: 'HubSpot', status: 'connected', statusColor: 'text-green-600' },
      { name: 'Salesforce', status: 'not connected', statusColor: 'text-gray-400' },
      { name: 'Google Calendar', status: 'not connected', statusColor: 'text-gray-400' },
    ],
  },
  {
    id: 'api',
    title: 'API Keys',
    description: 'Manage your API credentials',
    icon: Key,
    items: [
      { name: 'OpenAI API', status: 'configured', statusColor: 'text-green-600' },
      { name: 'Tavily API', status: 'configured', statusColor: 'text-green-600' },
    ],
  },
  {
    id: 'notifications',
    title: 'Notifications',
    description: 'Configure alerts and notifications',
    icon: Bell,
    items: [
      { name: 'Email notifications', status: 'enabled', statusColor: 'text-green-600' },
      { name: 'Slack notifications', status: 'disabled', statusColor: 'text-gray-400' },
    ],
  },
];

export default function SettingsPage() {
  return (
    <div className="min-h-screen">
      <Header
        title="Settings"
        description="Configure your CRM AI Orchestrator"
      />

      <div className="p-6 space-y-6 max-w-4xl">
        {settingsSections.map((section) => {
          const Icon = section.icon;
          return (
            <Card key={section.id}>
              <div className="flex items-start gap-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gray-100 shrink-0">
                  <Icon className="h-5 w-5 text-gray-600" />
                </div>
                <div className="flex-1">
                  <CardHeader
                    title={section.title}
                    description={section.description}
                  />
                  <div className="space-y-3">
                    {section.items.map((item) => (
                      <div
                        key={item.name}
                        className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0"
                      >
                        <span className="text-sm text-gray-900">{item.name}</span>
                        <span className={`text-sm ${item.statusColor}`}>
                          {item.status}
                        </span>
                      </div>
                    ))}
                  </div>
                  <div className="mt-4">
                    <Button variant="secondary" size="sm">
                      Manage
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          );
        })}

        {/* Account Section */}
        <Card>
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gray-100 shrink-0">
              <User className="h-5 w-5 text-gray-600" />
            </div>
            <div className="flex-1">
              <CardHeader
                title="Account"
                description="Manage your account settings"
              />
              <div className="flex items-center gap-4">
                <div className="h-12 w-12 rounded-full bg-brand-100 flex items-center justify-center">
                  <span className="text-lg font-medium text-brand-700">AC</span>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">Acme Corporation</p>
                  <p className="text-sm text-gray-500">admin@acme.com</p>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
