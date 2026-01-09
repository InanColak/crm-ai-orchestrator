'use client';

import { useState } from 'react';
import { Header } from '@/components/layout';
import { Card, Button } from '@/components/ui';
import {
  Send,
  Sparkles,
  User,
  Building,
  Mail,
  RefreshCw,
  Copy,
  Check,
} from 'lucide-react';
import { cn } from '@/lib/utils';

type EmailType = 'cold_outreach' | 'follow_up' | 'post_meeting';

const emailTypes: { value: EmailType; label: string; description: string }[] = [
  {
    value: 'cold_outreach',
    label: 'Cold Outreach',
    description: 'First contact with a new prospect',
  },
  {
    value: 'follow_up',
    label: 'Follow Up',
    description: 'Continue an existing conversation',
  },
  {
    value: 'post_meeting',
    label: 'Post Meeting',
    description: 'Recap after a meeting or call',
  },
];

export default function EmailCopilotPage() {
  const [emailType, setEmailType] = useState<EmailType>('cold_outreach');
  const [recipientName, setRecipientName] = useState('');
  const [recipientEmail, setRecipientEmail] = useState('');
  const [company, setCompany] = useState('');
  const [context, setContext] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedEmail, setGeneratedEmail] = useState<{
    subject: string;
    body: string;
  } | null>(null);
  const [copied, setCopied] = useState(false);

  const handleGenerate = async () => {
    setIsGenerating(true);

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000));

    setGeneratedEmail({
      subject: `Quick question about ${company}'s growth strategy`,
      body: `Hi ${recipientName || 'there'},

I noticed that ${company || 'your company'} has been making impressive strides in the market lately. ${context ? `Specifically, ${context.toLowerCase()}` : ''}

At CRM AI Orchestrator, we help companies like yours automate their sales processes and improve lead conversion rates by up to 40%.

Would you be open to a brief 15-minute call next week to explore if there might be a fit?

Best regards,
Your Name`,
    });

    setIsGenerating(false);
  };

  const handleCopy = () => {
    if (generatedEmail) {
      navigator.clipboard.writeText(
        `Subject: ${generatedEmail.subject}\n\n${generatedEmail.body}`
      );
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleRegenerate = () => {
    setGeneratedEmail(null);
    handleGenerate();
  };

  return (
    <div className="min-h-screen">
      <Header
        title="Email Copilot"
        description="Generate personalized sales emails with AI"
      />

      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Form */}
          <Card>
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Email Details
            </h2>

            <div className="space-y-4">
              {/* Email Type */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Email Type
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {emailTypes.map((type) => (
                    <button
                      key={type.value}
                      onClick={() => setEmailType(type.value)}
                      className={cn(
                        'p-3 rounded-lg border text-left transition-all',
                        emailType === type.value
                          ? 'border-brand-500 bg-brand-50 ring-1 ring-brand-500'
                          : 'border-gray-200 hover:border-gray-300'
                      )}
                    >
                      <p className="text-sm font-medium text-gray-900">
                        {type.label}
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {type.description}
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Recipient Info */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Recipient Name
                  </label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input
                      type="text"
                      value={recipientName}
                      onChange={(e) => setRecipientName(e.target.value)}
                      placeholder="John Smith"
                      className="input pl-9"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Recipient Email
                  </label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input
                      type="email"
                      value={recipientEmail}
                      onChange={(e) => setRecipientEmail(e.target.value)}
                      placeholder="john@company.com"
                      className="input pl-9"
                    />
                  </div>
                </div>
              </div>

              {/* Company */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Company
                </label>
                <div className="relative">
                  <Building className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    value={company}
                    onChange={(e) => setCompany(e.target.value)}
                    placeholder="Acme Corporation"
                    className="input pl-9"
                  />
                </div>
              </div>

              {/* Context */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Additional Context (optional)
                </label>
                <textarea
                  value={context}
                  onChange={(e) => setContext(e.target.value)}
                  placeholder="Any specific details about the prospect, recent news, or talking points..."
                  rows={4}
                  className="input resize-none"
                />
              </div>

              {/* Generate Button */}
              <Button
                className="w-full"
                size="lg"
                onClick={handleGenerate}
                loading={isGenerating}
                icon={<Sparkles className="h-4 w-4" />}
              >
                {isGenerating ? 'Generating...' : 'Generate Email'}
              </Button>
            </div>
          </Card>

          {/* Generated Email */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">
                Generated Email
              </h2>
              {generatedEmail && (
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleRegenerate}
                    icon={<RefreshCw className="h-4 w-4" />}
                  >
                    Regenerate
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={handleCopy}
                    icon={
                      copied ? (
                        <Check className="h-4 w-4 text-green-600" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )
                    }
                  >
                    {copied ? 'Copied!' : 'Copy'}
                  </Button>
                </div>
              )}
            </div>

            {generatedEmail ? (
              <div className="space-y-4">
                {/* Subject */}
                <div>
                  <label className="block text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
                    Subject
                  </label>
                  <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                    <p className="text-sm text-gray-900">
                      {generatedEmail.subject}
                    </p>
                  </div>
                </div>

                {/* Body */}
                <div>
                  <label className="block text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
                    Body
                  </label>
                  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <pre className="text-sm text-gray-900 whitespace-pre-wrap font-sans">
                      {generatedEmail.body}
                    </pre>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2 pt-2">
                  <Button
                    className="flex-1"
                    icon={<Send className="h-4 w-4" />}
                  >
                    Send for Approval
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-80 text-center">
                <div className="h-16 w-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                  <Sparkles className="h-8 w-8 text-gray-400" />
                </div>
                <p className="text-sm text-gray-500">
                  Fill in the details and click "Generate Email" to create a
                  personalized message
                </p>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
