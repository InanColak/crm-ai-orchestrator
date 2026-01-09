'use client';

import { useEffect } from 'react';
import { Button } from '@/components/ui';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error('Application error:', error);
  }, [error]);

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="text-center max-w-md">
        <div className="mx-auto h-16 w-16 rounded-full bg-red-100 flex items-center justify-center mb-4">
          <AlertTriangle className="h-8 w-8 text-red-600" />
        </div>
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          Something went wrong
        </h2>
        <p className="text-sm text-gray-500 mb-6">
          An unexpected error occurred. Please try again or contact support if the
          problem persists.
        </p>
        <div className="flex items-center justify-center gap-3">
          <Button
            variant="primary"
            onClick={reset}
            icon={<RefreshCw className="h-4 w-4" />}
          >
            Try Again
          </Button>
          <Button variant="secondary" onClick={() => window.location.href = '/'}>
            Go to Dashboard
          </Button>
        </div>
        {error.digest && (
          <p className="mt-4 text-xs text-gray-400 font-mono">
            Error ID: {error.digest}
          </p>
        )}
      </div>
    </div>
  );
}
