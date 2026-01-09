import Link from 'next/link';
import { Button } from '@/components/ui';
import { Home, Search } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="text-center max-w-md">
        <p className="text-6xl font-bold text-gray-200 mb-4">404</p>
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          Page not found
        </h2>
        <p className="text-sm text-gray-500 mb-6">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="flex items-center justify-center gap-3">
          <Link href="/">
            <Button variant="primary" icon={<Home className="h-4 w-4" />}>
              Go to Dashboard
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
