'use client';

import { Bell, Search, HelpCircle } from 'lucide-react';

interface HeaderProps {
  title: string;
  description?: string;
}

export function Header({ title, description }: HeaderProps) {
  return (
    <header className="sticky top-0 z-30 border-b border-[#414141] bg-[#191919]/80 backdrop-blur-md">
      <div className="flex h-16 items-center justify-between px-6">
        {/* Page Title */}
        <div>
          <h1 className="text-xl font-semibold text-white">{title}</h1>
          {description && (
            <p className="text-sm text-[#9D9D9D]">{description}</p>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {/* Search */}
          <button className="p-2 rounded-lg hover:bg-[#414141]/50 transition-colors">
            <Search className="h-4 w-4 text-[#9D9D9D]" />
          </button>

          {/* Notifications */}
          <button className="p-2 rounded-lg hover:bg-[#414141]/50 transition-colors relative">
            <Bell className="h-4 w-4 text-[#9D9D9D]" />
            <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-[#00C0F0]" />
          </button>

          {/* Help */}
          <button className="p-2 rounded-lg hover:bg-[#414141]/50 transition-colors">
            <HelpCircle className="h-4 w-4 text-[#9D9D9D]" />
          </button>
        </div>
      </div>
    </header>
  );
}
