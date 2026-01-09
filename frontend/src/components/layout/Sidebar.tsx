'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Workflow,
  CheckCircle,
  Mail,
  FileText,
  Settings,
  Bot,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: number;
}

const navItems: NavItem[] = [
  { label: 'Dashboard', href: '/', icon: LayoutDashboard },
  { label: 'Workflows', href: '/workflows', icon: Workflow },
  { label: 'Approvals', href: '/approvals', icon: CheckCircle },
  { label: 'Email Copilot', href: '/email-copilot', icon: Mail },
  { label: 'Documents', href: '/documents', icon: FileText },
  { label: 'Settings', href: '/settings', icon: Settings },
];

interface SidebarProps {
  pendingApprovals?: number;
}

export function Sidebar({ pendingApprovals = 0 }: SidebarProps) {
  const pathname = usePathname();

  // Add badge to approvals nav item
  const itemsWithBadges = navItems.map(item => ({
    ...item,
    badge: item.href === '/approvals' && pendingApprovals > 0 ? pendingApprovals : undefined,
  }));

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r border-[#414141] bg-[#191919]">
      {/* Logo */}
      <div className="flex h-16 items-center gap-2 border-b border-[#414141] px-6">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#00C0F0]">
          <Bot className="h-5 w-5 text-white" />
        </div>
        <span className="text-lg font-semibold text-white">CRM AI</span>
      </div>

      {/* Navigation */}
      <nav className="space-y-1 p-4">
        {itemsWithBadges.map((item) => {
          const isActive = pathname === item.href ||
            (item.href !== '/' && pathname.startsWith(item.href));
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-[#00C0F0] text-white'
                  : 'text-[#D6D6D6] hover:bg-[#414141]/30 hover:text-white'
              )}
            >
              <Icon className={cn('h-5 w-5', isActive ? 'text-white' : 'text-[#9D9D9D]')} />
              <span className="flex-1">{item.label}</span>
              {item.badge && (
                <span className="flex h-5 min-w-5 items-center justify-center rounded-full bg-red-500 px-1.5 text-xs font-medium text-white">
                  {item.badge > 99 ? '99+' : item.badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="absolute bottom-0 left-0 right-0 border-t border-[#414141] p-4">
        <div className="flex items-center gap-3 rounded-lg bg-[#414141]/30 px-3 py-2">
          <div className="h-8 w-8 rounded-full bg-[#00C0F0] flex items-center justify-center">
            <span className="text-sm font-medium text-white">AC</span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-white truncate">Acme Corp</p>
            <p className="text-xs text-[#9D9D9D] truncate">client@acme.com</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
