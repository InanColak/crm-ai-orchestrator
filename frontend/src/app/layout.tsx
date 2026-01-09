import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Sidebar } from '@/components/layout';
import { Providers } from './providers';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: 'CRM AI Orchestrator',
  description: 'Autonomous Growth & Sales Engine powered by AI',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans">
        <Providers>
          <div className="flex min-h-screen">
            <Sidebar />
            <main
              className="flex-1 ml-64 relative min-h-screen"
              style={{
                backgroundImage: 'url(/bg-hero.jpg)',
                backgroundSize: 'cover',
                backgroundPosition: 'center right',
                backgroundRepeat: 'no-repeat',
                backgroundAttachment: 'fixed',
              }}
            >
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
