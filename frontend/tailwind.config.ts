import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Content Moves Design System
        cm: {
          bg: '#191919',           // Background - near black
          'bg-light': '#1f1f1f',   // Slightly lighter bg
          card: 'rgba(25, 25, 25, 0.8)', // Card with transparency
          border: '#414141',       // Border - dark gray
          cyan: '#00C0F0',         // Primary accent - cyan
          'cyan-hover': '#00a8d4', // Cyan hover state
          'text-white': '#ffffff', // Primary text
          'text-light': '#D6D6D6', // Secondary text
          'text-muted': '#9D9D9D', // Muted text
          input: 'rgba(0, 0, 0, 0.5)', // Input background
        },
        // Status colors - glass style
        status: {
          pending: '#f59e0b',
          running: '#00C0F0',
          completed: '#10b981',
          failed: '#ef4444',
          waiting: '#8b5cf6',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 2s linear infinite',
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
        'card-hover': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
      },
    },
  },
  plugins: [],
};

export default config;
