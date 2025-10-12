/**
 * Test Utilities
 * -------------
 * 
 * Custom render functions and utilities for testing React components
 * with proper providers and context setup.
 */

import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { vi } from 'vitest';

// Create test theme (simplified version of the app theme)
const testTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#0a0e27',
      paper: '#1a1d3a',
    },
  },
});

// Create test query client
const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      gcTime: 0, // Replaced cacheTime with gcTime (TanStack Query v5+)
    },
    mutations: {
      retry: false,
    },
  },
});

interface AllTheProvidersProps {
  children: React.ReactNode;
  queryClient?: QueryClient;
}

const AllTheProviders: React.FC<AllTheProvidersProps> = ({
  children,
  queryClient = createTestQueryClient()
}) => {
  return (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <ThemeProvider theme={testTheme}>
          <CssBaseline />
          {children}
        </ThemeProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
};

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  queryClient?: QueryClient;
}

const customRender = (
  ui: ReactElement,
  options: CustomRenderOptions = {}
) => {
  const { queryClient, ...renderOptions } = options;

  return render(ui, {
    wrapper: ({ children }) => (
      <AllTheProviders queryClient={queryClient}>
        {children}
      </AllTheProviders>
    ),
    ...renderOptions,
  });
};

// Mock user for authentication tests
export const mockUser = {
  username: 'testuser',
  role: 'admin' as const,
};

export const mockTraderUser = {
  username: 'trader',
  role: 'trader' as const,
};

export const mockViewerUser = {
  username: 'viewer',
  role: 'viewer' as const,
};

// Mock auth store states
export const mockAuthenticatedState = {
  isAuthenticated: true,
  user: mockUser,
  token: 'mock-jwt-token',
  login: vi.fn(),
  logout: vi.fn(),
  setToken: vi.fn(),
};

export const mockUnauthenticatedState = {
  isAuthenticated: false,
  user: null,
  token: null,
  login: vi.fn(),
  logout: vi.fn(),
  setToken: vi.fn(),
};

// Mock API responses
export const mockApiResponse = <T,>(data: T, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: () => Promise.resolve(data),
  text: () => Promise.resolve(JSON.stringify(data)),
});

export const mockApiError = (message: string, status = 500) => ({
  ok: false,
  status,
  json: () => Promise.resolve({ detail: message }),
  text: () => Promise.resolve(JSON.stringify({ detail: message })),
});

// Mock fetch responses
export const mockFetchSuccess = <T,>(data: T) => {
  const mockFetch = vi.fn().mockResolvedValue(mockApiResponse(data));
  (globalThis as any).fetch = mockFetch;
  return mockFetch;
};

export const mockFetchError = (message: string, status = 500) => {
  const mockFetch = vi.fn().mockResolvedValue(mockApiError(message, status));
  (globalThis as any).fetch = mockFetch;
  return mockFetch;
};

// Mock Telegram data
export const mockTelegramUsers = [
  {
    telegram_user_id: '123456789',
    email: 'user1@example.com',
    verified: true,
    approved: true,
    language: 'en',
    is_admin: false,
    max_alerts: 5,
    max_schedules: 5,
  },
  {
    telegram_user_id: '987654321',
    email: 'user2@example.com',
    verified: true,
    approved: false,
    language: 'en',
    is_admin: false,
    max_alerts: 5,
    max_schedules: 5,
  },
];

export const mockTelegramAlerts = [
  {
    id: 1,
    user_id: '123456789',
    ticker: 'BTCUSDT',
    price: 50000,
    condition: 'above',
    active: true,
    email: false,
    created: '2024-01-01T00:00:00Z',
  },
  {
    id: 2,
    user_id: '123456789',
    ticker: 'ETHUSDT',
    price: 3000,
    condition: 'below',
    active: false,
    email: true,
    created: '2024-01-01T00:00:00Z',
  },
];

export const mockTelegramStats = {
  total_users: 10,
  verified_users: 8,
  approved_users: 6,
  pending_approvals: 2,
  admin_users: 1,
};

// Mock strategy data
export const mockStrategies = [
  {
    instance_id: 'strategy-1',
    name: 'Test Strategy 1',
    status: 'running',
    uptime_seconds: 3600,
    error_count: 0,
    last_error: null,
    broker_type: 'paper',
    trading_mode: 'paper',
    symbol: 'BTCUSDT',
    strategy_type: 'sma_crossover',
  },
  {
    instance_id: 'strategy-2',
    name: 'Test Strategy 2',
    status: 'stopped',
    uptime_seconds: 0,
    error_count: 1,
    last_error: 'Connection timeout',
    broker_type: 'live',
    trading_mode: 'live',
    symbol: 'ETHUSDT',
    strategy_type: 'rsi',
  },
];

// Mock system status
export const mockSystemStatus = {
  service_name: 'Enhanced Multi-Strategy Trading System',
  version: '2.0.0',
  status: 'running',
  uptime_seconds: 86400,
  active_strategies: 1,
  total_strategies: 2,
  system_metrics: {
    cpu_percent: 25.5,
    memory_percent: 60.2,
    temperature_c: 45.0,
    disk_usage_percent: 75.0,
  },
};

// Utility to wait for async operations
export const waitFor = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Re-export everything from React Testing Library
export * from '@testing-library/react';
export { customRender as render };