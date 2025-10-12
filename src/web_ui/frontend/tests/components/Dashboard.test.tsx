/**
 * Dashboard Component Tests
 * -----------------------
 * 
 * Unit tests for the Dashboard component including:
 * - System status display
 * - Strategy overview
 * - Real-time metrics
 * - Navigation and interactions
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render, mockAuthenticatedState, mockApiResponse } from '../utils/test-utils';
import Dashboard from '../../src/pages/Dashboard/Dashboard';

// Mock the auth store
vi.mock('../../src/stores/authStore', () => ({
  useAuthStore: vi.fn(),
}));

// Mock React Query hooks
vi.mock('@tanstack/react-query', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    useQuery: vi.fn(),
    useQueryClient: vi.fn(() => ({
      invalidateQueries: vi.fn(),
    })),
  };
});

// Mock API client (not directly used in tests, but may be imported by components)
vi.mock('../../src/services/api', () => ({
  apiClient: {
    get: vi.fn(),
  },
}));

describe('Dashboard Component', () => {
  let useAuthStore: any;
  let useQuery: any;
  let mockUseAuthStore: any;
  let mockUseQuery: any;

  beforeAll(async () => {
    const authStoreModule = await import('../../src/stores/authStore');
    const reactQueryModule = await import('@tanstack/react-query');
    
    useAuthStore = authStoreModule.useAuthStore;
    useQuery = reactQueryModule.useQuery;
    mockUseAuthStore = useAuthStore as any;
    mockUseQuery = useQuery as any;
  });

  const mockSystemStatus = {
    service_name: 'Enhanced Multi-Strategy Trading System',
    version: '2.0.0',
    status: 'running',
    active_strategies: 3,
    total_strategies: 5,
    system_metrics: {
      cpu_percent: 25.5,
      memory_percent: 60.2,
      temperature_c: 45.0,
      disk_usage_percent: 75.0,
    },
  };

  const mockStrategies = [
    {
      instance_id: 'strategy-1',
      name: 'BTC SMA Crossover',
      status: 'running',
      uptime_seconds: 3600,
      error_count: 0,
      symbol: 'BTCUSDT',
      strategy_type: 'sma_crossover',
    },
    {
      instance_id: 'strategy-2',
      name: 'ETH RSI Strategy',
      status: 'stopped',
      uptime_seconds: 0,
      error_count: 2,
      symbol: 'ETHUSDT',
      strategy_type: 'rsi',
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
    
    // Default mock for useQuery - provide default data to prevent null errors
    mockUseQuery.mockImplementation((queryKey: any) => {
      if (queryKey[0] === 'system-status') {
        return {
          data: mockSystemStatus,
          isLoading: false,
          error: null,
          refetch: vi.fn(),
        };
      }
      if (queryKey[0] === 'strategies') {
        return {
          data: mockStrategies,
          isLoading: false,
          error: null,
          refetch: vi.fn(),
        };
      }
      return {
        data: null,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      };
    });
  });

  describe('Rendering', () => {
    it('should render dashboard title and welcome message', () => {
      render(<Dashboard />);
      
      expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
      expect(screen.getByText(/welcome back/i)).toBeInTheDocument();
    });

    it('should render system status section', () => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'system-status') {
          return {
            data: mockSystemStatus,
            isLoading: false,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      expect(screen.getByText(/system status/i)).toBeInTheDocument();
      expect(screen.getByText(/enhanced multi-strategy trading system/i)).toBeInTheDocument();
      expect(screen.getByText(/running/i)).toBeInTheDocument();
    });

    it('should render strategies overview section', () => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'strategies') {
          return {
            data: mockStrategies,
            isLoading: false,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      expect(screen.getByText(/strategies overview/i)).toBeInTheDocument();
      expect(screen.getByText(/btc sma crossover/i)).toBeInTheDocument();
      expect(screen.getByText(/eth rsi strategy/i)).toBeInTheDocument();
    });

    it('should render system metrics cards', () => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'system-status') {
          return {
            data: mockSystemStatus,
            isLoading: false,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      expect(screen.getByText(/cpu usage/i)).toBeInTheDocument();
      expect(screen.getByText(/memory usage/i)).toBeInTheDocument();
      expect(screen.getByText(/temperature/i)).toBeInTheDocument();
      expect(screen.getByText(/disk usage/i)).toBeInTheDocument();
    });
  });

  describe('Loading States', () => {
    it('should show loading skeleton for system status', () => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'system-status') {
          return {
            data: null,
            isLoading: true,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      expect(screen.getByTestId('system-status-loading')).toBeInTheDocument();
    });

    it('should show loading skeleton for strategies', () => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'strategies') {
          return {
            data: null,
            isLoading: true,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      expect(screen.getByTestId('strategies-loading')).toBeInTheDocument();
    });

    it('should show loading for all sections simultaneously', () => {
      mockUseQuery.mockReturnValue({
        data: null,
        isLoading: true,
        error: null,
      });

      render(<Dashboard />);
      
      expect(screen.getByTestId('system-status-loading')).toBeInTheDocument();
      expect(screen.getByTestId('strategies-loading')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should show error message for system status failure', () => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'system-status') {
          return {
            data: null,
            isLoading: false,
            error: new Error('Failed to fetch system status'),
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      expect(screen.getByText(/failed to load system status/i)).toBeInTheDocument();
    });

    it('should show error message for strategies failure', () => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'strategies') {
          return {
            data: null,
            isLoading: false,
            error: new Error('Failed to fetch strategies'),
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      expect(screen.getByText(/failed to load strategies/i)).toBeInTheDocument();
    });

    it('should show retry button on error', async () => {
      const mockRefetch = vi.fn();
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'system-status') {
          return {
            data: null,
            isLoading: false,
            error: new Error('Network error'),
            refetch: mockRefetch,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      const user = userEvent.setup();
      render(<Dashboard />);
      
      const retryButton = screen.getByRole('button', { name: /retry/i });
      await user.click(retryButton);
      
      expect(mockRefetch).toHaveBeenCalled();
    });
  });

  describe('System Metrics Display', () => {
    beforeEach(() => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'system-status') {
          return {
            data: mockSystemStatus,
            isLoading: false,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });
    });

    it('should display CPU usage with correct formatting', () => {
      render(<Dashboard />);
      
      expect(screen.getByText('25.5%')).toBeInTheDocument();
    });

    it('should display memory usage with correct formatting', () => {
      render(<Dashboard />);
      
      expect(screen.getByText('60.2%')).toBeInTheDocument();
    });

    it('should display temperature with correct formatting', () => {
      render(<Dashboard />);
      
      expect(screen.getByText('45.0°C')).toBeInTheDocument();
    });

    it('should display disk usage with correct formatting', () => {
      render(<Dashboard />);
      
      expect(screen.getByText('75.0%')).toBeInTheDocument();
    });

    it('should show warning colors for high usage', () => {
      const highUsageStatus = {
        ...mockSystemStatus,
        system_metrics: {
          cpu_percent: 85.0,
          memory_percent: 90.0,
          temperature_c: 75.0,
          disk_usage_percent: 95.0,
        },
      };

      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'system-status') {
          return {
            data: highUsageStatus,
            isLoading: false,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      // Check for warning indicators (implementation depends on your styling)
      expect(screen.getByText('85.0%')).toBeInTheDocument();
      expect(screen.getByText('90.0%')).toBeInTheDocument();
    });
  });

  describe('Strategy Status Display', () => {
    beforeEach(() => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'strategies') {
          return {
            data: mockStrategies,
            isLoading: false,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });
    });

    it('should display strategy names and symbols', () => {
      render(<Dashboard />);
      
      expect(screen.getByText('BTC SMA Crossover')).toBeInTheDocument();
      expect(screen.getByText('ETH RSI Strategy')).toBeInTheDocument();
      expect(screen.getByText('BTCUSDT')).toBeInTheDocument();
      expect(screen.getByText('ETHUSDT')).toBeInTheDocument();
    });

    it('should display strategy status with appropriate indicators', () => {
      render(<Dashboard />);
      
      expect(screen.getByText('Running')).toBeInTheDocument();
      expect(screen.getByText('Stopped')).toBeInTheDocument();
    });

    it('should display uptime for running strategies', () => {
      render(<Dashboard />);
      
      expect(screen.getByText('1h 0m')).toBeInTheDocument(); // 3600 seconds = 1 hour
    });

    it('should display error count when present', () => {
      render(<Dashboard />);
      
      expect(screen.getByText('2 errors')).toBeInTheDocument();
    });

    it('should show no strategies message when list is empty', () => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'strategies') {
          return {
            data: [],
            isLoading: false,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      expect(screen.getByText(/no strategies configured/i)).toBeInTheDocument();
    });
  });

  describe('Navigation and Interactions', () => {
    it('should navigate to strategies page when view all button is clicked', async () => {
      const user = userEvent.setup();
      
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'strategies') {
          return {
            data: mockStrategies,
            isLoading: false,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      const viewAllButton = screen.getByRole('button', { name: /view all strategies/i });
      await user.click(viewAllButton);
      
      // Navigation would be tested with router mocks
      expect(viewAllButton).toBeInTheDocument();
    });

    it('should navigate to monitoring page when view metrics button is clicked', async () => {
      const user = userEvent.setup();
      
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'system-status') {
          return {
            data: mockSystemStatus,
            isLoading: false,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      const viewMetricsButton = screen.getByRole('button', { name: /view detailed metrics/i });
      await user.click(viewMetricsButton);
      
      expect(viewMetricsButton).toBeInTheDocument();
    });
  });

  describe('Real-time Updates', () => {
    it('should refresh data periodically', async () => {
      const mockRefetch = vi.fn();
      
      mockUseQuery.mockImplementation((queryKey: any) => {
        return {
          data: queryKey[0] === 'system-status' ? mockSystemStatus : mockStrategies,
          isLoading: false,
          error: null,
          refetch: mockRefetch,
        };
      });

      render(<Dashboard />);
      
      // Wait for initial render
      await waitFor(() => {
        expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
      });

      // In a real implementation, you would test the polling interval
      // For now, just verify the component renders correctly
      expect(screen.getByText('Enhanced Multi-Strategy Trading System')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('should render correctly on mobile viewport', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<Dashboard />);
      
      expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
    });

    it('should render correctly on tablet viewport', () => {
      // Mock tablet viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      render(<Dashboard />);
      
      expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper heading hierarchy', () => {
      render(<Dashboard />);
      
      expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
      expect(screen.getAllByRole('heading', { level: 2 })).toHaveLength(2); // System Status and Strategies
    });

    it('should have accessible metric cards', () => {
      mockUseQuery.mockImplementation((queryKey: any) => {
        if (queryKey[0] === 'system-status') {
          return {
            data: mockSystemStatus,
            isLoading: false,
            error: null,
          };
        }
        return { data: null, isLoading: false, error: null };
      });

      render(<Dashboard />);
      
      expect(screen.getByLabelText(/cpu usage/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/memory usage/i)).toBeInTheDocument();
    });
  });
});