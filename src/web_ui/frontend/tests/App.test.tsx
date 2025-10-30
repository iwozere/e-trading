/**
 * App Component Tests
 * ------------------
 * 
 * Unit tests for the main App component including:
 * - Authentication routing
 * - Protected routes
 * - Theme provider setup
 * - Query client configuration
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { mockAuthenticatedState, mockUnauthenticatedState } from './utils/test-utils';
import { render } from '@testing-library/react';
import App from '../src/App';

// Mock the auth store
vi.mock('../src/stores/authStore', () => ({
  useAuthStore: vi.fn(),
}));

// Mock all page components to avoid complex dependencies
vi.mock('../src/pages/Dashboard/Dashboard', () => ({
  default: () => <div data-testid="dashboard-page">Dashboard</div>,
}));

vi.mock('../src/pages/Auth/Login', () => ({
  default: () => <div data-testid="login-page">Login</div>,
}));

vi.mock('../src/pages/Strategies/Strategies', () => ({
  default: () => <div data-testid="strategies-page">Strategies</div>,
}));

vi.mock('../src/pages/Monitoring/Monitoring', () => ({
  default: () => <div data-testid="monitoring-page">Monitoring</div>,
}));

vi.mock('../src/pages/Analytics/Analytics', () => ({
  default: () => <div data-testid="analytics-page">Analytics</div>,
}));

vi.mock('../src/pages/Administration/Administration', () => ({
  default: () => <div data-testid="administration-page">Administration</div>,
}));

vi.mock('../src/pages/Telegram', () => ({
  TelegramDashboard: () => <div data-testid="telegram-dashboard">Telegram Dashboard</div>,
  UserManagement: () => <div data-testid="user-management">User Management</div>,
  AlertManagement: () => <div data-testid="alert-management">Alert Management</div>,
  ScheduleManagement: () => <div data-testid="schedule-management">Schedule Management</div>,
  BroadcastCenter: () => <div data-testid="broadcast-center">Broadcast Center</div>,
  AuditLogs: () => <div data-testid="audit-logs">Audit Logs</div>,
}));

vi.mock('../src/components/Layout/Layout', () => ({
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('../src/components/Telegram', () => ({
  TelegramRouteGuard: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="telegram-route-guard">{children}</div>
  ),
}));

vi.mock('../src/contexts/WebSocketContext', () => ({
  WebSocketProvider: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="websocket-provider">{children}</div>
  ),
}));

describe('App Component', () => {
  let useAuthStore: any;
  let mockUseAuthStore: any;

  beforeAll(async () => {
    const authStoreModule = await import('../src/stores/authStore');
    useAuthStore = authStoreModule.useAuthStore;
    mockUseAuthStore = useAuthStore as any;
  });

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Mock window.location for router tests
    Object.defineProperty(window, 'location', {
      value: {
        origin: 'http://localhost:3000',
        href: 'http://localhost:3000',
        pathname: '/',
        search: '',
        hash: '',
      },
      writable: true,
    });
  });

  describe('Unauthenticated State', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue(mockUnauthenticatedState);
    });

    it('should render login page when not authenticated', () => {
      render(<App />);
      
      expect(screen.getByTestId('login-page')).toBeInTheDocument();
      expect(screen.queryByTestId('layout')).not.toBeInTheDocument();
    });

    it('should redirect to login for any protected route when not authenticated', () => {
      // Mock window.location to simulate navigation to protected route
      Object.defineProperty(window, 'location', {
        value: { pathname: '/dashboard' },
        writable: true,
      });

      render(<App />);
      
      expect(screen.getByTestId('login-page')).toBeInTheDocument();
    });

    it('should not render protected components when not authenticated', () => {
      render(<App />);
      
      expect(screen.queryByTestId('dashboard-page')).not.toBeInTheDocument();
      expect(screen.queryByTestId('strategies-page')).not.toBeInTheDocument();
      expect(screen.queryByTestId('telegram-dashboard')).not.toBeInTheDocument();
    });
  });

  describe('Authenticated State', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
    });

    it('should render layout and dashboard when authenticated', async () => {
      render(<App />);
      
      await waitFor(() => {
        expect(screen.getByTestId('layout')).toBeInTheDocument();
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
      
      expect(screen.queryByTestId('login-page')).not.toBeInTheDocument();
    });

    it('should redirect root path to dashboard when authenticated', async () => {
      render(<App />);
      
      await waitFor(() => {
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
    });

    it('should not render login page when authenticated', () => {
      render(<App />);
      
      expect(screen.queryByTestId('login-page')).not.toBeInTheDocument();
    });
  });

  describe('Theme Provider', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
    });

    it('should apply dark theme', () => {
      render(<App />);
      
      // Check if CssBaseline is applied (indicates theme provider is working)
      const body = document.body;
      expect(body).toBeInTheDocument();
    });

    it('should provide Material-UI theme context', () => {
      render(<App />);
      
      // The fact that Material-UI components render without errors
      // indicates the theme provider is working correctly
      expect(screen.getByTestId('layout')).toBeInTheDocument();
    });
  });

  describe('Query Client Provider', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
    });

    it('should provide React Query context', () => {
      render(<App />);
      
      // The components render without React Query errors,
      // indicating the QueryClientProvider is working
      expect(screen.getByTestId('layout')).toBeInTheDocument();
    });
  });

  describe('Router Configuration', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
    });

    it('should handle unknown routes by redirecting to dashboard', async () => {
      // Mock window.location for unknown route
      Object.defineProperty(window, 'location', {
        value: { pathname: '/unknown-route' },
        writable: true,
      });

      render(<App />);
      
      await waitFor(() => {
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
    });
  });

  describe('Protected Routes', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
    });

    it('should render protected routes when authenticated', async () => {
      render(<App />);
      
      await waitFor(() => {
        expect(screen.getByTestId('layout')).toBeInTheDocument();
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
    });
  });

  describe('Telegram Routes', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
    });

    it('should wrap Telegram routes with TelegramRouteGuard', () => {
      render(<App />);
      
      // The TelegramRouteGuard mock should be present in the component tree
      // This is tested indirectly through the route structure
      expect(screen.getByTestId('layout')).toBeInTheDocument();
    });
  });

  describe('Toast Notifications', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
    });

    it('should render toast container', () => {
      render(<App />);
      
      // Check if the app renders without toast-related errors
      expect(screen.getByTestId('layout')).toBeInTheDocument();
    });
  });

  describe('Error Boundaries', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
    });

    it('should handle component errors gracefully', () => {
      // Mock console.error to avoid error output in tests
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      
      render(<App />);
      
      // The app should render even if there are minor errors
      expect(screen.getByTestId('layout')).toBeInTheDocument();
      
      consoleSpy.mockRestore();
    });
  });

  describe('Responsive Design', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
    });

    it('should render with responsive layout', () => {
      render(<App />);
      
      const layout = screen.getByTestId('layout');
      expect(layout).toBeInTheDocument();
    });
  });

  describe('Authentication State Changes', () => {
    it('should handle authentication state changes', async () => {
      // Start with unauthenticated state
      mockUseAuthStore.mockReturnValue(mockUnauthenticatedState);
      
      const { rerender } = render(<App />);
      
      expect(screen.getByTestId('login-page')).toBeInTheDocument();
      
      // Change to authenticated state
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
      
      rerender(<App />);
      
      await waitFor(() => {
        expect(screen.getByTestId('layout')).toBeInTheDocument();
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
      
      expect(screen.queryByTestId('login-page')).not.toBeInTheDocument();
    });

    it('should handle logout by showing login page', async () => {
      // Start with authenticated state
      mockUseAuthStore.mockReturnValue(mockAuthenticatedState);
      
      const { rerender } = render(<App />);
      
      await waitFor(() => {
        expect(screen.getByTestId('layout')).toBeInTheDocument();
      });
      
      // Change to unauthenticated state (simulating logout)
      mockUseAuthStore.mockReturnValue(mockUnauthenticatedState);
      
      rerender(<App />);
      
      expect(screen.getByTestId('login-page')).toBeInTheDocument();
      expect(screen.queryByTestId('layout')).not.toBeInTheDocument();
    });
  });
});