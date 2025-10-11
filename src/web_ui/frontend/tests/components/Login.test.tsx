/**
 * Login Component Tests
 * -------------------
 * 
 * Unit tests for the Login component including:
 * - Form validation
 * - Authentication flow
 * - Error handling
 * - Loading states
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render, mockUnauthenticatedState } from '../utils/test-utils';
import Login from '../../src/pages/Auth/Login';

// Mock the auth store
vi.mock('../../src/stores/authStore', () => ({
  useAuthStore: vi.fn(),
}));

// Mock react-router-dom
vi.mock('react-router-dom', () => ({
  useNavigate: () => vi.fn(),
}));

describe('Login Component', () => {
  const mockLogin = vi.fn();
  let useAuthStore: any;
  let mockUseAuthStore: any;

  beforeAll(async () => {
    const authStoreModule = await import('../../src/stores/authStore');
    useAuthStore = authStoreModule.useAuthStore;
    mockUseAuthStore = useAuthStore as any;
  });

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthStore.mockReturnValue({
      ...mockUnauthenticatedState,
      login: mockLogin,
    });
  });

  describe('Rendering', () => {
    it('should render login form elements', () => {
      render(<Login />);
      
      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    });

    it('should render trading system branding', () => {
      render(<Login />);
      
      expect(screen.getByText(/trading web ui/i)).toBeInTheDocument();
      expect(screen.getByText(/enhanced multi-strategy trading system/i)).toBeInTheDocument();
    });

    it('should render login form in a card layout', () => {
      render(<Login />);
      
      const form = screen.getByRole('form');
      expect(form).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('should show validation errors for empty fields', async () => {
      const user = userEvent.setup();
      render(<Login />);
      
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);
      
      expect(screen.getByText(/username is required/i)).toBeInTheDocument();
      expect(screen.getByText(/password is required/i)).toBeInTheDocument();
    });

    it('should show validation error for empty username', async () => {
      const user = userEvent.setup();
      render(<Login />);
      
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);
      
      expect(screen.getByText(/username is required/i)).toBeInTheDocument();
      expect(screen.queryByText(/password is required/i)).not.toBeInTheDocument();
    });

    it('should show validation error for empty password', async () => {
      const user = userEvent.setup();
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(usernameInput, 'admin');
      await user.click(submitButton);
      
      expect(screen.getByText(/password is required/i)).toBeInTheDocument();
      expect(screen.queryByText(/username is required/i)).not.toBeInTheDocument();
    });

    it('should clear validation errors when user starts typing', async () => {
      const user = userEvent.setup();
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      // Trigger validation error
      await user.click(submitButton);
      expect(screen.getByText(/username is required/i)).toBeInTheDocument();
      
      // Start typing to clear error
      await user.type(usernameInput, 'a');
      expect(screen.queryByText(/username is required/i)).not.toBeInTheDocument();
    });
  });

  describe('Authentication Flow', () => {
    it('should call login function with correct credentials', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue(true);
      
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);
      
      expect(mockLogin).toHaveBeenCalledWith('admin', 'password123');
    });

    it('should show loading state during authentication', async () => {
      const user = userEvent.setup();
      let resolveLogin: (value: boolean) => void;
      const loginPromise = new Promise<boolean>((resolve) => {
        resolveLogin = resolve;
      });
      mockLogin.mockReturnValue(loginPromise);
      
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);
      
      // Should show loading state
      expect(screen.getByRole('button', { name: /signing in/i })).toBeInTheDocument();
      expect(submitButton).toBeDisabled();
      
      // Resolve the login
      resolveLogin!(true);
      
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
      });
    });

    it('should handle successful login', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue(true);
      
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith('admin', 'password123');
      });
    });

    it('should handle failed login', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue(false);
      
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'wrongpassword');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('should show error message for authentication failure', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue(false);
      
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'wrongpassword');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
      });
    });

    it('should handle network errors gracefully', async () => {
      const user = userEvent.setup();
      mockLogin.mockRejectedValue(new Error('Network error'));
      
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(screen.getByText(/network error/i)).toBeInTheDocument();
      });
    });

    it('should clear error message when user starts typing', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue(false);
      
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      // Trigger error
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'wrongpassword');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
      });
      
      // Clear error by typing
      await user.clear(passwordInput);
      await user.type(passwordInput, 'newpassword');
      
      expect(screen.queryByText(/invalid credentials/i)).not.toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper form labels', () => {
      render(<Login />);
      
      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    });

    it('should support keyboard navigation', async () => {
      const user = userEvent.setup();
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      // Tab through form elements
      await user.tab();
      expect(usernameInput).toHaveFocus();
      
      await user.tab();
      expect(passwordInput).toHaveFocus();
      
      await user.tab();
      expect(submitButton).toHaveFocus();
    });

    it('should support form submission with Enter key', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue(true);
      
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'password123');
      await user.keyboard('{Enter}');
      
      expect(mockLogin).toHaveBeenCalledWith('admin', 'password123');
    });
  });

  describe('Password Visibility Toggle', () => {
    it('should toggle password visibility', async () => {
      const user = userEvent.setup();
      render(<Login />);
      
      const passwordInput = screen.getByLabelText(/password/i);
      const toggleButton = screen.getByRole('button', { name: /toggle password visibility/i });
      
      // Initially password should be hidden
      expect(passwordInput).toHaveAttribute('type', 'password');
      
      // Click to show password
      await user.click(toggleButton);
      expect(passwordInput).toHaveAttribute('type', 'text');
      
      // Click to hide password again
      await user.click(toggleButton);
      expect(passwordInput).toHaveAttribute('type', 'password');
    });
  });

  describe('Form Reset', () => {
    it('should clear form when component unmounts and remounts', () => {
      const { unmount } = render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      
      fireEvent.change(usernameInput, { target: { value: 'testuser' } });
      fireEvent.change(passwordInput, { target: { value: 'testpass' } });
      
      expect(usernameInput).toHaveValue('testuser');
      expect(passwordInput).toHaveValue('testpass');
      
      unmount();
      
      render(<Login />);
      
      const newUsernameInput = screen.getByLabelText(/username/i);
      const newPasswordInput = screen.getByLabelText(/password/i);
      
      expect(newUsernameInput).toHaveValue('');
      expect(newPasswordInput).toHaveValue('');
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long usernames and passwords', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue(true);
      
      render(<Login />);
      
      const longString = 'a'.repeat(1000);
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(usernameInput, longString);
      await user.type(passwordInput, longString);
      await user.click(submitButton);
      
      expect(mockLogin).toHaveBeenCalledWith(longString, longString);
    });

    it('should handle special characters in credentials', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue(true);
      
      render(<Login />);
      
      const specialUsername = 'user@domain.com';
      const specialPassword = 'p@ssw0rd!#$%';
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(usernameInput, specialUsername);
      await user.type(passwordInput, specialPassword);
      await user.click(submitButton);
      
      expect(mockLogin).toHaveBeenCalledWith(specialUsername, specialPassword);
    });
  });
});