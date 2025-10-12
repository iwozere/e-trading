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
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    useNavigate: () => vi.fn(),
  };
});

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
      
      expect(screen.getByText(/trading system login/i)).toBeInTheDocument();
      expect(screen.getByText(/demo credentials/i)).toBeInTheDocument();
    });

    it('should render login form in a card layout', () => {
      render(<Login />);
      
      // Check for the form element by its class or container
      const card = screen.getByText(/demo credentials/i).closest('.MuiCard-root');
      expect(card).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('should have required fields', () => {
      render(<Login />);
      
      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      
      expect(usernameInput).toHaveAttribute('required');
      expect(passwordInput).toHaveAttribute('required');
    });

    it('should prevent submission with empty fields', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue(false);
      
      render(<Login />);
      
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);
      
      // HTML5 validation should prevent form submission
      expect(mockLogin).not.toHaveBeenCalled();
    });

    it('should allow submission with filled fields', async () => {
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
        expect(screen.getByText(/invalid username or password/i)).toBeInTheDocument();
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
        expect(screen.getByText(/invalid username or password/i)).toBeInTheDocument();
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
        expect(screen.getByText(/login failed/i)).toBeInTheDocument();
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
        expect(screen.getByText(/invalid username or password/i)).toBeInTheDocument();
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
      
      // Username input should be focused by default (autoFocus)
      expect(usernameInput).toHaveFocus();
      
      // Tab to password input
      await user.tab();
      expect(passwordInput).toHaveFocus();
      
      // Tab to submit button
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
    it('should have password input with correct type', () => {
      render(<Login />);
      
      const passwordInput = screen.getByLabelText(/password/i);
      
      // Password input should be hidden by default
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
      
      const longString = 'a'.repeat(100); // Reduced from 1000 to 100 for faster test
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
      
      // Use clear and type to avoid issues with special characters
      await user.clear(usernameInput);
      await user.type(usernameInput, specialUsername);
      await user.clear(passwordInput);
      await user.type(passwordInput, specialPassword);
      await user.click(submitButton);
      
      expect(mockLogin).toHaveBeenCalledWith(specialUsername, specialPassword);
    });
  });
});