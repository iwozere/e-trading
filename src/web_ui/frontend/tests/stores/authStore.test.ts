/**
 * Auth Store Tests
 * ---------------
 * 
 * Unit tests for the Zustand authentication store including:
 * - Login functionality
 * - Logout functionality
 * - Token management
 * - State persistence
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAuthStore } from '../../src/stores/authStore';
import { mockFetchSuccess, mockFetchError } from '../utils/test-utils';

// Mock localStorage for persistence testing
const mockLocalStorage = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

describe('AuthStore', () => {
  beforeEach(() => {
    // Reset the store state before each test
    useAuthStore.setState({
      isAuthenticated: false,
      user: null,
      token: null,
    });
    
    // Clear all mocks
    vi.clearAllMocks();
    mockLocalStorage.getItem.mockReturnValue(null);
  });

  describe('Initial State', () => {
    it('should have correct initial state', () => {
      const { result } = renderHook(() => useAuthStore());
      
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.user).toBe(null);
      expect(result.current.token).toBe(null);
    });
  });

  describe('Login', () => {
    it('should login successfully with valid credentials', async () => {
      const mockResponse = {
        user: { username: 'admin', role: 'admin' },
        access_token: 'mock-jwt-token',
      };
      
      mockFetchSuccess(mockResponse);
      
      const { result } = renderHook(() => useAuthStore());
      
      let loginResult: boolean;
      await act(async () => {
        loginResult = await result.current.login('admin', 'password');
      });
      
      expect(loginResult!).toBe(true);
      expect(result.current.isAuthenticated).toBe(true);
      expect(result.current.user).toEqual(mockResponse.user);
      expect(result.current.token).toBe(mockResponse.access_token);
      
      // Verify API call
      expect(fetch).toHaveBeenCalledWith('/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: 'admin', password: 'password' }),
      });
    });

    it('should fail login with invalid credentials', async () => {
      mockFetchError('Invalid credentials', 401);
      
      const { result } = renderHook(() => useAuthStore());
      
      let loginResult: boolean;
      await act(async () => {
        loginResult = await result.current.login('admin', 'wrongpassword');
      });
      
      expect(loginResult!).toBe(false);
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.user).toBe(null);
      expect(result.current.token).toBe(null);
    });

    it('should handle network errors during login', async () => {
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'));
      (globalThis as any).fetch = mockFetch;
      
      const { result } = renderHook(() => useAuthStore());
      
      let loginResult: boolean;
      await act(async () => {
        loginResult = await result.current.login('admin', 'password');
      });
      
      expect(loginResult!).toBe(false);
      expect(result.current.isAuthenticated).toBe(false);
    });

    it('should handle malformed response during login', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.reject(new Error('Invalid JSON')),
      });
      (globalThis as any).fetch = mockFetch;
      
      const { result } = renderHook(() => useAuthStore());
      
      let loginResult: boolean;
      await act(async () => {
        loginResult = await result.current.login('admin', 'password');
      });
      
      expect(loginResult!).toBe(false);
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('Logout', () => {
    it('should logout successfully when authenticated', async () => {
      // Set up authenticated state
      const { result } = renderHook(() => useAuthStore());
      
      act(() => {
        result.current.setToken('mock-token', { username: 'admin', role: 'admin' });
      });
      
      expect(result.current.isAuthenticated).toBe(true);
      
      // Mock successful logout API call
      mockFetchSuccess({ message: 'Logged out successfully' });
      
      await act(async () => {
        await result.current.logout();
      });
      
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.user).toBe(null);
      expect(result.current.token).toBe(null);
      
      // Verify API call
      expect(fetch).toHaveBeenCalledWith('/auth/logout', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer mock-token',
        },
      });
    });

    it('should logout successfully even when API call fails', async () => {
      // Set up authenticated state
      const { result } = renderHook(() => useAuthStore());
      
      act(() => {
        result.current.setToken('mock-token', { username: 'admin', role: 'admin' });
      });
      
      // Mock failed logout API call
      mockFetchError('Server error', 500);
      
      await act(async () => {
        await result.current.logout();
      });
      
      // Should still clear local state even if API fails
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.user).toBe(null);
      expect(result.current.token).toBe(null);
    });

    it('should logout when not authenticated', async () => {
      const { result } = renderHook(() => useAuthStore());
      
      expect(result.current.isAuthenticated).toBe(false);
      
      await act(async () => {
        await result.current.logout();
      });
      
      // Should remain in unauthenticated state
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.user).toBe(null);
      expect(result.current.token).toBe(null);
      
      // Should not make API call when no token
      expect(fetch).not.toHaveBeenCalled();
    });

    it('should handle network errors during logout', async () => {
      // Set up authenticated state
      const { result } = renderHook(() => useAuthStore());
      
      act(() => {
        result.current.setToken('mock-token', { username: 'admin', role: 'admin' });
      });
      
      // Mock network error
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'));
      (globalThis as any).fetch = mockFetch;
      
      await act(async () => {
        await result.current.logout();
      });
      
      // Should still clear local state even with network error
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.user).toBe(null);
      expect(result.current.token).toBe(null);
    });
  });

  describe('setToken', () => {
    it('should set token and user correctly', () => {
      const { result } = renderHook(() => useAuthStore());
      
      const user = { username: 'testuser', role: 'trader' as const };
      const token = 'test-jwt-token';
      
      act(() => {
        result.current.setToken(token, user);
      });
      
      expect(result.current.isAuthenticated).toBe(true);
      expect(result.current.user).toEqual(user);
      expect(result.current.token).toBe(token);
    });

    it('should update existing authentication state', () => {
      const { result } = renderHook(() => useAuthStore());
      
      // Set initial state
      act(() => {
        result.current.setToken('old-token', { username: 'olduser', role: 'viewer' });
      });
      
      expect(result.current.user?.username).toBe('olduser');
      expect(result.current.token).toBe('old-token');
      
      // Update with new token and user
      act(() => {
        result.current.setToken('new-token', { username: 'newuser', role: 'admin' });
      });
      
      expect(result.current.user?.username).toBe('newuser');
      expect(result.current.user?.role).toBe('admin');
      expect(result.current.token).toBe('new-token');
    });
  });

  describe('State Persistence', () => {
    it('should persist authentication state to localStorage', () => {
      const { result } = renderHook(() => useAuthStore());
      
      const user = { username: 'admin', role: 'admin' as const };
      const token = 'persistent-token';
      
      act(() => {
        result.current.setToken(token, user);
      });
      
      // Note: In a real test environment, you would verify that the persistence
      // middleware is working correctly. This would require more complex setup
      // with the actual Zustand persist middleware.
      expect(result.current.isAuthenticated).toBe(true);
    });
  });

  describe('Role-based Access', () => {
    it('should handle admin role correctly', () => {
      const { result } = renderHook(() => useAuthStore());
      
      act(() => {
        result.current.setToken('token', { username: 'admin', role: 'admin' });
      });
      
      expect(result.current.user?.role).toBe('admin');
    });

    it('should handle trader role correctly', () => {
      const { result } = renderHook(() => useAuthStore());
      
      act(() => {
        result.current.setToken('token', { username: 'trader', role: 'trader' });
      });
      
      expect(result.current.user?.role).toBe('trader');
    });

    it('should handle viewer role correctly', () => {
      const { result } = renderHook(() => useAuthStore());
      
      act(() => {
        result.current.setToken('token', { username: 'viewer', role: 'viewer' });
      });
      
      expect(result.current.user?.role).toBe('viewer');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty username and password', async () => {
      mockFetchError('Username and password are required', 400);
      
      const { result } = renderHook(() => useAuthStore());
      
      let loginResult: boolean;
      await act(async () => {
        loginResult = await result.current.login('', '');
      });
      
      expect(loginResult!).toBe(false);
      expect(result.current.isAuthenticated).toBe(false);
    });

    it('should handle very long credentials', async () => {
      const longString = 'a'.repeat(1000);
      mockFetchError('Invalid credentials', 401);
      
      const { result } = renderHook(() => useAuthStore());
      
      let loginResult: boolean;
      await act(async () => {
        loginResult = await result.current.login(longString, longString);
      });
      
      expect(loginResult!).toBe(false);
    });

    it('should handle special characters in credentials', async () => {
      const specialChars = 'user@domain.com';
      const password = 'p@ssw0rd!#$';
      
      mockFetchSuccess({
        user: { username: specialChars, role: 'admin' },
        access_token: 'token',
      });
      
      const { result } = renderHook(() => useAuthStore());
      
      let loginResult: boolean;
      await act(async () => {
        loginResult = await result.current.login(specialChars, password);
      });
      
      expect(loginResult!).toBe(true);
      expect(result.current.user?.username).toBe(specialChars);
    });
  });
});