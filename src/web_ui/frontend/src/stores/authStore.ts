import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  username: string;
  role: 'admin' | 'trader' | 'viewer';
}

interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  setToken: (token: string, user: User) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      isAuthenticated: false,
      user: null,
      token: null,

      login: async (username: string, password: string) => {
        try {
          // TODO: Replace with actual API call
          // For now, accept any credentials for development
          if (username && password) {
            const mockUser: User = {
              username,
              role: username === 'admin' ? 'admin' : 'trader',
            };
            
            const mockToken = `mock-token-${Date.now()}`;
            
            set({
              isAuthenticated: true,
              user: mockUser,
              token: mockToken,
            });
            
            return true;
          }
          
          return false;
        } catch (error) {
          console.error('Login error:', error);
          return false;
        }
      },

      logout: () => {
        set({
          isAuthenticated: false,
          user: null,
          token: null,
        });
      },

      setToken: (token: string, user: User) => {
        set({
          isAuthenticated: true,
          user,
          token,
        });
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        isAuthenticated: state.isAuthenticated,
        user: state.user,
        token: state.token,
      }),
    }
  )
);