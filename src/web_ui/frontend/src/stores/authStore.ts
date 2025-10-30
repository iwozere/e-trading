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
          // Call the actual authentication API
          const response = await fetch('/auth/login', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
          });

          if (response.ok) {
            const data = await response.json();
            
            set({
              isAuthenticated: true,
              user: data.user,
              token: data.access_token,
            });
            
            return true;
          } else {
            console.error('Login failed:', response.status, response.statusText);
            return false;
          }
        } catch (error) {
          console.error('Login error:', error);
          return false;
        }
      },

      logout: async () => {
        try {
          // Call logout API if we have a token
          const currentState = get();
          if (currentState.token) {
            await fetch('/auth/logout', {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${currentState.token}`,
              },
            });
          }
        } catch (error) {
          console.error('Logout error:', error);
        } finally {
          // Always clear local state
          set({
            isAuthenticated: false,
            user: null,
            token: null,
          });
        }
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