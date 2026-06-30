import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { Toaster } from 'react-hot-toast';

// Components
import Layout from './components/Layout/Layout';
import Dashboard from './pages/Dashboard/Dashboard';
import Strategies from './pages/Strategies/Strategies';
import StrategyForm from './pages/Strategies/StrategyForm';
import Monitoring from './pages/Monitoring/Monitoring';
import Analytics from './pages/Analytics/Analytics';
import Administration from './pages/Administration/Administration';
import Messages from './pages/Messages/Messages';
import Login from './pages/Auth/Login';

// Telegram Bot Management Components
import {
  TelegramDashboard,
  UserManagement,
  AlertManagement,
  ScheduleManagement,
  BroadcastCenter,
  AuditLogs,
  ConfigBuilderPage
} from './pages/Telegram';
import { TelegramRouteGuard } from './components/Telegram';

// Hooks and Context
import { useAuthStore } from './stores/authStore';
import { WebSocketProvider } from './contexts/WebSocketContext';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: 'hsl(190, 100%, 50%)', // Electric Cyan
      light: 'hsl(190, 100%, 70%)',
      dark: 'hsl(190, 100%, 30%)',
    },
    secondary: {
      main: 'hsl(280, 100%, 65%)', // Vibrant Purple
    },
    background: {
      default: 'hsl(230, 60%, 5%)', // Deep Space
      paper: 'hsl(230, 50%, 10%)',
    },
    success: {
      main: 'hsl(145, 100%, 45%)', // Neon Green
    },
    divider: 'rgba(255, 255, 255, 0.08)',
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: { fontFamily: 'Outfit, sans-serif', fontWeight: 800 },
    h2: { fontFamily: 'Outfit, sans-serif', fontWeight: 700 },
    h3: { fontFamily: 'Outfit, sans-serif', fontWeight: 700 },
    h4: { fontFamily: 'Outfit, sans-serif', fontWeight: 600, letterSpacing: '-0.02em' },
    h5: { fontFamily: 'Outfit, sans-serif', fontWeight: 600, letterSpacing: '-0.01em' },
    h6: { fontFamily: 'Outfit, sans-serif', fontWeight: 600 },
    button: { fontFamily: 'Outfit, sans-serif', fontWeight: 600, textTransform: 'none' },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: "hsl(230, 50%, 20%) transparent",
          "&::-webkit-scrollbar": { width: 8 },
          "&::-webkit-scrollbar-thumb": { backgroundColor: "hsl(230, 50%, 20%)", borderRadius: 8 },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(10, 14, 39, 0.7)',
          backdropFilter: 'blur(12px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          backgroundImage: 'none',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: 'hsl(230, 60%, 5%)',
          borderRight: '1px solid rgba(255, 255, 255, 0.08)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(26, 29, 58, 0.4)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
          transition: 'transform 0.2s ease-in-out, border-color 0.2s ease-in-out',
          '&:hover': {
            borderColor: 'rgba(255, 255, 255, 0.2)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '8px 16px',
        },
        containedPrimary: {
          boxShadow: '0 0 15px rgba(0, 229, 255, 0.3)',
          '&:hover': {
            boxShadow: '0 0 25px rgba(0, 229, 255, 0.5)',
          },
        },
      },
    },
  },
});

// Protected Route Component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuthStore();
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
};

// Main App Component
const App: React.FC = () => {
  const { isAuthenticated } = useAuthStore();

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
          <Router>
            {isAuthenticated ? (
              // Temporarily disable WebSocket until backend implements it
              // <WebSocketProvider>
                <Layout>
                  <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route
                      path="/dashboard"
                      element={
                        <ProtectedRoute>
                          <Dashboard />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/strategies"
                      element={
                        <ProtectedRoute>
                          <Strategies />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/strategies/new"
                      element={
                        <ProtectedRoute>
                          <StrategyForm />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/strategies/:id/edit"
                      element={
                        <ProtectedRoute>
                          <StrategyForm />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/monitoring"
                      element={
                        <ProtectedRoute>
                          <Monitoring />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/analytics"
                      element={
                        <ProtectedRoute>
                          <Analytics />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/administration"
                      element={
                        <ProtectedRoute>
                          <Administration />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/messages"
                      element={
                        <ProtectedRoute>
                          <Messages />
                        </ProtectedRoute>
                      }
                    />

                    {/* Telegram Bot Management Routes */}
                    <Route
                      path="/telegram"
                      element={
                        <ProtectedRoute>
                          <TelegramRouteGuard>
                            <TelegramDashboard />
                          </TelegramRouteGuard>
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/telegram/dashboard"
                      element={
                        <ProtectedRoute>
                          <TelegramRouteGuard>
                            <TelegramDashboard />
                          </TelegramRouteGuard>
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/telegram/users"
                      element={
                        <ProtectedRoute>
                          <TelegramRouteGuard requiredPermission="manage_users">
                            <UserManagement />
                          </TelegramRouteGuard>
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/telegram/alerts"
                      element={
                        <ProtectedRoute>
                          <TelegramRouteGuard requiredPermission="manage_alerts">
                            <AlertManagement />
                          </TelegramRouteGuard>
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/telegram/schedules"
                      element={
                        <ProtectedRoute>
                          <TelegramRouteGuard requiredPermission="manage_alerts">
                            <ScheduleManagement />
                          </TelegramRouteGuard>
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/telegram/broadcast"
                      element={
                        <ProtectedRoute>
                          <TelegramRouteGuard requiredPermission="send_broadcasts">
                            <BroadcastCenter />
                          </TelegramRouteGuard>
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/telegram/audit"
                      element={
                        <ProtectedRoute>
                          <TelegramRouteGuard requiredPermission="view_audit_logs">
                            <AuditLogs />
                          </TelegramRouteGuard>
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/telegram/builder"
                      element={
                        <ProtectedRoute>
                          <TelegramRouteGuard requiredPermission="manage_alerts">
                            <ConfigBuilderPage />
                          </TelegramRouteGuard>
                        </ProtectedRoute>
                      }
                    />
                    
                    <Route path="*" element={<Navigate to="/dashboard" replace />} />
                  </Routes>
                </Layout>
              // </WebSocketProvider>
            ) : (
              <Routes>
                <Route path="/login" element={<Login />} />
                <Route path="*" element={<Navigate to="/login" replace />} />
              </Routes>
            )}
          </Router>
        </Box>
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: theme.palette.background.paper,
              color: theme.palette.text.primary,
              border: `1px solid ${theme.palette.divider}`,
            },
          }}
        />
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default App;