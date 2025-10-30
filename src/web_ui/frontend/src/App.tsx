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
import Login from './pages/Auth/Login';

// Telegram Bot Management Components
import {
  TelegramDashboard,
  UserManagement,
  AlertManagement,
  ScheduleManagement,
  BroadcastCenter,
  AuditLogs
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
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          border: '1px solid rgba(255, 255, 255, 0.12)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
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