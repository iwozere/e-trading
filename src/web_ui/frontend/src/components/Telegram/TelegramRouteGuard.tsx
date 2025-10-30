/**
 * Telegram Route Guard Component
 * 
 * Protects Telegram bot management routes based on user permissions.
 * Redirects unauthorized users to appropriate pages.
 */

import React from 'react';
import { Navigate } from 'react-router-dom';
import { Box, Typography, Card, CardContent, Alert } from '@mui/material';
import { useAuthStore } from '../../stores/authStore';

interface TelegramRouteGuardProps {
  children: React.ReactNode;
  requiredPermission?: 'view' | 'manage_users' | 'manage_alerts' | 'send_broadcasts' | 'view_audit_logs';
}

/**
 * Hook to check Telegram bot permissions
 * TODO: This will be implemented when the backend permission system is ready
 */
const useTelegramPermissions = () => {
  const { user } = useAuthStore();
  
  // For now, allow admin users to access all Telegram features
  // This will be expanded when the backend implements proper permission checking
  const hasPermission = (permission: string): boolean => {
    if (!user) return false;
    
    // Admin users have all permissions
    if (user.role === 'admin') return true;
    
    // Traders can view but not manage
    if (user.role === 'trader') {
      return permission === 'view' || permission === 'view_audit_logs';
    }
    
    // Viewers can only view basic information
    if (user.role === 'viewer') {
      return permission === 'view';
    }
    
    return false;
  };

  return {
    canView: hasPermission('view'),
    canManageUsers: hasPermission('manage_users'),
    canManageAlerts: hasPermission('manage_alerts'),
    canSendBroadcasts: hasPermission('send_broadcasts'),
    canViewAuditLogs: hasPermission('view_audit_logs'),
    hasPermission
  };
};

/**
 * Route guard component for Telegram bot management pages
 */
const TelegramRouteGuard: React.FC<TelegramRouteGuardProps> = ({ 
  children, 
  requiredPermission = 'view' 
}) => {
  const { isAuthenticated } = useAuthStore();
  const { hasPermission } = useTelegramPermissions();

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // Check if user has required permission
  if (!hasPermission(requiredPermission)) {
    return (
      <Box p={3}>
        <Card>
          <CardContent>
            <Alert severity="warning">
              <Typography variant="h6" gutterBottom>
                Access Denied
              </Typography>
              <Typography variant="body1">
                You don't have permission to access this Telegram bot management feature.
                Please contact your administrator if you need access.
              </Typography>
            </Alert>
          </CardContent>
        </Card>
      </Box>
    );
  }

  return <>{children}</>;
};

export default TelegramRouteGuard;
export { useTelegramPermissions };