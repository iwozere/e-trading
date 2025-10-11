/**
 * Telegram Bot Dashboard Page
 * 
 * Main dashboard for Telegram bot management showing statistics,
 * pending approvals, and recent activity.
 */

import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Chip,
  LinearProgress
} from '@mui/material';
import {
  People as PeopleIcon,
  NotificationsActive as AlertIcon,
  Schedule as ScheduleIcon,
  Assessment as StatsIcon
} from '@mui/icons-material';
import TelegramBreadcrumbs from '../../components/Telegram/TelegramBreadcrumbs';
import { useTelegramDashboardStats } from '../../hooks/telegram/useTelegramStats';

const TelegramDashboard: React.FC = () => {
  const {
    totalUsers,
    activeAlerts,
    activeSchedules,
    commandsToday,
    approvalRate,
    systemHealth,
    isLoading,
    isError,
    error
  } = useTelegramDashboardStats();

  if (isError) {
    return (
      <Box sx={{ p: 3 }}>
        <TelegramBreadcrumbs />
        <Typography variant="h4" gutterBottom>
          Telegram Bot Dashboard
        </Typography>
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to load dashboard data: {error?.message || 'Unknown error'}
        </Alert>
      </Box>
    );
  }

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'good': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <TelegramBreadcrumbs />
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" gutterBottom>
          Telegram Bot Dashboard
        </Typography>
        <Chip 
          label={`System Health: ${systemHealth.toUpperCase()}`}
          color={getHealthColor(systemHealth) as any}
          variant="outlined"
        />
      </Box>
      
      {isLoading ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
          <CircularProgress />
          <Typography variant="body1" sx={{ ml: 2 }}>
            Loading dashboard...
          </Typography>
        </Box>
      ) : (
        <Grid container spacing={3}>
          {/* User Statistics */}
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <PeopleIcon color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Users</Typography>
                </Box>
                <Typography variant="h4" color="primary">
                  {totalUsers}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total registered users
                </Typography>
                <Box mt={1}>
                  <Typography variant="caption" color="text.secondary">
                    Approval Rate: {approvalRate}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={approvalRate} 
                    sx={{ mt: 0.5 }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Alert Statistics */}
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <AlertIcon color="warning" sx={{ mr: 1 }} />
                  <Typography variant="h6">Alerts</Typography>
                </Box>
                <Typography variant="h4" color="warning.main">
                  {activeAlerts}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Active alerts
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Schedule Statistics */}
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <ScheduleIcon color="info" sx={{ mr: 1 }} />
                  <Typography variant="h6">Schedules</Typography>
                </Box>
                <Typography variant="h4" color="info.main">
                  {activeSchedules}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Active schedules
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Command Statistics */}
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <StatsIcon color="success" sx={{ mr: 1 }} />
                  <Typography variant="h6">Commands</Typography>
                </Box>
                <Typography variant="h4" color="success.main">
                  {commandsToday}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Commands today
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Quick Actions */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Quick Actions
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Use the navigation menu to manage users, alerts, schedules, and view audit logs.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default TelegramDashboard;