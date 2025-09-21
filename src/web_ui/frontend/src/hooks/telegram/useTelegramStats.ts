/**
 * React Query hooks for Telegram statistics management
 * 
 * This module provides custom hooks for managing all Telegram bot statistics,
 * including user stats, alert stats, schedule stats, and audit stats
 * with real-time updates and WebSocket integration.
 */

import React from 'react';
import { useQuery, useQueryClient, UseQueryOptions } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';
import { telegramApi } from '../../api/telegramApi';
import {
  UserStats,
  AlertStats,
  ScheduleStats,
  AuditStats,
  TelegramApiError
} from '../../types/telegram';

// Import query keys from other hooks
import { telegramUserKeys } from './useTelegramUsers';
import { telegramAlertKeys } from './useTelegramAlerts';
import { telegramScheduleKeys } from './useTelegramSchedules';
import { telegramAuditKeys } from './useTelegramAudit';

// ============================================================================
// Combined Statistics Interface
// ============================================================================

/**
 * Combined statistics from all Telegram bot services
 */
export interface TelegramCombinedStats {
  users: UserStats;
  alerts: AlertStats;
  schedules: ScheduleStats;
  audit: AuditStats;
}

/**
 * Real-time statistics update interface
 */
export interface TelegramStatsUpdate {
  type: 'user' | 'alert' | 'schedule' | 'audit';
  data: UserStats | AlertStats | ScheduleStats | AuditStats;
  timestamp: string;
}

// ============================================================================
// Combined Statistics Hooks
// ============================================================================

/**
 * Hook for fetching all Telegram bot statistics
 */
export const useTelegramCombinedStats = (
  options?: Omit<UseQueryOptions<TelegramCombinedStats, TelegramApiError>, 'queryKey' | 'queryFn'>
) => {
  return useQuery({
    queryKey: ['telegram-combined-stats'],
    queryFn: async (): Promise<TelegramCombinedStats> => {
      const [users, alerts, schedules, audit] = await Promise.all([
        telegramApi.getUserStats(),
        telegramApi.getAlertStats(),
        telegramApi.getScheduleStats(),
        telegramApi.getAuditStats(),
      ]);

      return { users, alerts, schedules, audit };
    },
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
    refetchInterval: 60000, // Refetch every minute for real-time stats
    onError: (error: TelegramApiError) => {
      console.error('Failed to fetch combined Telegram statistics:', error);
      toast.error(`Failed to load statistics: ${error.message}`);
    },
    ...options,
  });
};

/**
 * Hook for individual statistics with automatic refresh
 */
export const useTelegramStatsWithRefresh = () => {
  const userStats = useQuery({
    queryKey: telegramUserKeys.stats(),
    queryFn: () => telegramApi.getUserStats(),
    staleTime: 30000,
    refetchInterval: 60000,
  });

  const alertStats = useQuery({
    queryKey: telegramAlertKeys.stats(),
    queryFn: () => telegramApi.getAlertStats(),
    staleTime: 30000,
    refetchInterval: 60000,
  });

  const scheduleStats = useQuery({
    queryKey: telegramScheduleKeys.stats(),
    queryFn: () => telegramApi.getScheduleStats(),
    staleTime: 30000,
    refetchInterval: 60000,
  });

  const auditStats = useQuery({
    queryKey: telegramAuditKeys.stats(),
    queryFn: () => telegramApi.getAuditStats(),
    staleTime: 30000,
    refetchInterval: 60000,
  });

  const isLoading = userStats.isLoading || alertStats.isLoading || 
                   scheduleStats.isLoading || auditStats.isLoading;
  
  const isError = userStats.isError || alertStats.isError || 
                 scheduleStats.isError || auditStats.isError;

  const error = userStats.error || alertStats.error || 
               scheduleStats.error || auditStats.error;

  return {
    userStats: userStats.data,
    alertStats: alertStats.data,
    scheduleStats: scheduleStats.data,
    auditStats: auditStats.data,
    isLoading,
    isError,
    error,
    refetch: () => {
      userStats.refetch();
      alertStats.refetch();
      scheduleStats.refetch();
      auditStats.refetch();
    },
  };
};

// ============================================================================
// Real-time Statistics Hooks
// ============================================================================

/**
 * Hook for managing real-time statistics updates via WebSocket
 */
export const useRealtimeTelegramStats = () => {
  const queryClient = useQueryClient();
  const [lastUpdate, setLastUpdate] = React.useState<Date | null>(null);
  const [connectionStatus, setConnectionStatus] = React.useState<'connected' | 'disconnected' | 'connecting'>('disconnected');

  // Update user statistics
  const updateUserStats = React.useCallback((newStats: UserStats) => {
    queryClient.setQueryData(telegramUserKeys.stats(), newStats);
    queryClient.invalidateQueries({ queryKey: ['telegram-combined-stats'] });
    setLastUpdate(new Date());
  }, [queryClient]);

  // Update alert statistics
  const updateAlertStats = React.useCallback((newStats: AlertStats) => {
    queryClient.setQueryData(telegramAlertKeys.stats(), newStats);
    queryClient.invalidateQueries({ queryKey: ['telegram-combined-stats'] });
    setLastUpdate(new Date());
  }, [queryClient]);

  // Update schedule statistics
  const updateScheduleStats = React.useCallback((newStats: ScheduleStats) => {
    queryClient.setQueryData(telegramScheduleKeys.stats(), newStats);
    queryClient.invalidateQueries({ queryKey: ['telegram-combined-stats'] });
    setLastUpdate(new Date());
  }, [queryClient]);

  // Update audit statistics
  const updateAuditStats = React.useCallback((newStats: AuditStats) => {
    queryClient.setQueryData(telegramAuditKeys.stats(), newStats);
    queryClient.invalidateQueries({ queryKey: ['telegram-combined-stats'] });
    setLastUpdate(new Date());
  }, [queryClient]);

  // Handle generic statistics update
  const handleStatsUpdate = React.useCallback((update: TelegramStatsUpdate) => {
    switch (update.type) {
      case 'user':
        updateUserStats(update.data as UserStats);
        break;
      case 'alert':
        updateAlertStats(update.data as AlertStats);
        break;
      case 'schedule':
        updateScheduleStats(update.data as ScheduleStats);
        break;
      case 'audit':
        updateAuditStats(update.data as AuditStats);
        break;
    }
  }, [updateUserStats, updateAlertStats, updateScheduleStats, updateAuditStats]);

  return {
    lastUpdate,
    connectionStatus,
    setConnectionStatus,
    updateUserStats,
    updateAlertStats,
    updateScheduleStats,
    updateAuditStats,
    handleStatsUpdate,
  };
};

// ============================================================================
// Statistics Dashboard Hooks
// ============================================================================

/**
 * Hook for dashboard statistics with calculated metrics
 */
export const useTelegramDashboardStats = () => {
  const combinedStats = useTelegramCombinedStats();

  const dashboardMetrics = React.useMemo(() => {
    if (!combinedStats.data) {
      return {
        totalUsers: 0,
        activeAlerts: 0,
        activeSchedules: 0,
        commandsToday: 0,
        approvalRate: 0,
        alertSuccessRate: 0,
        scheduleSuccessRate: 0,
        systemHealth: 'unknown' as 'good' | 'warning' | 'error' | 'unknown',
      };
    }

    const { users, alerts, schedules, audit } = combinedStats.data;

    // Calculate approval rate
    const approvalRate = users.total_users > 0 
      ? (users.approved_users / users.total_users) * 100 
      : 0;

    // Calculate alert success rate (assuming triggered alerts are successful)
    const alertSuccessRate = alerts.total_alerts > 0 
      ? ((alerts.total_alerts - alerts.triggered_today) / alerts.total_alerts) * 100 
      : 100;

    // Calculate schedule success rate
    const scheduleSuccessRate = schedules.total_schedules > 0 
      ? ((schedules.executed_today - schedules.failed_executions) / schedules.executed_today) * 100 
      : 100;

    // Determine system health
    let systemHealth: 'good' | 'warning' | 'error' | 'unknown' = 'good';
    
    if (schedules.failed_executions > schedules.executed_today * 0.1) {
      systemHealth = 'error'; // More than 10% schedule failures
    } else if (audit.failed_commands > audit.total_commands * 0.05) {
      systemHealth = 'warning'; // More than 5% command failures
    } else if (users.pending_approvals > users.total_users * 0.2) {
      systemHealth = 'warning'; // More than 20% pending approvals
    }

    return {
      totalUsers: users.total_users,
      activeAlerts: alerts.active_alerts,
      activeSchedules: schedules.active_schedules,
      commandsToday: audit.recent_activity_24h,
      approvalRate: Math.round(approvalRate * 100) / 100,
      alertSuccessRate: Math.round(alertSuccessRate * 100) / 100,
      scheduleSuccessRate: Math.round(scheduleSuccessRate * 100) / 100,
      systemHealth,
    };
  }, [combinedStats.data]);

  return {
    ...dashboardMetrics,
    isLoading: combinedStats.isLoading,
    isError: combinedStats.isError,
    error: combinedStats.error,
    refetch: combinedStats.refetch,
  };
};

/**
 * Hook for statistics comparison over time
 */
export const useTelegramStatsComparison = (timeRange: '24h' | '7d' | '30d' = '24h') => {
  const currentStats = useTelegramCombinedStats();
  
  // This would typically fetch historical data from the backend
  // For now, we'll simulate comparison data
  const comparison = React.useMemo(() => {
    if (!currentStats.data) {
      return {
        userGrowth: 0,
        alertGrowth: 0,
        scheduleGrowth: 0,
        commandGrowth: 0,
      };
    }

    // Simulate growth percentages (in a real implementation, 
    // this would compare with historical data)
    return {
      userGrowth: Math.random() * 20 - 10, // -10% to +10%
      alertGrowth: Math.random() * 30 - 15, // -15% to +15%
      scheduleGrowth: Math.random() * 25 - 12.5, // -12.5% to +12.5%
      commandGrowth: Math.random() * 40 - 20, // -20% to +20%
    };
  }, [currentStats.data, timeRange]);

  return {
    ...comparison,
    isLoading: currentStats.isLoading,
    isError: currentStats.isError,
    error: currentStats.error,
  };
};

// ============================================================================
// Statistics Export Hooks
// ============================================================================

/**
 * Hook for exporting statistics data
 */
export const useTelegramStatsExport = () => {
  const combinedStats = useTelegramCombinedStats();

  const exportToCSV = React.useCallback(() => {
    if (!combinedStats.data) {
      toast.error('No statistics data available to export');
      return;
    }

    const { users, alerts, schedules, audit } = combinedStats.data;
    
    const csvData = [
      ['Metric', 'Value'],
      ['Total Users', users.total_users.toString()],
      ['Verified Users', users.verified_users.toString()],
      ['Approved Users', users.approved_users.toString()],
      ['Pending Approvals', users.pending_approvals.toString()],
      ['Admin Users', users.admin_users.toString()],
      ['Total Alerts', alerts.total_alerts.toString()],
      ['Active Alerts', alerts.active_alerts.toString()],
      ['Alerts Triggered Today', alerts.triggered_today.toString()],
      ['Alert Rearm Cycles', alerts.rearm_cycles.toString()],
      ['Total Schedules', schedules.total_schedules.toString()],
      ['Active Schedules', schedules.active_schedules.toString()],
      ['Schedules Executed Today', schedules.executed_today.toString()],
      ['Failed Schedule Executions', schedules.failed_executions.toString()],
      ['Total Commands', audit.total_commands.toString()],
      ['Successful Commands', audit.successful_commands.toString()],
      ['Failed Commands', audit.failed_commands.toString()],
      ['Recent Activity (24h)', audit.recent_activity_24h.toString()],
    ];

    const csvContent = csvData.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `telegram-bot-stats-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);

    toast.success('Statistics exported successfully');
  }, [combinedStats.data]);

  const exportToJSON = React.useCallback(() => {
    if (!combinedStats.data) {
      toast.error('No statistics data available to export');
      return;
    }

    const jsonData = {
      exportDate: new Date().toISOString(),
      statistics: combinedStats.data,
    };

    const jsonContent = JSON.stringify(jsonData, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `telegram-bot-stats-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);

    toast.success('Statistics exported successfully');
  }, [combinedStats.data]);

  return {
    exportToCSV,
    exportToJSON,
    canExport: !!combinedStats.data,
    isLoading: combinedStats.isLoading,
  };
};
