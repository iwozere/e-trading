/**
 * React Query hooks for Telegram audit log management
 * 
 * This module provides custom hooks for managing Telegram bot audit logs,
 * including fetching command history, filtering, and infinite scroll
 * with real-time updates and proper error handling.
 */

import React from 'react';
import { useQuery, useInfiniteQuery, useQueryClient, UseQueryOptions, UseInfiniteQueryOptions } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';
import { telegramApi } from '../../api/telegramApi';
import {
  CommandAudit,
  AuditLogParams,
  AuditStats,
  TelegramApiError
} from '../../types/telegram';

// ============================================================================
// Query Keys
// ============================================================================

/**
 * Query key factory for Telegram audit-related queries
 */
export const telegramAuditKeys = {
  all: ['telegram-audit'] as const,
  logs: () => [...telegramAuditKeys.all, 'logs'] as const,
  logsList: (params: AuditLogParams) => [...telegramAuditKeys.logs(), params] as const,
  userLogs: (userId: string) => [...telegramAuditKeys.all, 'user-logs', userId] as const,
  userLogsPaginated: (userId: string, page: number, limit: number) => 
    [...telegramAuditKeys.userLogs(userId), page, limit] as const,
  stats: () => [...telegramAuditKeys.all, 'stats'] as const,
};

// ============================================================================
// Audit Log Queries
// ============================================================================

/**
 * Hook for fetching audit logs with filtering and pagination
 */
export const useTelegramAuditLogs = (
  params?: AuditLogParams,
  options?: Omit<UseQueryOptions<any, TelegramApiError>, 'queryKey' | 'queryFn'>
) => {
  return useQuery({
    queryKey: telegramAuditKeys.logsList(params || {}),
    queryFn: () => telegramApi.getAuditLogs(params),
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
    keepPreviousData: true, // Keep previous data while fetching new page
    onError: (error: TelegramApiError) => {
      console.error('Failed to fetch audit logs:', error);
      toast.error(`Failed to load audit logs: ${error.message}`);
    },
    ...options,
  });
};

/**
 * Hook for fetching audit logs with infinite scroll
 */
export const useTelegramAuditLogsInfinite = (
  params?: Omit<AuditLogParams, 'page' | 'limit'>,
  options?: Omit<UseInfiniteQueryOptions<any, TelegramApiError>, 'queryKey' | 'queryFn' | 'getNextPageParam'>
) => {
  return useInfiniteQuery({
    queryKey: telegramAuditKeys.logsList({ ...params, page: 0, limit: 50 }),
    queryFn: ({ pageParam = 1 }) => 
      telegramApi.getAuditLogs({ ...params, page: pageParam, limit: 50 }),
    getNextPageParam: (lastPage, pages) => {
      if (!lastPage.has_more) return undefined;
      return pages.length + 1;
    },
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
    onError: (error: TelegramApiError) => {
      console.error('Failed to fetch audit logs:', error);
      toast.error(`Failed to load audit logs: ${error.message}`);
    },
    ...options,
  });
};

/**
 * Hook for fetching audit logs for a specific user
 */
export const useUserAuditLogs = (
  userId: string,
  page = 1,
  limit = 50,
  options?: Omit<UseQueryOptions<any, TelegramApiError>, 'queryKey' | 'queryFn'>
) => {
  return useQuery({
    queryKey: telegramAuditKeys.userLogsPaginated(userId, page, limit),
    queryFn: () => telegramApi.getUserAuditLogs(userId, page, limit),
    enabled: !!userId,
    staleTime: 60000, // 1 minute
    gcTime: 300000, // 5 minutes
    keepPreviousData: true,
    onError: (error: TelegramApiError) => {
      console.error(`Failed to fetch user audit logs for ${userId}:`, error);
      toast.error(`Failed to load user audit logs: ${error.message}`);
    },
    ...options,
  });
};

/**
 * Hook for fetching audit statistics
 */
export const useTelegramAuditStats = (
  options?: Omit<UseQueryOptions<AuditStats, TelegramApiError>, 'queryKey' | 'queryFn'>
) => {
  return useQuery({
    queryKey: telegramAuditKeys.stats(),
    queryFn: () => telegramApi.getAuditStats(),
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
    refetchInterval: 60000, // Refetch every minute for real-time stats
    onError: (error: TelegramApiError) => {
      console.error('Failed to fetch audit statistics:', error);
      toast.error(`Failed to load audit statistics: ${error.message}`);
    },
    ...options,
  });
};

// ============================================================================
// Real-time Audit Hooks
// ============================================================================

/**
 * Hook for managing real-time audit log updates
 */
export const useRealtimeAuditLogs = (params?: AuditLogParams) => {
  const queryClient = useQueryClient();
  const [recentActivity, setRecentActivity] = React.useState<CommandAudit[]>([]);

  // Add new audit log entry from WebSocket
  const addAuditLog = React.useCallback((newLog: CommandAudit) => {
    // Add to recent activity (keep last 10)
    setRecentActivity(prev => [newLog, ...prev.slice(0, 9)]);

    // Update relevant queries
    queryClient.setQueriesData<any>(
      { queryKey: telegramAuditKeys.logs() },
      (old) => {
        if (!old?.data) return old;
        
        return {
          ...old,
          data: [newLog, ...old.data],
          total: old.total + 1,
        };
      }
    );

    // Update user-specific logs if applicable
    if (params?.user_id === newLog.telegram_user_id) {
      queryClient.setQueriesData<any>(
        { queryKey: telegramAuditKeys.userLogs(newLog.telegram_user_id) },
        (old) => {
          if (!old?.data) return old;
          
          return {
            ...old,
            data: [newLog, ...old.data],
            total: old.total + 1,
          };
        }
      );
    }

    // Invalidate stats to trigger refresh
    queryClient.invalidateQueries({ queryKey: telegramAuditKeys.stats() });
  }, [queryClient, params?.user_id]);

  return {
    recentActivity,
    addAuditLog,
  };
};

// ============================================================================
// Audit Analytics Hooks
// ============================================================================

/**
 * Hook for audit log analytics and insights
 */
export const useAuditAnalytics = (timeRange: '24h' | '7d' | '30d' = '24h') => {
  const statsQuery = useTelegramAuditStats();
  
  // Calculate time range for filtering
  const getTimeRangeParams = React.useCallback((): AuditLogParams => {
    const now = new Date();
    const startDate = new Date();
    
    switch (timeRange) {
      case '24h':
        startDate.setHours(now.getHours() - 24);
        break;
      case '7d':
        startDate.setDate(now.getDate() - 7);
        break;
      case '30d':
        startDate.setDate(now.getDate() - 30);
        break;
    }
    
    return {
      start_date: startDate.toISOString(),
      end_date: now.toISOString(),
      limit: 1000, // Get more data for analytics
    };
  }, [timeRange]);

  const logsQuery = useTelegramAuditLogs(getTimeRangeParams());

  const analytics = React.useMemo(() => {
    if (!logsQuery.data?.data || !statsQuery.data) {
      return {
        totalCommands: 0,
        successRate: 0,
        failureRate: 0,
        topCommands: [],
        commandTrends: [],
        errorPatterns: [],
        userActivity: [],
      };
    }

    const logs = logsQuery.data.data;
    const stats = statsQuery.data;
    
    // Calculate success/failure rates
    const successfulCommands = logs.filter(log => log.success).length;
    const totalCommands = logs.length;
    const successRate = totalCommands > 0 ? (successfulCommands / totalCommands) * 100 : 0;
    const failureRate = 100 - successRate;

    // Analyze command patterns
    const commandCounts = logs.reduce((acc, log) => {
      acc[log.command] = (acc[log.command] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const topCommands = Object.entries(commandCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([command, count]) => ({ command, count }));

    // Analyze error patterns
    const errorLogs = logs.filter(log => !log.success && log.error_message);
    const errorCounts = errorLogs.reduce((acc, log) => {
      const error = log.error_message || 'Unknown error';
      acc[error] = (acc[error] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const errorPatterns = Object.entries(errorCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([error, count]) => ({ error, count }));

    // Analyze user activity
    const userCounts = logs.reduce((acc, log) => {
      acc[log.telegram_user_id] = (acc[log.telegram_user_id] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const userActivity = Object.entries(userCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([userId, count]) => ({ userId, count }));

    // Generate command trends (hourly breakdown for 24h, daily for longer periods)
    const commandTrends = generateCommandTrends(logs, timeRange);

    return {
      totalCommands,
      successRate: Math.round(successRate * 100) / 100,
      failureRate: Math.round(failureRate * 100) / 100,
      topCommands,
      commandTrends,
      errorPatterns,
      userActivity,
    };
  }, [logsQuery.data, statsQuery.data, timeRange]);

  return {
    ...analytics,
    isLoading: logsQuery.isLoading || statsQuery.isLoading,
    isError: logsQuery.isError || statsQuery.isError,
    error: logsQuery.error || statsQuery.error,
  };
};

/**
 * Hook for command execution performance metrics
 */
export const useCommandPerformanceMetrics = () => {
  const logsQuery = useTelegramAuditLogs({ limit: 1000 });

  const metrics = React.useMemo(() => {
    if (!logsQuery.data?.data) {
      return {
        averageExecutionTime: 0,
        slowestCommands: [],
        fastestCommands: [],
        performanceTrends: [],
      };
    }

    const logs = logsQuery.data.data;
    
    // Calculate average execution time
    const totalExecutionTime = logs.reduce((sum, log) => sum + log.execution_time_ms, 0);
    const averageExecutionTime = logs.length > 0 ? totalExecutionTime / logs.length : 0;

    // Find slowest and fastest commands
    const commandPerformance = logs.reduce((acc, log) => {
      if (!acc[log.command]) {
        acc[log.command] = { times: [], count: 0 };
      }
      acc[log.command].times.push(log.execution_time_ms);
      acc[log.command].count++;
      return acc;
    }, {} as Record<string, { times: number[]; count: number }>);

    const commandAverages = Object.entries(commandPerformance).map(([command, data]) => ({
      command,
      averageTime: data.times.reduce((sum, time) => sum + time, 0) / data.times.length,
      count: data.count,
    }));

    const slowestCommands = commandAverages
      .sort((a, b) => b.averageTime - a.averageTime)
      .slice(0, 5);

    const fastestCommands = commandAverages
      .sort((a, b) => a.averageTime - b.averageTime)
      .slice(0, 5);

    return {
      averageExecutionTime: Math.round(averageExecutionTime * 100) / 100,
      slowestCommands,
      fastestCommands,
      performanceTrends: [], // Could be implemented with time-series data
    };
  }, [logsQuery.data]);

  return {
    ...metrics,
    isLoading: logsQuery.isLoading,
    isError: logsQuery.isError,
    error: logsQuery.error,
  };
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Generate command trends based on time range
 */
function generateCommandTrends(logs: CommandAudit[], timeRange: '24h' | '7d' | '30d') {
  const now = new Date();
  const trends: Array<{ time: string; count: number }> = [];

  if (timeRange === '24h') {
    // Hourly breakdown for last 24 hours
    for (let i = 23; i >= 0; i--) {
      const hour = new Date(now.getTime() - i * 60 * 60 * 1000);
      const hourStart = new Date(hour);
      hourStart.setMinutes(0, 0, 0);
      const hourEnd = new Date(hourStart.getTime() + 60 * 60 * 1000);

      const count = logs.filter(log => {
        const logTime = new Date(log.timestamp);
        return logTime >= hourStart && logTime < hourEnd;
      }).length;

      trends.push({
        time: hourStart.toISOString(),
        count,
      });
    }
  } else {
    // Daily breakdown for longer periods
    const days = timeRange === '7d' ? 7 : 30;
    for (let i = days - 1; i >= 0; i--) {
      const day = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const dayStart = new Date(day);
      dayStart.setHours(0, 0, 0, 0);
      const dayEnd = new Date(dayStart.getTime() + 24 * 60 * 60 * 1000);

      const count = logs.filter(log => {
        const logTime = new Date(log.timestamp);
        return logTime >= dayStart && logTime < dayEnd;
      }).length;

      trends.push({
        time: dayStart.toISOString(),
        count,
      });
    }
  }

  return trends;
}
