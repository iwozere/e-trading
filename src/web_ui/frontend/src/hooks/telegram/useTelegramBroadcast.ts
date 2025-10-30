/**
 * React Query hooks for Telegram broadcast management
 * 
 * This module provides custom hooks for managing Telegram bot broadcasts,
 * including sending messages, tracking delivery status, and viewing history
 * with proper error handling.
 */

import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';
import { telegramApi } from '../../api/telegramApi';
import {
  BroadcastMessage,
  BroadcastRequest,
  BroadcastResult,
  TelegramApiError
} from '../../types/telegram';

// ============================================================================
// Query Keys
// ============================================================================

/**
 * Query key factory for Telegram broadcast-related queries
 */
export const telegramBroadcastKeys = {
  all: ['telegram-broadcasts'] as const,
  history: () => [...telegramBroadcastKeys.all, 'history'] as const,
  historyPaginated: (page: number, limit: number) => [...telegramBroadcastKeys.history(), page, limit] as const,
  status: (broadcastId: string) => [...telegramBroadcastKeys.all, 'status', broadcastId] as const,
};

// ============================================================================
// Broadcast History Queries
// ============================================================================

/**
 * Hook for fetching broadcast history with pagination
 */
export const useBroadcastHistory = (page = 1, limit = 50) => {
  return useQuery({
    queryKey: telegramBroadcastKeys.historyPaginated(page, limit),
    queryFn: () => telegramApi.getBroadcastHistory(page, limit),
    staleTime: 60000, // 1 minute
    gcTime: 300000, // 5 minutes
  });
};

/**
 * Hook for fetching broadcast delivery status
 */
export const useBroadcastStatus = (broadcastId: string) => {
  return useQuery({
    queryKey: telegramBroadcastKeys.status(broadcastId),
    queryFn: () => telegramApi.getBroadcastStatus(broadcastId),
    enabled: !!broadcastId,
    staleTime: 10000, // 10 seconds for real-time status
    gcTime: 60000, // 1 minute
    refetchInterval: (query) => {
      const data = query.state.data as BroadcastResult | undefined;
      // Stop refetching if broadcast is completed or failed
      if (data?.delivery_status === 'completed' || data?.delivery_status === 'failed') {
        return false;
      }
      // Refetch every 5 seconds for pending/in_progress broadcasts
      return 5000;
    },
  });
};

// ============================================================================
// Broadcast Mutations
// ============================================================================

/**
 * Hook for sending broadcast messages
 */
export const useSendBroadcast = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (broadcastData: BroadcastRequest) => telegramApi.sendBroadcast(broadcastData),
    onSuccess: (result) => {
      // Invalidate broadcast history to ensure fresh data
      queryClient.invalidateQueries({ queryKey: telegramBroadcastKeys.history() });

      // Set up status tracking for the new broadcast
      queryClient.setQueryData(telegramBroadcastKeys.status(result.broadcast_id), result);

      toast.success(`Broadcast sent to ${result.total_recipients} recipients`);
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to send broadcast:', error);
      toast.error(`Failed to send broadcast: ${error.message}`);
    },
  });
};

// ============================================================================
// Real-time Broadcast Hooks
// ============================================================================

/**
 * Hook for tracking broadcast delivery in real-time
 */
export const useBroadcastDeliveryTracking = (broadcastId: string) => {
  const queryClient = useQueryClient();
  
  // Get initial status
  const statusQuery = useBroadcastStatus(broadcastId);

  // Update broadcast status from WebSocket events
  const updateBroadcastStatus = (updatedStatus: BroadcastResult) => {
    queryClient.setQueryData(telegramBroadcastKeys.status(broadcastId), updatedStatus);
  };

  return {
    status: statusQuery.data,
    isLoading: statusQuery.isLoading,
    isError: statusQuery.isError,
    error: statusQuery.error,
    updateStatus: updateBroadcastStatus,
  };
};

/**
 * Hook for managing broadcast templates and quick actions
 */
export const useBroadcastTemplates = () => {
  const sendBroadcast = useSendBroadcast();

  const templates = {
    maintenance: {
      title: 'Maintenance Notice',
      message: 'The trading bot will be undergoing scheduled maintenance. All alerts and schedules will be temporarily paused.',
    },
    update: {
      title: 'System Update',
      message: 'New features have been added to the trading bot. Check out the latest updates in your dashboard.',
    },
    alert: {
      title: 'Important Alert',
      message: 'This is an important notification regarding your trading alerts.',
    },
  };

  const sendTemplate = async (templateKey: keyof typeof templates, customMessage?: string) => {
    const template = templates[templateKey];
    const message = customMessage || template.message;
    
    return sendBroadcast.mutateAsync({
      message,
      title: template.title,
    });
  };

  return {
    templates,
    sendTemplate,
    isLoading: sendBroadcast.isPending,
  };
};

// ============================================================================
// Broadcast Analytics Hooks
// ============================================================================

/**
 * Hook for broadcast analytics and statistics
 */
export const useBroadcastAnalytics = () => {
  const historyQuery = useBroadcastHistory(1, 100); // Get recent broadcasts for analytics

  const analytics = React.useMemo(() => {
    if (!historyQuery.data?.data) {
      return {
        totalBroadcasts: 0,
        totalRecipients: 0,
        averageDeliveryRate: 0,
        recentBroadcasts: [],
      };
    }

    const broadcasts = historyQuery.data.data;
    const totalBroadcasts = broadcasts.length;
    const totalRecipients = broadcasts.reduce((sum: number, b: BroadcastMessage) => sum + b.total_recipients, 0);
    const totalSuccessful = broadcasts.reduce((sum: number, b: BroadcastMessage) => sum + b.successful_deliveries, 0);
    const averageDeliveryRate = totalRecipients > 0 ? (totalSuccessful / totalRecipients) * 100 : 0;

    return {
      totalBroadcasts,
      totalRecipients,
      averageDeliveryRate: Math.round(averageDeliveryRate * 100) / 100,
      recentBroadcasts: broadcasts.slice(0, 5),
    };
  }, [historyQuery.data]);

  return {
    ...analytics,
    isLoading: historyQuery.isLoading,
    isError: historyQuery.isError,
    error: historyQuery.error,
  };
};