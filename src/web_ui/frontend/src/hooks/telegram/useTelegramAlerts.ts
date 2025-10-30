/**
 * React Query hooks for Telegram alert management
 * 
 * This module provides custom hooks for managing Telegram bot alerts,
 * including fetching, filtering, and performing alert operations with
 * proper error handling.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';
import { telegramApi } from '../../api/telegramApi';
import {
  TelegramAlert,
  AlertFilterParams,
  AlertFormData,
  AlertStats,
  TelegramApiError
} from '../../types/telegram';

// ============================================================================
// Query Keys
// ============================================================================

/**
 * Query key factory for Telegram alert-related queries
 */
export const telegramAlertKeys = {
  all: ['telegram-alerts'] as const,
  lists: () => [...telegramAlertKeys.all, 'list'] as const,
  list: (filters: AlertFilterParams) => [...telegramAlertKeys.lists(), filters] as const,
  details: () => [...telegramAlertKeys.all, 'detail'] as const,
  detail: (id: string) => [...telegramAlertKeys.details(), id] as const,
  config: (id: string) => [...telegramAlertKeys.detail(id), 'config'] as const,
  stats: () => [...telegramAlertKeys.all, 'stats'] as const,
};

// ============================================================================
// Alert List Queries
// ============================================================================

/**
 * Hook for fetching Telegram alerts with filtering and pagination
 */
export const useTelegramAlerts = (params?: AlertFilterParams) => {
  return useQuery({
    queryKey: telegramAlertKeys.list(params || {}),
    queryFn: () => telegramApi.getAlerts(params),
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
    refetchOnWindowFocus: false,
    retry: (failureCount, error: any) => {
      if (error?.code === 'HTTP_401') return false;
      return failureCount < 3;
    },
  });
};

/**
 * Hook for fetching a specific Telegram alert by ID
 */
export const useTelegramAlert = (alertId: string) => {
  return useQuery({
    queryKey: telegramAlertKeys.detail(alertId),
    queryFn: () => telegramApi.getAlert(alertId),
    enabled: !!alertId,
    staleTime: 60000, // 1 minute
    gcTime: 300000, // 5 minutes
  });
};

/**
 * Hook for fetching alert configuration details
 */
export const useTelegramAlertConfig = (alertId: string) => {
  return useQuery({
    queryKey: telegramAlertKeys.config(alertId),
    queryFn: () => telegramApi.getAlertConfig(alertId),
    enabled: !!alertId,
    staleTime: 60000, // 1 minute
    gcTime: 300000, // 5 minutes
  });
};

/**
 * Hook for fetching alert statistics
 */
export const useTelegramAlertStats = () => {
  return useQuery({
    queryKey: telegramAlertKeys.stats(),
    queryFn: () => telegramApi.getAlertStats(),
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
    refetchInterval: 60000, // Refetch every minute for real-time stats
  });
};

// ============================================================================
// Alert Management Mutations
// ============================================================================

/**
 * Hook for creating a new Telegram alert
 */
export const useCreateTelegramAlert = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ userId, alertData }: { userId: string; alertData: AlertFormData }) => 
      telegramApi.createAlert(userId, alertData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: telegramAlertKeys.all });
      queryClient.invalidateQueries({ queryKey: telegramAlertKeys.stats() });
      toast.success('Alert created successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to create alert:', error);
      toast.error(`Failed to create alert: ${error.message}`);
    },
  });
};

/**
 * Hook for updating an existing Telegram alert
 */
export const useUpdateTelegramAlert = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ alertId, alertData }: { alertId: string; alertData: Partial<AlertFormData> }) => 
      telegramApi.updateAlert(alertId, alertData),
    onSuccess: (updatedAlert, { alertId }) => {
      queryClient.setQueryData(telegramAlertKeys.detail(alertId), updatedAlert);
      queryClient.invalidateQueries({ queryKey: telegramAlertKeys.lists() });
      queryClient.invalidateQueries({ queryKey: telegramAlertKeys.stats() });
      toast.success('Alert updated successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to update alert:', error);
      toast.error(`Failed to update alert: ${error.message}`);
    },
  });
};

/**
 * Hook for toggling alert active status
 */
export const useToggleTelegramAlert = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (alertId: string) => telegramApi.toggleAlert(alertId),
    onSuccess: (updatedAlert) => {
      queryClient.setQueryData(telegramAlertKeys.detail(updatedAlert.id), updatedAlert);
      queryClient.invalidateQueries({ queryKey: telegramAlertKeys.lists() });
      queryClient.invalidateQueries({ queryKey: telegramAlertKeys.stats() });
      
      const status = updatedAlert.is_active ? 'activated' : 'deactivated';
      toast.success(`Alert ${status} successfully`);
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to toggle alert:', error);
      toast.error(`Failed to toggle alert: ${error.message}`);
    },
  });
};

/**
 * Hook for deleting a Telegram alert
 */
export const useDeleteTelegramAlert = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (alertId: string) => telegramApi.deleteAlert(alertId),
    onSuccess: (_, alertId) => {
      queryClient.removeQueries({ queryKey: telegramAlertKeys.detail(alertId) });
      queryClient.removeQueries({ queryKey: telegramAlertKeys.config(alertId) });
      queryClient.invalidateQueries({ queryKey: telegramAlertKeys.all });
      queryClient.invalidateQueries({ queryKey: telegramAlertKeys.stats() });
      toast.success('Alert deleted successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to delete alert:', error);
      toast.error(`Failed to delete alert: ${error.message}`);
    },
  });
};

// ============================================================================
// Utility Hooks
// ============================================================================

/**
 * Hook for bulk alert operations
 */
export const useBulkAlertOperations = () => {
  const toggleAlert = useToggleTelegramAlert();
  const deleteAlert = useDeleteTelegramAlert();

  const bulkToggle = async (alertIds: string[]) => {
    const results = await Promise.allSettled(
      alertIds.map(alertId => toggleAlert.mutateAsync(alertId))
    );
    
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;
    
    if (successful > 0) {
      toast.success(`${successful} alerts toggled successfully`);
    }
    if (failed > 0) {
      toast.error(`${failed} alerts failed to toggle`);
    }
    
    return { successful, failed };
  };

  const bulkDelete = async (alertIds: string[]) => {
    const results = await Promise.allSettled(
      alertIds.map(alertId => deleteAlert.mutateAsync(alertId))
    );
    
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;
    
    if (successful > 0) {
      toast.success(`${successful} alerts deleted successfully`);
    }
    if (failed > 0) {
      toast.error(`${failed} alerts failed to delete`);
    }
    
    return { successful, failed };
  };

  return {
    bulkToggle,
    bulkDelete,
    isLoading: toggleAlert.isPending || deleteAlert.isPending,
  };
};