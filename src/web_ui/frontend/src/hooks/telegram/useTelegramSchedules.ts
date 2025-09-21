/**
 * React Query hooks for Telegram schedule management
 * 
 * This module provides custom hooks for managing Telegram bot schedules,
 * including fetching, filtering, and performing schedule operations with
 * proper error handling.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';
import { telegramApi } from '../../api/telegramApi';
import {
  TelegramSchedule,
  ScheduleFilterParams,
  ScheduleFormData,
  ScheduleStats,
  TelegramApiError
} from '../../types/telegram';

// ============================================================================
// Query Keys
// ============================================================================

/**
 * Query key factory for Telegram schedule-related queries
 */
export const telegramScheduleKeys = {
  all: ['telegram-schedules'] as const,
  lists: () => [...telegramScheduleKeys.all, 'list'] as const,
  list: (filters: ScheduleFilterParams) => [...telegramScheduleKeys.lists(), filters] as const,
  details: () => [...telegramScheduleKeys.all, 'detail'] as const,
  detail: (id: string) => [...telegramScheduleKeys.details(), id] as const,
  stats: () => [...telegramScheduleKeys.all, 'stats'] as const,
};

// ============================================================================
// Schedule List Queries
// ============================================================================

/**
 * Hook for fetching Telegram schedules with filtering and pagination
 */
export const useTelegramSchedules = (params?: ScheduleFilterParams) => {
  return useQuery({
    queryKey: telegramScheduleKeys.list(params || {}),
    queryFn: () => telegramApi.getSchedules(params),
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
 * Hook for fetching a specific Telegram schedule by ID
 */
export const useTelegramSchedule = (scheduleId: string) => {
  return useQuery({
    queryKey: telegramScheduleKeys.detail(scheduleId),
    queryFn: () => telegramApi.getSchedule(scheduleId),
    enabled: !!scheduleId,
    staleTime: 60000, // 1 minute
    gcTime: 300000, // 5 minutes
  });
};

/**
 * Hook for fetching schedule statistics
 */
export const useTelegramScheduleStats = () => {
  return useQuery({
    queryKey: telegramScheduleKeys.stats(),
    queryFn: () => telegramApi.getScheduleStats(),
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
    refetchInterval: 60000, // Refetch every minute for real-time stats
  });
};

// ============================================================================
// Schedule Management Mutations
// ============================================================================

/**
 * Hook for creating a new Telegram schedule
 */
export const useCreateTelegramSchedule = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ userId, scheduleData }: { userId: string; scheduleData: ScheduleFormData }) => 
      telegramApi.createSchedule(userId, scheduleData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: telegramScheduleKeys.all });
      queryClient.invalidateQueries({ queryKey: telegramScheduleKeys.stats() });
      toast.success('Schedule created successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to create schedule:', error);
      toast.error(`Failed to create schedule: ${error.message}`);
    },
  });
};

/**
 * Hook for updating an existing Telegram schedule
 */
export const useUpdateTelegramSchedule = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ scheduleId, scheduleData }: { scheduleId: string; scheduleData: Partial<ScheduleFormData> }) => 
      telegramApi.updateSchedule(scheduleId, scheduleData),
    onSuccess: (updatedSchedule, { scheduleId }) => {
      queryClient.setQueryData(telegramScheduleKeys.detail(scheduleId), updatedSchedule);
      queryClient.invalidateQueries({ queryKey: telegramScheduleKeys.lists() });
      queryClient.invalidateQueries({ queryKey: telegramScheduleKeys.stats() });
      toast.success('Schedule updated successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to update schedule:', error);
      toast.error(`Failed to update schedule: ${error.message}`);
    },
  });
};

/**
 * Hook for toggling schedule active status
 */
export const useToggleTelegramSchedule = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (scheduleId: string) => telegramApi.toggleSchedule(scheduleId),
    onSuccess: (updatedSchedule) => {
      queryClient.setQueryData(telegramScheduleKeys.detail(updatedSchedule.id), updatedSchedule);
      queryClient.invalidateQueries({ queryKey: telegramScheduleKeys.lists() });
      queryClient.invalidateQueries({ queryKey: telegramScheduleKeys.stats() });
      
      const status = updatedSchedule.is_active ? 'activated' : 'deactivated';
      toast.success(`Schedule ${status} successfully`);
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to toggle schedule:', error);
      toast.error(`Failed to toggle schedule: ${error.message}`);
    },
  });
};

/**
 * Hook for deleting a Telegram schedule
 */
export const useDeleteTelegramSchedule = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (scheduleId: string) => telegramApi.deleteSchedule(scheduleId),
    onSuccess: (_, scheduleId) => {
      queryClient.removeQueries({ queryKey: telegramScheduleKeys.detail(scheduleId) });
      queryClient.invalidateQueries({ queryKey: telegramScheduleKeys.all });
      queryClient.invalidateQueries({ queryKey: telegramScheduleKeys.stats() });
      toast.success('Schedule deleted successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to delete schedule:', error);
      toast.error(`Failed to delete schedule: ${error.message}`);
    },
  });
};

// ============================================================================
// Utility Hooks
// ============================================================================

/**
 * Hook for bulk schedule operations
 */
export const useBulkScheduleOperations = () => {
  const toggleSchedule = useToggleTelegramSchedule();
  const deleteSchedule = useDeleteTelegramSchedule();

  const bulkToggle = async (scheduleIds: string[]) => {
    const results = await Promise.allSettled(
      scheduleIds.map(scheduleId => toggleSchedule.mutateAsync(scheduleId))
    );
    
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;
    
    if (successful > 0) {
      toast.success(`${successful} schedules toggled successfully`);
    }
    if (failed > 0) {
      toast.error(`${failed} schedules failed to toggle`);
    }
    
    return { successful, failed };
  };

  const bulkDelete = async (scheduleIds: string[]) => {
    const results = await Promise.allSettled(
      scheduleIds.map(scheduleId => deleteSchedule.mutateAsync(scheduleId))
    );
    
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;
    
    if (successful > 0) {
      toast.success(`${successful} schedules deleted successfully`);
    }
    if (failed > 0) {
      toast.error(`${failed} schedules failed to delete`);
    }
    
    return { successful, failed };
  };

  return {
    bulkToggle,
    bulkDelete,
    isLoading: toggleSchedule.isPending || deleteSchedule.isPending,
  };
};