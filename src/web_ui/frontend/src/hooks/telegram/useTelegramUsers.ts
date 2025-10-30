/**
 * React Query hooks for Telegram user management
 * 
 * This module provides custom hooks for managing Telegram bot users,
 * including fetching, filtering, and performing user operations with
 * proper cache management and error handling.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';
import { telegramApi } from '../../api/telegramApi';
import {
  TelegramUser,
  UserFilterParams,
  UserUpdateData,
  UserStats,
  TelegramApiError
} from '../../types/telegram';

// ============================================================================
// Query Keys
// ============================================================================

/**
 * Query key factory for Telegram user-related queries
 */
export const telegramUserKeys = {
  all: ['telegram-users'] as const,
  lists: () => [...telegramUserKeys.all, 'list'] as const,
  list: (filters: UserFilterParams) => [...telegramUserKeys.lists(), filters] as const,
  details: () => [...telegramUserKeys.all, 'detail'] as const,
  detail: (id: string) => [...telegramUserKeys.details(), id] as const,
  stats: () => [...telegramUserKeys.all, 'stats'] as const,
};

// ============================================================================
// User List Queries
// ============================================================================

/**
 * Hook for fetching Telegram users with filtering and pagination
 */
export const useTelegramUsers = (params?: UserFilterParams) => {
  return useQuery({
    queryKey: telegramUserKeys.list(params || {}),
    queryFn: () => telegramApi.getUsers(params),
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
    refetchOnWindowFocus: false,
    retry: (failureCount, error: any) => {
      // Don't retry on authentication errors
      if (error?.code === 'HTTP_401') return false;
      return failureCount < 3;
    },
  });
};

/**
 * Hook for fetching a specific Telegram user by ID
 */
export const useTelegramUser = (userId: string) => {
  return useQuery({
    queryKey: telegramUserKeys.detail(userId),
    queryFn: () => telegramApi.getUser(userId),
    enabled: !!userId,
    staleTime: 60000, // 1 minute
    gcTime: 300000, // 5 minutes
  });
};

/**
 * Hook for fetching user statistics
 */
export const useTelegramUserStats = () => {
  return useQuery({
    queryKey: telegramUserKeys.stats(),
    queryFn: () => telegramApi.getUserStats(),
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
    refetchInterval: 60000, // Refetch every minute for real-time stats
  });
};

// ============================================================================
// User Management Mutations
// ============================================================================

/**
 * Hook for verifying a Telegram user's email
 */
export const useVerifyTelegramUser = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (userId: string) => telegramApi.verifyUser(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.all });
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.stats() });
      toast.success('User verified successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to verify user:', error);
      toast.error(`Failed to verify user: ${error.message}`);
    },
  });
};

/**
 * Hook for approving a Telegram user
 */
export const useApproveTelegramUser = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (userId: string) => telegramApi.approveUser(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.all });
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.stats() });
      toast.success('User approved successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to approve user:', error);
      toast.error(`Failed to approve user: ${error.message}`);
    },
  });
};

/**
 * Hook for rejecting a Telegram user
 */
export const useRejectTelegramUser = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (userId: string) => telegramApi.rejectUser(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.all });
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.stats() });
      toast.success('User rejected successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to reject user:', error);
      toast.error(`Failed to reject user: ${error.message}`);
    },
  });
};

/**
 * Hook for resetting a Telegram user's email
 */
export const useResetTelegramUserEmail = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (userId: string) => telegramApi.resetUserEmail(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.all });
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.stats() });
      toast.success('User email reset successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to reset user email:', error);
      toast.error(`Failed to reset user email: ${error.message}`);
    },
  });
};

/**
 * Hook for updating a Telegram user's settings
 */
export const useUpdateTelegramUser = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ userId, data }: { userId: string; data: UserUpdateData }) => 
      telegramApi.updateUser(userId, data),
    onSuccess: (updatedUser, { userId }) => {
      queryClient.setQueryData(telegramUserKeys.detail(userId), updatedUser);
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.lists() });
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.stats() });
      toast.success('User updated successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to update user:', error);
      toast.error(`Failed to update user: ${error.message}`);
    },
  });
};

/**
 * Hook for deleting a Telegram user
 */
export const useDeleteTelegramUser = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (userId: string) => telegramApi.deleteUser(userId),
    onSuccess: (_, userId) => {
      queryClient.removeQueries({ queryKey: telegramUserKeys.detail(userId) });
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.all });
      queryClient.invalidateQueries({ queryKey: telegramUserKeys.stats() });
      toast.success('User deleted successfully');
    },
    onError: (error: TelegramApiError) => {
      console.error('Failed to delete user:', error);
      toast.error(`Failed to delete user: ${error.message}`);
    },
  });
};

// ============================================================================
// Utility Hooks
// ============================================================================

/**
 * Hook for bulk user operations
 */
export const useBulkUserOperations = () => {
  const verifyUser = useVerifyTelegramUser();
  const approveUser = useApproveTelegramUser();
  const rejectUser = useRejectTelegramUser();
  const deleteUser = useDeleteTelegramUser();

  const bulkVerify = async (userIds: string[]) => {
    const results = await Promise.allSettled(
      userIds.map(userId => verifyUser.mutateAsync(userId))
    );
    
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;
    
    if (successful > 0) {
      toast.success(`${successful} users verified successfully`);
    }
    if (failed > 0) {
      toast.error(`${failed} users failed to verify`);
    }
    
    return { successful, failed };
  };

  const bulkApprove = async (userIds: string[]) => {
    const results = await Promise.allSettled(
      userIds.map(userId => approveUser.mutateAsync(userId))
    );
    
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;
    
    if (successful > 0) {
      toast.success(`${successful} users approved successfully`);
    }
    if (failed > 0) {
      toast.error(`${failed} users failed to approve`);
    }
    
    return { successful, failed };
  };

  const bulkReject = async (userIds: string[]) => {
    const results = await Promise.allSettled(
      userIds.map(userId => rejectUser.mutateAsync(userId))
    );
    
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;
    
    if (successful > 0) {
      toast.success(`${successful} users rejected successfully`);
    }
    if (failed > 0) {
      toast.error(`${failed} users failed to reject`);
    }
    
    return { successful, failed };
  };

  const bulkDelete = async (userIds: string[]) => {
    const results = await Promise.allSettled(
      userIds.map(userId => deleteUser.mutateAsync(userId))
    );
    
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;
    
    if (successful > 0) {
      toast.success(`${successful} users deleted successfully`);
    }
    if (failed > 0) {
      toast.error(`${failed} users failed to delete`);
    }
    
    return { successful, failed };
  };

  return {
    bulkVerify,
    bulkApprove,
    bulkReject,
    bulkDelete,
    isLoading: verifyUser.isPending || approveUser.isPending || rejectUser.isPending || deleteUser.isPending,
  };
};