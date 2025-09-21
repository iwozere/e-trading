/**
 * Telegram Bot Management Hooks
 * 
 * This module exports all React Query hooks for managing Telegram bot
 * functionality within the unified web UI system.
 */

// User Management Hooks
export {
  useTelegramUsers,
  useTelegramUser,
  useTelegramUserStats,
  useVerifyTelegramUser,
  useApproveTelegramUser,
  useRejectTelegramUser,
  useResetTelegramUserEmail,
  useUpdateTelegramUser,
  useDeleteTelegramUser,
  useBulkUserOperations,
  telegramUserKeys,
} from './useTelegramUsers';

// Alert Management Hooks
export {
  useTelegramAlerts,
  useTelegramAlert,
  useTelegramAlertConfig,
  useTelegramAlertStats,
  useCreateTelegramAlert,
  useUpdateTelegramAlert,
  useToggleTelegramAlert,
  useDeleteTelegramAlert,
  useBulkAlertOperations,
  telegramAlertKeys,
} from './useTelegramAlerts';

// Schedule Management Hooks
export {
  useTelegramSchedules,
  useTelegramSchedule,
  useTelegramScheduleStats,
  useCreateTelegramSchedule,
  useUpdateTelegramSchedule,
  useToggleTelegramSchedule,
  useDeleteTelegramSchedule,
  useBulkScheduleOperations,
  telegramScheduleKeys,
} from './useTelegramSchedules';

// Broadcast Management Hooks
export {
  useBroadcastHistory,
  useBroadcastStatus,
  useSendBroadcast,
  useBroadcastDeliveryTracking,
  useBroadcastTemplates,
  useBroadcastAnalytics,
  telegramBroadcastKeys,
} from './useTelegramBroadcast';

// Audit Log Management Hooks
export {
  useTelegramAuditLogs,
  useTelegramAuditLogsInfinite,
  useUserAuditLogs,
  useTelegramAuditStats,
  useRealtimeAuditLogs,
  useAuditAnalytics,
  useCommandPerformanceMetrics,
  telegramAuditKeys,
} from './useTelegramAudit';

// Statistics and Analytics Hooks
export {
  useTelegramCombinedStats,
  useTelegramStatsWithRefresh,
  useRealtimeTelegramStats,
  useTelegramDashboardStats,
  useTelegramStatsComparison,
  useTelegramStatsExport,
  type TelegramCombinedStats,
  type TelegramStatsUpdate,
} from './useTelegramStats';

// Re-export types for convenience
export type {
  TelegramUser,
  TelegramAlert,
  TelegramSchedule,
  CommandAudit,
  BroadcastMessage,
  BroadcastResult,
  UserStats,
  AlertStats,
  ScheduleStats,
  AuditStats,
  UserFilterParams,
  AlertFilterParams,
  ScheduleFilterParams,
  AuditLogParams,
  BroadcastRequest,
  AlertFormData,
  ScheduleFormData,
  UserUpdateData,
  TelegramApiError,
} from '../../types/telegram';
