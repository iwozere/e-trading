/**
 * Zod validation schemas for Telegram bot management
 * 
 * This file contains validation schemas for form inputs and API requests
 * related to Telegram bot management functionality.
 */

import { z } from 'zod';
import { AlertType, ScheduleType } from '../types/telegram';

// ============================================================================
// Base Validation Schemas
// ============================================================================

/**
 * Schema for validating Telegram user IDs
 */
export const telegramUserIdSchema = z.string()
  .min(1, 'Telegram user ID is required')
  .regex(/^\d+$/, 'Telegram user ID must be numeric');

/**
 * Schema for validating email addresses
 */
export const emailSchema = z.string()
  .email('Invalid email format')
  .optional()
  .nullable();

/**
 * Schema for validating trading symbols
 */
export const symbolSchema = z.string()
  .min(1, 'Symbol is required')
  .max(20, 'Symbol must be 20 characters or less')
  .regex(/^[A-Z0-9]+$/, 'Symbol must contain only uppercase letters and numbers');

/**
 * Schema for validating time in HH:MM format
 */
export const timeSchema = z.string()
  .regex(/^([01]?[0-9]|2[0-3]):[0-5][0-9]$/, 'Time must be in HH:MM format');

/**
 * Schema for validating timezone strings
 */
export const timezoneSchema = z.string()
  .min(1, 'Timezone is required')
  .max(50, 'Timezone must be 50 characters or less');

// ============================================================================
// Re-arm Configuration Schema
// ============================================================================

/**
 * Schema for validating alert re-arm configuration
 */
export const rearmConfigSchema = z.object({
  enabled: z.boolean(),
  type: z.enum(['immediate', 'time_based', 'price_based']),
  cooldown_minutes: z.number()
    .min(1, 'Cooldown must be at least 1 minute')
    .max(10080, 'Cooldown cannot exceed 7 days (10080 minutes)')
    .optional(),
  price_threshold: z.number()
    .positive('Price threshold must be positive')
    .optional(),
  hysteresis_percent: z.number()
    .min(0, 'Hysteresis cannot be negative')
    .max(100, 'Hysteresis cannot exceed 100%')
    .optional()
}).refine((data) => {
  // Validate that required fields are present based on type
  if (data.enabled) {
    if (data.type === 'time_based' && !data.cooldown_minutes) {
      return false;
    }
    if (data.type === 'price_based' && (!data.price_threshold || !data.hysteresis_percent)) {
      return false;
    }
  }
  return true;
}, {
  message: 'Required fields missing for selected re-arm type'
});

// ============================================================================
// Schedule Configuration Schema
// ============================================================================

/**
 * Schema for validating schedule configuration
 */
export const scheduleConfigSchema = z.object({
  report_type: z.enum(['portfolio', 'alerts', 'screener']),
  symbols: z.array(symbolSchema).optional(),
  parameters: z.record(z.any()).optional()
});

// ============================================================================
// Form Validation Schemas
// ============================================================================

/**
 * Schema for validating alert creation/editing forms
 */
export const alertFormSchema = z.object({
  symbol: symbolSchema,
  alert_type: z.nativeEnum(AlertType),
  target_value: z.number()
    .positive('Target value must be positive')
    .finite('Target value must be a valid number'),
  rearm_config: rearmConfigSchema
});

/**
 * Schema for validating schedule creation/editing forms
 */
export const scheduleFormSchema = z.object({
  schedule_type: z.nativeEnum(ScheduleType),
  time: timeSchema,
  timezone: timezoneSchema,
  config: scheduleConfigSchema
});

/**
 * Schema for validating user update forms
 */
export const userUpdateSchema = z.object({
  max_alerts: z.number()
    .min(0, 'Max alerts cannot be negative')
    .max(1000, 'Max alerts cannot exceed 1000')
    .optional(),
  max_schedules: z.number()
    .min(0, 'Max schedules cannot be negative')
    .max(100, 'Max schedules cannot exceed 100')
    .optional(),
  is_admin: z.boolean().optional()
});

/**
 * Schema for validating broadcast message forms
 */
export const broadcastMessageSchema = z.object({
  message: z.string()
    .min(1, 'Message cannot be empty')
    .max(4096, 'Message too long (max 4096 characters)'),
  title: z.string()
    .max(100, 'Title too long (max 100 characters)')
    .optional(),
  target_users: z.array(telegramUserIdSchema)
    .optional()
});

// ============================================================================
// API Request Validation Schemas
// ============================================================================

/**
 * Schema for validating user filter parameters
 */
export const userFilterParamsSchema = z.object({
  status: z.enum(['all', 'verified', 'approved', 'pending']).optional(),
  search: z.string().max(100, 'Search term too long').optional(),
  page: z.number().min(1, 'Page must be at least 1').optional(),
  limit: z.number().min(1, 'Limit must be at least 1').max(100, 'Limit cannot exceed 100').optional()
});

/**
 * Schema for validating alert filter parameters
 */
export const alertFilterParamsSchema = z.object({
  user_id: telegramUserIdSchema.optional(),
  symbol: symbolSchema.optional(),
  alert_type: z.nativeEnum(AlertType).optional(),
  status: z.enum(['active', 'inactive', 'triggered', 'cooldown', 'paused']).optional(),
  page: z.number().min(1, 'Page must be at least 1').optional(),
  limit: z.number().min(1, 'Limit must be at least 1').max(100, 'Limit cannot exceed 100').optional()
});

/**
 * Schema for validating schedule filter parameters
 */
export const scheduleFilterParamsSchema = z.object({
  user_id: telegramUserIdSchema.optional(),
  schedule_type: z.nativeEnum(ScheduleType).optional(),
  status: z.enum(['active', 'inactive', 'executing', 'failed']).optional(),
  page: z.number().min(1, 'Page must be at least 1').optional(),
  limit: z.number().min(1, 'Limit must be at least 1').max(100, 'Limit cannot exceed 100').optional()
});

/**
 * Schema for validating audit log filter parameters
 */
export const auditLogParamsSchema = z.object({
  user_id: telegramUserIdSchema.optional(),
  command: z.string().max(50, 'Command filter too long').optional(),
  success: z.boolean().optional(),
  start_date: z.string().datetime('Invalid start date format').optional(),
  end_date: z.string().datetime('Invalid end date format').optional(),
  page: z.number().min(1, 'Page must be at least 1').optional(),
  limit: z.number().min(1, 'Limit must be at least 1').max(100, 'Limit cannot exceed 100').optional()
}).refine((data) => {
  // Validate that end_date is after start_date if both are provided
  if (data.start_date && data.end_date) {
    return new Date(data.end_date) > new Date(data.start_date);
  }
  return true;
}, {
  message: 'End date must be after start date'
});

// ============================================================================
// Response Validation Schemas
// ============================================================================

/**
 * Schema for validating Telegram user responses
 */
export const telegramUserSchema = z.object({
  telegram_user_id: telegramUserIdSchema,
  email: emailSchema,
  verified: z.boolean(),
  approved: z.boolean(),
  language: z.string().min(1, 'Language is required'),
  is_admin: z.boolean(),
  max_alerts: z.number().min(0),
  max_schedules: z.number().min(0),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime()
});

/**
 * Schema for validating Telegram alert responses
 */
export const telegramAlertSchema = z.object({
  id: z.string().min(1, 'Alert ID is required'),
  user_id: telegramUserIdSchema,
  symbol: symbolSchema,
  alert_type: z.nativeEnum(AlertType),
  target_value: z.number().positive(),
  current_value: z.number(),
  is_active: z.boolean(),
  rearm_config: rearmConfigSchema,
  created_at: z.string().datetime(),
  last_triggered: z.string().datetime().nullable(),
  trigger_count: z.number().min(0)
});

/**
 * Schema for validating Telegram schedule responses
 */
export const telegramScheduleSchema = z.object({
  id: z.string().min(1, 'Schedule ID is required'),
  user_id: telegramUserIdSchema,
  schedule_type: z.nativeEnum(ScheduleType),
  time: timeSchema,
  timezone: timezoneSchema,
  config: scheduleConfigSchema,
  is_active: z.boolean(),
  created_at: z.string().datetime(),
  last_executed: z.string().datetime().nullable()
});

/**
 * Schema for validating command audit responses
 */
export const commandAuditSchema = z.object({
  id: z.string().min(1, 'Audit ID is required'),
  telegram_user_id: telegramUserIdSchema,
  command: z.string().min(1, 'Command is required'),
  full_message: z.string(),
  success: z.boolean(),
  error_message: z.string().optional(),
  execution_time_ms: z.number().min(0),
  timestamp: z.string().datetime()
});

/**
 * Schema for validating broadcast message responses
 */
export const broadcastMessageResponseSchema = z.object({
  id: z.string().min(1, 'Broadcast ID is required'),
  message: z.string().min(1, 'Message is required'),
  sent_by: z.string().min(1, 'Sender is required'),
  sent_at: z.string().datetime(),
  total_recipients: z.number().min(0),
  successful_deliveries: z.number().min(0),
  failed_deliveries: z.number().min(0)
});

// ============================================================================
// Statistics Validation Schemas
// ============================================================================

/**
 * Schema for validating user statistics responses
 */
export const userStatsSchema = z.object({
  total_users: z.number().min(0),
  verified_users: z.number().min(0),
  approved_users: z.number().min(0),
  pending_approvals: z.number().min(0),
  admin_users: z.number().min(0)
});

/**
 * Schema for validating alert statistics responses
 */
export const alertStatsSchema = z.object({
  total_alerts: z.number().min(0),
  active_alerts: z.number().min(0),
  triggered_today: z.number().min(0),
  rearm_cycles: z.number().min(0)
});

/**
 * Schema for validating schedule statistics responses
 */
export const scheduleStatsSchema = z.object({
  total_schedules: z.number().min(0),
  active_schedules: z.number().min(0),
  executed_today: z.number().min(0),
  failed_executions: z.number().min(0)
});

/**
 * Schema for validating audit statistics responses
 */
export const auditStatsSchema = z.object({
  total_commands: z.number().min(0),
  successful_commands: z.number().min(0),
  failed_commands: z.number().min(0),
  recent_activity_24h: z.number().min(0),
  top_commands: z.array(z.object({
    command: z.string(),
    count: z.number().min(0)
  }))
});

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Validates data against a schema and returns typed result
 */
export function validateData<T>(schema: z.ZodSchema<T>, data: unknown): { success: true; data: T } | { success: false; errors: z.ZodError } {
  const result = schema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  } else {
    return { success: false, errors: result.error };
  }
}

/**
 * Formats validation errors for display
 */
export function formatValidationErrors(errors: z.ZodError): string[] {
  return errors.errors.map(error => {
    const path = error.path.join('.');
    return path ? `${path}: ${error.message}` : error.message;
  });
}