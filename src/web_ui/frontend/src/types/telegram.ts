/**
 * TypeScript interfaces and types for Telegram bot management
 * 
 * This file contains all the data models and types used for managing
 * the Telegram bot system within the unified web UI.
 */

// ============================================================================
// Core Entity Interfaces
// ============================================================================

/**
 * Represents a Telegram bot user
 */
export interface TelegramUser {
  telegram_user_id: string;
  email: string | null;
  verified: boolean;
  approved: boolean;
  language: string;
  is_admin: boolean;
  max_alerts: number;
  max_schedules: number;
  created_at: string;
  updated_at: string;
}

/**
 * Configuration for alert re-arm functionality
 */
export interface RearmConfig {
  enabled: boolean;
  type: 'immediate' | 'time_based' | 'price_based';
  cooldown_minutes?: number;
  price_threshold?: number;
  hysteresis_percent?: number;
}

/**
 * Represents a Telegram price alert
 */
export interface TelegramAlert {
  id: string;
  user_id: string;
  symbol: string;
  alert_type: AlertType;
  target_value: number;
  current_value: number;
  is_active: boolean;
  rearm_config: RearmConfig;
  created_at: string;
  last_triggered: string | null;
  trigger_count: number;
}

/**
 * Configuration for scheduled reports
 */
export interface ScheduleConfig {
  report_type: 'portfolio' | 'alerts' | 'screener';
  symbols?: string[];
  parameters?: Record<string, any>;
}

/**
 * Represents a scheduled report
 */
export interface TelegramSchedule {
  id: string;
  user_id: string;
  schedule_type: ScheduleType;
  time: string; // HH:MM format
  timezone: string;
  config: ScheduleConfig;
  is_active: boolean;
  created_at: string;
  last_executed: string | null;
}

/**
 * Represents a command audit log entry
 */
export interface CommandAudit {
  id: string;
  telegram_user_id: string;
  command: string;
  full_message: string;
  success: boolean;
  error_message?: string;
  execution_time_ms: number;
  timestamp: string;
}

/**
 * Represents a broadcast message
 */
export interface BroadcastMessage {
  id: string;
  message: string;
  sent_by: string;
  sent_at: string;
  total_recipients: number;
  successful_deliveries: number;
  failed_deliveries: number;
}

// ============================================================================
// Enum Types
// ============================================================================

/**
 * Types of alerts that can be configured
 */
export enum AlertType {
  PRICE_ABOVE = 'price_above',
  PRICE_BELOW = 'price_below',
  PERCENTAGE_CHANGE = 'percentage_change'
}

/**
 * Types of scheduled reports
 */
export enum ScheduleType {
  DAILY = 'daily',
  WEEKLY = 'weekly'
}

/**
 * User verification and approval statuses
 */
export enum UserStatus {
  PENDING = 'pending',
  VERIFIED = 'verified',
  APPROVED = 'approved',
  REJECTED = 'rejected'
}

/**
 * Alert status types
 */
export enum AlertStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  TRIGGERED = 'triggered',
  COOLDOWN = 'cooldown',
  PAUSED = 'paused'
}

/**
 * Schedule execution status
 */
export enum ScheduleStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  EXECUTING = 'executing',
  FAILED = 'failed'
}

// ============================================================================
// Statistics and Analytics Interfaces
// ============================================================================

/**
 * User management statistics
 */
export interface UserStats {
  total_users: number;
  verified_users: number;
  approved_users: number;
  pending_approvals: number;
  admin_users: number;
}

/**
 * Alert management statistics
 */
export interface AlertStats {
  total_alerts: number;
  active_alerts: number;
  triggered_today: number;
  rearm_cycles: number;
}

/**
 * Schedule management statistics
 */
export interface ScheduleStats {
  total_schedules: number;
  active_schedules: number;
  executed_today: number;
  failed_executions: number;
}

/**
 * Command audit statistics
 */
export interface AuditStats {
  total_commands: number;
  successful_commands: number;
  failed_commands: number;
  recent_activity_24h: number;
  top_commands: Array<{ command: string; count: number }>;
}

// ============================================================================
// API Request/Response Types
// ============================================================================

/**
 * Parameters for filtering users
 */
export interface UserFilterParams {
  status?: 'all' | 'verified' | 'approved' | 'pending';
  search?: string;
  page?: number;
  limit?: number;
}

/**
 * Parameters for filtering alerts
 */
export interface AlertFilterParams {
  user_id?: string;
  symbol?: string;
  alert_type?: AlertType;
  status?: AlertStatus;
  page?: number;
  limit?: number;
}

/**
 * Parameters for filtering schedules
 */
export interface ScheduleFilterParams {
  user_id?: string;
  schedule_type?: ScheduleType;
  status?: ScheduleStatus;
  page?: number;
  limit?: number;
}

/**
 * Parameters for filtering audit logs
 */
export interface AuditLogParams {
  user_id?: string;
  command?: string;
  success?: boolean;
  start_date?: string;
  end_date?: string;
  page?: number;
  limit?: number;
}

/**
 * Broadcast message request
 */
export interface BroadcastRequest {
  message: string;
  title?: string;
  target_users?: string[]; // If empty, broadcast to all users
}

/**
 * Broadcast result response
 */
export interface BroadcastResult {
  broadcast_id: string;
  total_recipients: number;
  successful_deliveries: number;
  failed_deliveries: number;
  delivery_status: 'pending' | 'in_progress' | 'completed' | 'failed';
}

// ============================================================================
// Form Data Types
// ============================================================================

/**
 * Form data for creating/editing alerts
 */
export interface AlertFormData {
  symbol: string;
  alert_type: AlertType;
  target_value: number;
  rearm_config: RearmConfig;
}

/**
 * Form data for creating/editing schedules
 */
export interface ScheduleFormData {
  schedule_type: ScheduleType;
  time: string;
  timezone: string;
  config: ScheduleConfig;
}

/**
 * Form data for user management operations
 */
export interface UserUpdateData {
  max_alerts?: number;
  max_schedules?: number;
  is_admin?: boolean;
}

// ============================================================================
// WebSocket Event Types
// ============================================================================

/**
 * WebSocket event types for real-time updates
 */
export enum TelegramWebSocketEvent {
  USER_REGISTERED = 'telegram_user_registered',
  USER_VERIFIED = 'telegram_user_verified',
  USER_APPROVED = 'telegram_user_approved',
  ALERT_TRIGGERED = 'telegram_alert_triggered',
  SCHEDULE_EXECUTED = 'telegram_schedule_executed',
  COMMAND_EXECUTED = 'telegram_command_executed',
  BROADCAST_SENT = 'telegram_broadcast_sent',
  BROADCAST_DELIVERED = 'telegram_broadcast_delivered'
}

/**
 * WebSocket event data for user events
 */
export interface UserWebSocketEvent {
  type: TelegramWebSocketEvent.USER_REGISTERED | TelegramWebSocketEvent.USER_VERIFIED | TelegramWebSocketEvent.USER_APPROVED;
  user: TelegramUser;
  timestamp: string;
}

/**
 * WebSocket event data for alert events
 */
export interface AlertWebSocketEvent {
  type: TelegramWebSocketEvent.ALERT_TRIGGERED;
  alert: TelegramAlert;
  timestamp: string;
}

/**
 * WebSocket event data for schedule events
 */
export interface ScheduleWebSocketEvent {
  type: TelegramWebSocketEvent.SCHEDULE_EXECUTED;
  schedule: TelegramSchedule;
  success: boolean;
  timestamp: string;
}

/**
 * WebSocket event data for command events
 */
export interface CommandWebSocketEvent {
  type: TelegramWebSocketEvent.COMMAND_EXECUTED;
  audit: CommandAudit;
  timestamp: string;
}

/**
 * WebSocket event data for broadcast events
 */
export interface BroadcastWebSocketEvent {
  type: TelegramWebSocketEvent.BROADCAST_SENT | TelegramWebSocketEvent.BROADCAST_DELIVERED;
  broadcast: BroadcastMessage;
  timestamp: string;
}

/**
 * Union type for all Telegram WebSocket events
 */
export type TelegramWebSocketEventData = 
  | UserWebSocketEvent
  | AlertWebSocketEvent
  | ScheduleWebSocketEvent
  | CommandWebSocketEvent
  | BroadcastWebSocketEvent;

// ============================================================================
// Error Types
// ============================================================================

/**
 * Telegram API error response
 */
export interface TelegramApiError {
  message: string;
  code: string;
  details?: Record<string, any>;
}

/**
 * Validation error for forms
 */
export interface ValidationError {
  field: string;
  message: string;
}