/**
 * Telegram Bot API Service
 * 
 * This service provides methods for interacting with the Telegram bot
 * management backend API. It handles all CRUD operations for users,
 * alerts, schedules, broadcasts, and audit logs.
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  TelegramUser,
  TelegramAlert,
  TelegramSchedule,
  CommandAudit,
  BroadcastMessage,
  UserStats,
  AlertStats,
  ScheduleStats,
  AuditStats,
  UserFilterParams,
  AlertFilterParams,
  ScheduleFilterParams,
  AuditLogParams,
  BroadcastRequest,
  BroadcastResult,
  AlertFormData,
  ScheduleFormData,
  UserUpdateData,
  TelegramApiError
} from '../types/telegram';

/**
 * Configuration for the Telegram API service
 */
interface TelegramApiConfig {
  baseURL: string;
  timeout: number;
}

/**
 * Paginated response wrapper
 */
interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
}

/**
 * Telegram Bot API Service Class
 */
export class TelegramApiService {
  private api: AxiosInstance;
  private baseUrl: string;

  constructor(config?: Partial<TelegramApiConfig>) {
    this.baseUrl = config?.baseURL || '/api/telegram';
    
    this.api = axios.create({
      baseURL: this.baseUrl,
      timeout: config?.timeout || 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  /**
   * Set up request and response interceptors
   */
  private setupInterceptors(): void {
    // Request interceptor - add auth token
    this.api.interceptors.request.use(
      (config) => {
        const token = this.getAuthToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(this.handleError(error));
      }
    );

    // Response interceptor - handle errors
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        return Promise.reject(this.handleError(error));
      }
    );
  }

  /**
   * Get authentication token from storage
   */
  private getAuthToken(): string | null {
    try {
      const authData = localStorage.getItem('auth-storage');
      if (authData) {
        const parsed = JSON.parse(authData);
        return parsed.state?.token || null;
      }
    } catch (error) {
      console.error('Error parsing auth token:', error);
    }
    return null;
  }

  /**
   * Handle API errors and convert to TelegramApiError
   */
  private handleError(error: any): TelegramApiError {
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      if (status === 401) {
        // Handle unauthorized - redirect to login
        localStorage.removeItem('auth-storage');
        window.location.href = '/login';
      }

      return {
        message: data?.message || `HTTP ${status} Error`,
        code: data?.code || `HTTP_${status}`,
        details: data?.details || { status, url: error.config?.url }
      };
    } else if (error.request) {
      // Network error
      return {
        message: 'Network error - please check your connection',
        code: 'NETWORK_ERROR',
        details: { originalError: error.message }
      };
    } else {
      // Other error
      return {
        message: error.message || 'An unexpected error occurred',
        code: 'UNKNOWN_ERROR',
        details: { originalError: error }
      };
    }
  }

  /**
   * Build query string from parameters
   */
  private buildQueryString(params: Record<string, any>): string {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        searchParams.append(key, String(value));
      }
    });

    return searchParams.toString();
  }

  // ============================================================================
  // User Management API Methods
  // ============================================================================

  /**
   * Get list of Telegram users with optional filtering
   */
  async getUsers(params?: UserFilterParams): Promise<PaginatedResponse<TelegramUser>> {
    const queryString = params ? this.buildQueryString(params) : '';
    const url = queryString ? `/users?${queryString}` : '/users';
    
    const response: AxiosResponse<PaginatedResponse<TelegramUser>> = await this.api.get(url);
    return response.data;
  }

  /**
   * Get a specific user by ID
   */
  async getUser(userId: string): Promise<TelegramUser> {
    const response: AxiosResponse<TelegramUser> = await this.api.get(`/users/${userId}`);
    return response.data;
  }

  /**
   * Verify a user's email
   */
  async verifyUser(userId: string): Promise<void> {
    await this.api.post(`/users/${userId}/verify`);
  }

  /**
   * Approve a user for bot access
   */
  async approveUser(userId: string): Promise<void> {
    await this.api.post(`/users/${userId}/approve`);
  }

  /**
   * Reject a user's access request
   */
  async rejectUser(userId: string): Promise<void> {
    await this.api.post(`/users/${userId}/reject`);
  }

  /**
   * Reset a user's email
   */
  async resetUserEmail(userId: string): Promise<void> {
    await this.api.post(`/users/${userId}/reset-email`);
  }

  /**
   * Update user settings
   */
  async updateUser(userId: string, data: UserUpdateData): Promise<TelegramUser> {
    const response: AxiosResponse<TelegramUser> = await this.api.put(`/users/${userId}`, data);
    return response.data;
  }

  /**
   * Delete a user
   */
  async deleteUser(userId: string): Promise<void> {
    await this.api.delete(`/users/${userId}`);
  }

  // ============================================================================
  // Alert Management API Methods
  // ============================================================================

  /**
   * Get list of alerts with optional filtering
   */
  async getAlerts(params?: AlertFilterParams): Promise<PaginatedResponse<TelegramAlert>> {
    const queryString = params ? this.buildQueryString(params) : '';
    const url = queryString ? `/alerts?${queryString}` : '/alerts';
    
    const response: AxiosResponse<PaginatedResponse<TelegramAlert>> = await this.api.get(url);
    return response.data;
  }

  /**
   * Get a specific alert by ID
   */
  async getAlert(alertId: string): Promise<TelegramAlert> {
    const response: AxiosResponse<TelegramAlert> = await this.api.get(`/alerts/${alertId}`);
    return response.data;
  }

  /**
   * Create a new alert
   */
  async createAlert(userId: string, alertData: AlertFormData): Promise<TelegramAlert> {
    const response: AxiosResponse<TelegramAlert> = await this.api.post('/alerts', {
      user_id: userId,
      ...alertData
    });
    return response.data;
  }

  /**
   * Update an existing alert
   */
  async updateAlert(alertId: string, alertData: Partial<AlertFormData>): Promise<TelegramAlert> {
    const response: AxiosResponse<TelegramAlert> = await this.api.put(`/alerts/${alertId}`, alertData);
    return response.data;
  }

  /**
   * Toggle alert active status
   */
  async toggleAlert(alertId: string): Promise<TelegramAlert> {
    const response: AxiosResponse<TelegramAlert> = await this.api.post(`/alerts/${alertId}/toggle`);
    return response.data;
  }

  /**
   * Delete an alert
   */
  async deleteAlert(alertId: string): Promise<void> {
    await this.api.delete(`/alerts/${alertId}`);
  }

  /**
   * Get alert configuration details
   */
  async getAlertConfig(alertId: string): Promise<any> {
    const response: AxiosResponse<any> = await this.api.get(`/alerts/${alertId}/config`);
    return response.data;
  }

  // ============================================================================
  // Schedule Management API Methods
  // ============================================================================

  /**
   * Get list of schedules with optional filtering
   */
  async getSchedules(params?: ScheduleFilterParams): Promise<PaginatedResponse<TelegramSchedule>> {
    const queryString = params ? this.buildQueryString(params) : '';
    const url = queryString ? `/schedules?${queryString}` : '/schedules';
    
    const response: AxiosResponse<PaginatedResponse<TelegramSchedule>> = await this.api.get(url);
    return response.data;
  }

  /**
   * Get a specific schedule by ID
   */
  async getSchedule(scheduleId: string): Promise<TelegramSchedule> {
    const response: AxiosResponse<TelegramSchedule> = await this.api.get(`/schedules/${scheduleId}`);
    return response.data;
  }

  /**
   * Create a new schedule
   */
  async createSchedule(userId: string, scheduleData: ScheduleFormData): Promise<TelegramSchedule> {
    const response: AxiosResponse<TelegramSchedule> = await this.api.post('/schedules', {
      user_id: userId,
      ...scheduleData
    });
    return response.data;
  }

  /**
   * Update an existing schedule
   */
  async updateSchedule(scheduleId: string, scheduleData: Partial<ScheduleFormData>): Promise<TelegramSchedule> {
    const response: AxiosResponse<TelegramSchedule> = await this.api.put(`/schedules/${scheduleId}`, scheduleData);
    return response.data;
  }

  /**
   * Toggle schedule active status
   */
  async toggleSchedule(scheduleId: string): Promise<TelegramSchedule> {
    const response: AxiosResponse<TelegramSchedule> = await this.api.post(`/schedules/${scheduleId}/toggle`);
    return response.data;
  }

  /**
   * Delete a schedule
   */
  async deleteSchedule(scheduleId: string): Promise<void> {
    await this.api.delete(`/schedules/${scheduleId}`);
  }

  // ============================================================================
  // Broadcast Management API Methods
  // ============================================================================

  /**
   * Send a broadcast message
   */
  async sendBroadcast(broadcastData: BroadcastRequest): Promise<BroadcastResult> {
    const response: AxiosResponse<BroadcastResult> = await this.api.post('/broadcast', broadcastData);
    return response.data;
  }

  /**
   * Get broadcast history
   */
  async getBroadcastHistory(page = 1, limit = 50): Promise<PaginatedResponse<BroadcastMessage>> {
    const response: AxiosResponse<PaginatedResponse<BroadcastMessage>> = await this.api.get(
      `/broadcast/history?page=${page}&limit=${limit}`
    );
    return response.data;
  }

  /**
   * Get broadcast delivery status
   */
  async getBroadcastStatus(broadcastId: string): Promise<BroadcastResult> {
    const response: AxiosResponse<BroadcastResult> = await this.api.get(`/broadcast/${broadcastId}/status`);
    return response.data;
  }

  // ============================================================================
  // Audit Log API Methods
  // ============================================================================

  /**
   * Get audit logs with optional filtering
   */
  async getAuditLogs(params?: AuditLogParams): Promise<PaginatedResponse<CommandAudit>> {
    const queryString = params ? this.buildQueryString(params) : '';
    const url = queryString ? `/audit?${queryString}` : '/audit';
    
    const response: AxiosResponse<PaginatedResponse<CommandAudit>> = await this.api.get(url);
    return response.data;
  }

  /**
   * Get audit logs for a specific user
   */
  async getUserAuditLogs(userId: string, page = 1, limit = 50): Promise<PaginatedResponse<CommandAudit>> {
    const response: AxiosResponse<PaginatedResponse<CommandAudit>> = await this.api.get(
      `/audit/user/${userId}?page=${page}&limit=${limit}`
    );
    return response.data;
  }

  // ============================================================================
  // Statistics API Methods
  // ============================================================================

  /**
   * Get user statistics
   */
  async getUserStats(): Promise<UserStats> {
    const response: AxiosResponse<UserStats> = await this.api.get('/stats/users');
    return response.data;
  }

  /**
   * Get alert statistics
   */
  async getAlertStats(): Promise<AlertStats> {
    const response: AxiosResponse<AlertStats> = await this.api.get('/stats/alerts');
    return response.data;
  }

  /**
   * Get schedule statistics
   */
  async getScheduleStats(): Promise<ScheduleStats> {
    const response: AxiosResponse<ScheduleStats> = await this.api.get('/stats/schedules');
    return response.data;
  }

  /**
   * Get audit statistics
   */
  async getAuditStats(): Promise<AuditStats> {
    const response: AxiosResponse<AuditStats> = await this.api.get('/stats/audit');
    return response.data;
  }

  // ============================================================================
  // Health Check Methods
  // ============================================================================

  /**
   * Check if the Telegram bot API is healthy
   */
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response: AxiosResponse<{ status: string; timestamp: string }> = await this.api.get('/health');
    return response.data;
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

/**
 * Default Telegram API service instance
 */
export const telegramApi = new TelegramApiService();

/**
 * Create a new Telegram API service instance with custom configuration
 */
export const createTelegramApiService = (config: Partial<TelegramApiConfig>): TelegramApiService => {
  return new TelegramApiService(config);
};

export default telegramApi;