import axios from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5003';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth-storage');
  if (token) {
    try {
      const authData = JSON.parse(token);
      if (authData.state?.token) {
        config.headers.Authorization = `Bearer ${authData.state.token}`;
      }
    } catch (error) {
      console.error('Error parsing auth token:', error);
    }
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized - redirect to login
      localStorage.removeItem('auth-storage');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Types
export interface StrategyStatus {
  instance_id: string;
  name: string;
  status: string;
  uptime_seconds: number;
  error_count: number;
  last_error?: string;
  broker_type?: string;
  trading_mode?: string;
  symbol?: string;
  strategy_type?: string;
}

export interface StrategyConfig {
  id: string;
  name: string;
  enabled: boolean;
  symbol: string;
  broker: {
    type: string;
    trading_mode: string;
    name: string;
    cash: number;
    paper_trading_config?: any;
    live_trading_confirmed?: boolean;
  };
  strategy: {
    type: string;
    parameters: any;
  };
  data?: any;
  trading?: any;
  risk_management: any;
  notifications: any;
}

export interface SystemStatus {
  service_name: string;
  version: string;
  status: string;
  uptime_seconds: number;
  active_strategies: number;
  total_strategies: number;
  system_metrics: {
    cpu_percent: number;
    memory_percent: number;
    temperature_c: number;
    disk_usage_percent: number;
  };
}

export interface StrategyAction {
  action: string;
  confirm_live_trading?: boolean;
}

// API Functions

// Authentication
export const login = async (username: string, password: string) => {
  const response = await api.post('/auth/login', { username, password });
  return response.data;
};

export const logout = async () => {
  const response = await api.post('/auth/logout');
  return response.data;
};

// Health Check
export const healthCheck = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

// Strategy Management
export const getStrategies = async (): Promise<StrategyStatus[]> => {
  const response = await api.get('/api/strategies');
  return response.data;
};

export const getStrategy = async (id: string): Promise<StrategyConfig> => {
  const response = await api.get(`/api/strategies/${id}`);
  return response.data;
};

export const createStrategy = async (strategy: StrategyConfig) => {
  const response = await api.post('/api/strategies', strategy);
  return response.data;
};

export const updateStrategy = async (id: string, strategy: StrategyConfig) => {
  const response = await api.put(`/api/strategies/${id}`, strategy);
  return response.data;
};

export const deleteStrategy = async (id: string) => {
  const response = await api.delete(`/api/strategies/${id}`);
  return response.data;
};

// Strategy Lifecycle
export const startStrategy = async (id: string, confirmLiveTrading = false) => {
  const response = await api.post(`/api/strategies/${id}/start`, {
    action: 'start',
    confirm_live_trading: confirmLiveTrading,
  });
  return response.data;
};

export const stopStrategy = async (id: string) => {
  const response = await api.post(`/api/strategies/${id}/stop`, {
    action: 'stop',
  });
  return response.data;
};

export const restartStrategy = async (id: string, confirmLiveTrading = false) => {
  const response = await api.post(`/api/strategies/${id}/restart`, {
    action: 'restart',
    confirm_live_trading: confirmLiveTrading,
  });
  return response.data;
};

// System Monitoring
export const getSystemStatus = async (): Promise<SystemStatus> => {
  const response = await api.get('/api/system/status');
  return response.data;
};

// Configuration Management
export const getStrategyTemplates = async () => {
  const response = await api.get('/api/config/templates');
  return response.data;
};

export const validateConfiguration = async (config: any) => {
  const response = await api.post('/api/config/validate', config);
  return response.data;
};

// Performance Analytics (placeholder)
export const getPerformanceData = async (strategyId?: string, timeRange?: string) => {
  const params = new URLSearchParams();
  if (strategyId) params.append('strategy_id', strategyId);
  if (timeRange) params.append('time_range', timeRange);
  
  const response = await api.get(`/api/analytics/performance?${params}`);
  return response.data;
};

export const getTradeHistory = async (strategyId?: string, limit = 100) => {
  const params = new URLSearchParams();
  if (strategyId) params.append('strategy_id', strategyId);
  params.append('limit', limit.toString());
  
  const response = await api.get(`/api/analytics/trades?${params}`);
  return response.data;
};

// System Administration
export const startTradingService = async () => {
  const response = await api.post('/api/admin/service/start');
  return response.data;
};

export const stopTradingService = async () => {
  const response = await api.post('/api/admin/service/stop');
  return response.data;
};

export const restartTradingService = async () => {
  const response = await api.post('/api/admin/service/restart');
  return response.data;
};

export const getSystemLogs = async (lines = 100) => {
  const response = await api.get(`/api/admin/logs?lines=${lines}`);
  return response.data;
};

export const createBackup = async () => {
  const response = await api.post('/api/admin/backup');
  return response.data;
};

export const restoreBackup = async (backupId: string) => {
  const response = await api.post('/api/admin/restore', { backup_id: backupId });
  return response.data;
};

export default api;