import api from './tradingApi';

// Relying on Relative path for API calls via the central api instance
const API_BASE_URL = '';

export const systemApi = {
  getHealth: async () => {
    const response = await api.get(`/api/health`);
    return response.data;
  },

  getChannelsHealth: async () => {
    const response = await api.get(`/api/health/channels`);
    return response.data;
  },

  getSystemStatus: async () => {
    const response = await api.get(`/api/system/status`);
    return response.data;
  },

  getSystemMetrics: async () => {
    const response = await api.get(`/api/monitoring/metrics`);
    return response.data;
  },

  getAnalyticsDashboard: async (days: number = 30) => {
    const response = await api.get(`/api/analytics/dashboard?days=${days}`);
    return response.data;
  },

  getServicesStatus: async () => {
    const response = await api.get(`/api/monitoring/services`);
    return response.data;
  },

  getPipelinesStatus: async () => {
    const response = await api.get(`/api/monitoring/pipelines`);
    return response.data;
  },
};



