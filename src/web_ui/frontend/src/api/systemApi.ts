import axios from 'axios';

// Relying on Vite proxy or relative path for API calls
const API_BASE_URL = '';

// Add interceptor to catch 401s globally
axios.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const systemApi = {
  getHealth: async () => {
    const response = await axios.get(`${API_BASE_URL}/api/health`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  },
  
  getChannelsHealth: async () => {
    const response = await axios.get(`${API_BASE_URL}/api/health/channels`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  },

  getSystemStatus: async () => {
    const response = await axios.get(`${API_BASE_URL}/api/system/status`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  },

  getSystemMetrics: async () => {
    const response = await axios.get(`${API_BASE_URL}/api/monitoring/metrics`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  },

  getAnalyticsDashboard: async (days: number = 30) => {
    const response = await axios.get(`${API_BASE_URL}/api/analytics/dashboard?days=${days}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  }
};


