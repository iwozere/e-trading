import { useQuery } from '@tanstack/react-query';
import { systemApi } from '../../api/systemApi';

export const useSystemHealth = () => {
  return useQuery({
    queryKey: ['system', 'health'],
    queryFn: () => systemApi.getHealth(),
    refetchInterval: 30000, // Refetch every 30 seconds
  });
};

export const useChannelsHealth = () => {
  return useQuery({
    queryKey: ['system', 'health', 'channels'],
    queryFn: () => systemApi.getChannelsHealth(),
    refetchInterval: 30000,
  });
};

export const useSystemStatus = () => {
  return useQuery({
    queryKey: ['system', 'status'],
    queryFn: () => systemApi.getSystemStatus(),
    refetchInterval: 15000, // Real-time
  });
};

export const useSystemMetrics = () => {
  return useQuery({
    queryKey: ['system', 'metrics'],
    queryFn: () => systemApi.getSystemMetrics(),
    refetchInterval: 15000, // Real-time
  });
};

export const useAnalyticsDashboard = (days: number = 30) => {
  return useQuery({
    queryKey: ['analytics', 'dashboard', days],
    queryFn: () => systemApi.getAnalyticsDashboard(days),
    refetchOnWindowFocus: false, // Don't refetch automatically
  });
};

