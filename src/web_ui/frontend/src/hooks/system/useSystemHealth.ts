import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { systemApi } from '../../api/systemApi';

export const useSystemHealth = () => {
  return useQuery({
    queryKey: ['system', 'health'],
    queryFn: () => systemApi.getHealth(),
    refetchInterval: 30000,
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
    refetchInterval: 15000,
  });
};

export const useSystemMetrics = () => {
  return useQuery({
    queryKey: ['system', 'metrics'],
    queryFn: () => systemApi.getSystemMetrics(),
    refetchInterval: 15000,
  });
};

export const useAnalyticsDashboard = (days: number = 30) => {
  return useQuery({
    queryKey: ['analytics', 'dashboard', days],
    queryFn: () => systemApi.getAnalyticsDashboard(days),
    refetchOnWindowFocus: false,
  });
};

export const useServicesStatus = () => {
  return useQuery({
    queryKey: ['monitoring', 'services'],
    queryFn: () => systemApi.getServicesStatus(),
    refetchInterval: 30000,
  });
};

export const usePipelinesStatus = () => {
  return useQuery({
    queryKey: ['monitoring', 'pipelines'],
    queryFn: () => systemApi.getPipelinesStatus(),
    refetchInterval: (query) => {
      const data = query.state.data as { pipelines?: Array<{ last_status: string }> } | undefined;
      const anyRunning = data?.pipelines?.some((p) => p.last_status === 'running') ?? false;
      return anyRunning ? 5000 : 60000;
    },
  });
};

export const useTriggerPipeline = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (scheduleId: number) => systemApi.triggerPipeline(scheduleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['monitoring', 'pipelines'] });
    },
  });
};

export const useRunLogs = (runId: number | null) => {
  return useQuery({
    queryKey: ['runs', 'logs', runId],
    queryFn: () => systemApi.getRunLogs(runId!),
    enabled: runId !== null,
    staleTime: 30000,
  });
};
