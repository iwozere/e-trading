import React from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  Divider,
  Paper
} from '@mui/material';
import { 
  Memory as MemoryIcon,
  Timeline as TimelineIcon,
  Timer as TimerIcon,
  Speed as SpeedIcon
} from '@mui/icons-material';
import { useSystemStatus, useSystemMetrics } from '../../hooks/system/useSystemHealth';

interface MetricBoxProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  subtitle?: string;
}

const MetricBox: React.FC<MetricBoxProps> = ({ title, value, icon, subtitle }) => (
  <Paper elevation={0} variant="outlined" sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, color: 'text.secondary' }}>
      {icon}
      <Typography variant="body2" sx={{ ml: 1, fontWeight: 'medium' }}>
        {title}
      </Typography>
    </Box>
    <Typography variant="h4" component="div" sx={{ mb: 1 }}>
      {value}
    </Typography>
    {subtitle && (
      <Typography variant="caption" color="text.secondary">
        {subtitle}
      </Typography>
    )}
  </Paper>
);

const Monitoring: React.FC = () => {
  const { data: statusData, isLoading: statusLoading, error: statusError } = useSystemStatus();
  const { data: metricsData, isLoading: metricsLoading, error: metricsError } = useSystemMetrics();

  if (statusLoading || metricsLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  const formatUptime = (seconds?: number) => {
    if (!seconds) return 'N/A';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  };

  const cpuUsage = metricsData?.system_metrics?.cpu_percent || 0;
  const memUsage = metricsData?.system_metrics?.memory_percent || 0;

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Real-Time Monitoring
      </Typography>
      
      {(statusError || metricsError) && (
        <Alert severity="error" sx={{ mb: 3 }}>
          Failed to fetch monitoring data. The backend may be offline or endpoints are unavailable.
        </Alert>
      )}

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox 
            title="Service Version" 
            value={statusData?.version || 'N/A'} 
            icon={<TimelineIcon />} 
            subtitle={statusData?.service_name || 'Alkotrader'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox 
            title="Uptime" 
            value={formatUptime(statusData?.uptime_seconds)} 
            icon={<TimerIcon />} 
            subtitle="Time since last restart"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox 
            title="CPU Usage" 
            value={`${cpuUsage.toFixed(1)}%`} 
            icon={<SpeedIcon />} 
            subtitle="Current system load"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox 
            title="Memory Usage" 
            value={`${memUsage.toFixed(1)}%`} 
            icon={<MemoryIcon />} 
            subtitle="Current RAM usage"
          />
        </Grid>
      </Grid>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Trading Strategies
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Box sx={{ display: 'flex', gap: 4 }}>
            <Box>
              <Typography variant="body2" color="text.secondary">Active Strategies</Typography>
              <Typography variant="h5">{statusData?.active_strategies || 0}</Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">Total Configured</Typography>
              <Typography variant="h5">{statusData?.total_strategies || 0}</Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Monitoring;