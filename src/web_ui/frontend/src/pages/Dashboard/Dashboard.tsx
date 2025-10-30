import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  IconButton,
  Button,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Refresh,
  TrendingUp,
  TrendingDown,
  Speed,
  Memory,
  Thermostat,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// API functions
import { getStrategies, getSystemStatus } from '../../api/tradingApi';
// import { useWebSocket } from '../../contexts/WebSocketContext';

// Types
interface StrategyStatus {
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

interface SystemStatus {
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

const Dashboard: React.FC = () => {
  // Temporarily disable WebSocket until backend implements it
  const isConnected = false;
  // const { isConnected, connectionStats } = useWebSocket();

  // Fetch strategies
  const {
    data: strategies = [],
    isLoading: strategiesLoading,
    refetch: refetchStrategies,
  } = useQuery<StrategyStatus[]>({
    queryKey: ['strategies'],
    queryFn: getStrategies,
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  // Fetch system status
  const {
    data: systemStatus,
    isLoading: systemLoading,
    refetch: refetchSystem,
  } = useQuery<SystemStatus>({
    queryKey: ['system-status'],
    queryFn: getSystemStatus,
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  // Calculate summary statistics
  const runningStrategies = strategies.filter(s => s.status === 'running').length;
  const stoppedStrategies = strategies.filter(s => s.status === 'stopped').length;
  const errorStrategies = strategies.filter(s => s.status === 'error').length;

  // Mock performance data for chart
  const performanceData = [
    { time: '00:00', pnl: 0 },
    { time: '04:00', pnl: 150 },
    { time: '08:00', pnl: 280 },
    { time: '12:00', pnl: 320 },
    { time: '16:00', pnl: 450 },
    { time: '20:00', pnl: 380 },
    { time: '24:00', pnl: 520 },
  ];

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'stopped':
        return 'default';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Trading Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Chip
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
            size="small"
          />
          <IconButton onClick={() => { refetchStrategies(); refetchSystem(); }}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* System Overview Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Strategies
              </Typography>
              <Typography variant="h4" component="div" color="success.main">
                {runningStrategies}
              </Typography>
              <Typography variant="body2">
                of {strategies.length} total
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                System Status
              </Typography>
              <Typography variant="h6" component="div">
                {systemStatus?.status || 'Unknown'}
              </Typography>
              <Typography variant="body2">
                Uptime: {systemStatus ? formatUptime(systemStatus.uptime_seconds) : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total P&L (Mock)
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <TrendingUp color="success" sx={{ mr: 1 }} />
                <Typography variant="h5" component="div" color="success.main">
                  +$520.00
                </Typography>
              </Box>
              <Typography variant="body2">
                +2.6% today
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Error Count
              </Typography>
              <Typography variant="h4" component="div" color={errorStrategies > 0 ? 'error.main' : 'text.primary'}>
                {errorStrategies}
              </Typography>
              <Typography variant="body2">
                strategies with errors
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* System Metrics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Metrics
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Speed sx={{ mr: 1, fontSize: 20 }} />
                  <Typography variant="body2" sx={{ flexGrow: 1 }}>
                    CPU Usage
                  </Typography>
                  <Typography variant="body2">
                    {systemStatus?.system_metrics.cpu_percent.toFixed(1) || '0.0'}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={systemStatus?.system_metrics.cpu_percent || 0}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Memory sx={{ mr: 1, fontSize: 20 }} />
                  <Typography variant="body2" sx={{ flexGrow: 1 }}>
                    Memory Usage
                  </Typography>
                  <Typography variant="body2">
                    {systemStatus?.system_metrics.memory_percent.toFixed(1) || '0.0'}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={systemStatus?.system_metrics.memory_percent || 0}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Thermostat sx={{ mr: 1, fontSize: 20 }} />
                  <Typography variant="body2" sx={{ flexGrow: 1 }}>
                    Temperature
                  </Typography>
                  <Typography variant="body2">
                    {systemStatus?.system_metrics.temperature_c.toFixed(1) || '0.0'}°C
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(systemStatus?.system_metrics.temperature_c || 0) / 80 * 100} // Assume 80°C max
                  color={systemStatus?.system_metrics.temperature_c > 70 ? 'error' : 'primary'}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Portfolio Performance (Mock)
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="pnl"
                    stroke="#1976d2"
                    strokeWidth={2}
                    dot={{ fill: '#1976d2' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Strategy List */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Strategy Overview
              </Typography>
              
              {strategiesLoading ? (
                <LinearProgress />
              ) : (
                <Grid container spacing={2}>
                  {strategies.map((strategy) => (
                    <Grid item xs={12} sm={6} md={4} key={strategy.instance_id}>
                      <Card variant="outlined">
                        <CardContent>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                            <Typography variant="subtitle1" component="div">
                              {strategy.name}
                            </Typography>
                            <Chip
                              label={strategy.status}
                              color={getStatusColor(strategy.status) as any}
                              size="small"
                            />
                          </Box>
                          
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            {strategy.symbol} • {strategy.broker_type} • {strategy.trading_mode}
                          </Typography>
                          
                          <Typography variant="body2" gutterBottom>
                            Uptime: {formatUptime(strategy.uptime_seconds)}
                          </Typography>
                          
                          {strategy.error_count > 0 && (
                            <Typography variant="body2" color="error" gutterBottom>
                              Errors: {strategy.error_count}
                            </Typography>
                          )}
                          
                          <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                            <Button
                              size="small"
                              startIcon={<PlayArrow />}
                              disabled={strategy.status === 'running'}
                            >
                              Start
                            </Button>
                            <Button
                              size="small"
                              startIcon={<Stop />}
                              disabled={strategy.status !== 'running'}
                            >
                              Stop
                            </Button>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;