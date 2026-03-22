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
  useTheme,
  Divider,
  Paper,
  MenuItem,
  Skeleton,
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
  ArrowUpward,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
} from 'recharts';
import { motion } from 'framer-motion';

// API functions
import { getStrategies, getSystemStatus } from '../../api/tradingApi';

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

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: 'spring' as const,
      stiffness: 100
    }
  }
};

const MetricCard: React.FC<{ 
  title: string; 
  value: string | number; 
  subtitle: string; 
  icon: React.ReactNode; 
  color?: string;
  trend?: string;
}> = ({ title, value, subtitle, icon, color = 'primary.main', trend }) => (
  <motion.div variants={itemVariants}>
    <Card sx={{ height: '100%', position: 'relative', overflow: 'hidden' }}>
      <Box sx={{ 
        position: 'absolute', 
        top: -10, 
        right: -10, 
        opacity: 0.1, 
        transform: 'rotate(15deg)',
        fontSize: '5rem',
        color 
      }}>
        {icon}
      </Box>
      <CardContent sx={{ position: 'relative', zIndex: 1 }}>
        <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 700, letterSpacing: '0.1em' }}>
          {title}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, mt: 1, mb: 0.5 }}>
          <Typography variant="h3" sx={{ fontWeight: 800, color }}>
            {value}
          </Typography>
          {trend && (
            <Box sx={{ display: 'flex', alignItems: 'center', color: 'success.main', fontSize: '0.875rem', fontWeight: 600 }}>
              <ArrowUpward sx={{ fontSize: '1rem', mr: 0.25 }} />
              {trend}
            </Box>
          )}
        </Box>
        <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
          {subtitle}
        </Typography>
      </CardContent>
    </Card>
  </motion.div>
);

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <Paper 
        elevation={10} 
        sx={{ 
          p: 2, 
          bgcolor: 'rgba(26, 29, 58, 0.9)', 
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 2
        }}
      >
        <Typography variant="overline" color="text.secondary" display="block">
          Time: {label}
        </Typography>
        <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 800 }}>
          PnL: ${payload[0].value.toFixed(2)}
        </Typography>
      </Paper>
    );
  }
  return null;
};

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const isConnected = true;

  // Fetch strategies
  const {
    data: strategies = [],
    isLoading: strategiesLoading,
    refetch: refetchStrategies,
  } = useQuery<StrategyStatus[]>({
    queryKey: ['strategies'],
    queryFn: getStrategies,
    refetchInterval: 30000,
  });

  // Fetch system status
  const {
    data: systemStatus,
    isLoading: systemLoading,
    refetch: refetchSystem,
  } = useQuery<SystemStatus>({
    queryKey: ['system-status'],
    queryFn: getSystemStatus,
    refetchInterval: 10000,
  });

  const runningStrategies = strategies.filter(s => s.status === 'running').length;
  const errorStrategies = strategies.filter(s => s.status === 'error').length;

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

  if (strategiesLoading || systemLoading) {
    return (
      <Box sx={{ p: 4, maxWidth: 1600, mx: 'auto' }}>
        <Box sx={{ mb: 4 }}>
          <Skeleton variant="text" width={300} height={60} />
          <Skeleton variant="text" width={450} height={30} />
        </Box>
        <Grid container spacing={4}>
          {[1, 2, 3, 4].map(i => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Skeleton variant="rectangular" height={140} sx={{ borderRadius: 2 }} />
            </Grid>
          ))}
          <Grid item xs={12} md={8}>
            <Skeleton variant="rectangular" height={400} sx={{ borderRadius: 2 }} />
          </Grid>
          <Grid item xs={12} md={4}>
            <Skeleton variant="rectangular" height={400} sx={{ borderRadius: 2 }} />
          </Grid>
        </Grid>
      </Box>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'success';
      case 'stopped': return 'default';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box 
      component={motion.div}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      sx={{ p: 4, maxWidth: 1600, mx: 'auto' }}
    >
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <Box>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 800 }}>
            Trading Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            System pulse and active strategy performance
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Chip
            label={isConnected ? 'REAL-TIME ACTIVE' : 'DISCONNECTED'}
            sx={{ 
              bgcolor: isConnected ? 'rgba(0, 229, 255, 0.1)' : 'rgba(255, 0, 0, 0.1)',
              color: isConnected ? 'primary.main' : 'error.main',
              fontWeight: 700,
              fontSize: '0.75rem',
              border: '1px solid currentColor',
              height: 28,
              '& .MuiChip-label': { px: 2 }
            }}
          />
          <IconButton onClick={() => { refetchStrategies(); refetchSystem(); }} sx={{ border: '1px solid rgba(255,255,255,0.1)' }}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      <Grid container spacing={4}>
        {/* Metric Cards Row */}
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Active Strategies" 
            value={runningStrategies} 
            subtitle={`of ${strategies.length} configured`} 
            icon={<PlayArrow />}
            color="primary.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Portfolio Balance" 
            value="+$520" 
            subtitle="Net profit/loss today" 
            icon={<TrendingUp />}
            color="success.main"
            trend="2.6%"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="System Pulse" 
            value={systemStatus?.status?.toUpperCase() || 'UP'} 
            subtitle={`Uptime: ${systemStatus ? formatUptime(systemStatus.uptime_seconds) : 'N/A'}`} 
            icon={<Speed />}
            color="secondary.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Critical Alerts" 
            value={errorStrategies} 
            subtitle="Errors requiring attention" 
            icon={<Stop />}
            color={errorStrategies > 0 ? 'error.main' : 'text.disabled'}
          />
        </Grid>

        {/* Charts & Metrics Row */}
        <Grid item xs={12} md={8}>
          <motion.div variants={itemVariants}>
            <Card sx={{ p: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
                  Aggregate Performance
                </Typography>
                <Box sx={{ height: 350, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={performanceData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={theme.palette.primary.main} stopOpacity={0.3}/>
                          <stop offset="95%" stopColor={theme.palette.primary.main} stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                      <XAxis 
                        dataKey="time" 
                        axisLine={false} 
                        tickLine={false} 
                        tick={{ fill: theme.palette.text.secondary, fontSize: 12 }} 
                        dy={10}
                      />
                      <YAxis 
                        axisLine={false} 
                        tickLine={false} 
                        tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <Area 
                        type="monotone" 
                        dataKey="pnl" 
                        stroke={theme.palette.primary.main} 
                        strokeWidth={3}
                        fillOpacity={1} 
                        fill="url(#colorPnl)" 
                        animationDuration={2000}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={4}>
          <motion.div variants={itemVariants}>
            <Card sx={{ height: '100%', p: 1 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
                  System Health
                </Typography>
                
                {[
                  { label: 'CPU Cluster', icon: <Speed />, value: systemStatus?.system_metrics.cpu_percent || 0, unit: '%' },
                  { label: 'Memory Bank', icon: <Memory />, value: systemStatus?.system_metrics.memory_percent || 0, unit: '%' },
                  { label: 'Thermal Core', icon: <Thermostat />, value: systemStatus?.system_metrics.temperature_c || 0, unit: '°C', max: 80 }
                ].map((metric, idx) => (
                  <Box key={idx} sx={{ mb: 4 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1.5 }}>
                      <Box sx={{ p: 0.75, borderRadius: 1.5, bgcolor: 'rgba(255,255,255,0.05)', mr: 2, display: 'flex' }}>
                        {metric.icon}
                      </Box>
                      <Typography variant="body2" sx={{ flexGrow: 1, fontWeight: 500 }}>
                        {metric.label}
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 700, color: 'primary.main' }}>
                        {metric.value.toFixed(1)}{metric.unit}
                      </Typography>
                    </Box>
                    <Box sx={{ width: '100%', bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 4, height: 6 }}>
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${(metric.value / (metric.max || 100)) * 100}%` }}
                        transition={{ duration: 1.5, ease: 'easeOut' }}
                        style={{ 
                          height: '100%', 
                          borderRadius: 4, 
                          backgroundColor: metric.value > 75 ? theme.palette.error.main : theme.palette.primary.main,
                          boxShadow: `0 0 10px ${metric.value > 75 ? theme.palette.error.main : theme.palette.primary.main}80`
                        }}
                      />
                    </Box>
                  </Box>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Strategy Grid */}
        <Grid item xs={12}>
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6" gutterBottom sx={{ mb: 3, display: 'flex', alignItems: 'center' }}>
              Execution Engines
              <Chip label={strategies.length} size="small" sx={{ ml: 2, bgcolor: 'rgba(255,255,255,0.05)', fontWeight: 600 }} />
            </Typography>
            <Grid container spacing={3}>
              {strategies.map((strategy) => (
                <Grid item xs={12} sm={6} md={4} key={strategy.instance_id}>
                  <motion.div 
                    variants={itemVariants}
                    whileHover={{ y: -5, transition: { duration: 0.2 } }}
                  >
                    <Card variant="outlined" sx={{ 
                      bgcolor: 'rgba(255,255,255,0.02)', 
                      borderColor: 'rgba(255,255,255,0.05)',
                      '&:hover': { borderColor: 'primary.main', bgcolor: 'rgba(0, 229, 255, 0.02)' }
                    }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                          <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                            {strategy.name}
                          </Typography>
                          <Chip
                            label={strategy.status.toUpperCase()}
                            color={getStatusColor(strategy.status) as any}
                            size="small"
                            sx={{ fontWeight: 800, fontSize: '0.65rem', height: 20 }}
                          />
                        </Box>
                        
                        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 2 }}>
                          {strategy.symbol} • {strategy.broker_type} • {strategy.trading_mode}
                        </Typography>
                        
                        <Grid container spacing={1} sx={{ mb: 2 }}>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="text.secondary">UPTIME</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>{formatUptime(strategy.uptime_seconds)}</Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="text.secondary">ERRORS</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 600, color: strategy.error_count > 0 ? 'error.main' : 'inherit' }}>
                              {strategy.error_count}
                            </Typography>
                          </Grid>
                        </Grid>
                        
                        <Divider sx={{ mb: 2, opacity: 0.5 }} />
                        
                        <Box sx={{ display: 'flex', gap: 1.5 }}>
                          <Button
                            variant="outlined"
                            fullWidth
                            size="small"
                            startIcon={<PlayArrow />}
                            disabled={strategy.status === 'running'}
                            sx={{ borderStyle: 'dashed' }}
                          >
                            RUN
                          </Button>
                          <Button
                            variant="outlined"
                            fullWidth
                            size="small"
                            startIcon={<Stop />}
                            disabled={strategy.status !== 'running'}
                            color="error"
                            sx={{ borderStyle: 'dashed' }}
                          >
                            HALT
                          </Button>
                        </Box>
                      </CardContent>
                    </Card>
                  </motion.div>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;