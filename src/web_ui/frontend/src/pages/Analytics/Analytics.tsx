import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper
} from '@mui/material';
import { 
  BarChart as BarChartIcon,
  Timeline as TimelineIcon,
  CheckCircleOutline as CheckIcon,
  NotificationsActive as AlertIcon
} from '@mui/icons-material';
import { useAnalyticsDashboard } from '../../hooks/system/useSystemHealth';

interface KPIBoxProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color?: string;
  subtitle?: string;
}

const KPIBox: React.FC<KPIBoxProps> = ({ title, value, icon, color = 'primary.main', subtitle }) => (
  <Paper elevation={0} variant="outlined" sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, color }}>
      {icon}
      <Typography variant="body2" sx={{ ml: 1, fontWeight: 'medium', color: 'text.secondary' }}>
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

const Analytics: React.FC = () => {
  const [days, setDays] = useState<number>(30);
  const { data, isLoading, error } = useAnalyticsDashboard(days);

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  const notifStats = data?.analytics?.notifications || {};
  const sysStats = data?.analytics?.system || {};

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Performance Analytics
        </Typography>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Time Range</InputLabel>
          <Select
            value={days}
            label="Time Range"
            onChange={(e) => setDays(Number(e.target.value))}
          >
            <MenuItem value={7}>Last 7 Days</MenuItem>
            <MenuItem value={30}>Last 30 Days</MenuItem>
            <MenuItem value={90}>Last 90 Days</MenuItem>
          </Select>
        </FormControl>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          Failed to fetch analytics data. The backend may be offline or endpoints are unavailable.
        </Alert>
      )}

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <KPIBox 
            title="Total Notifications" 
            value={notifStats.total_sent || 0} 
            icon={<AlertIcon />} 
            subtitle={`In the last ${days} days`}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <KPIBox 
            title="Delivery Success Rate" 
            value={`${(notifStats.success_rate || 0).toFixed(1)}%`} 
            icon={<CheckIcon />} 
            color="success.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <KPIBox 
            title="System Events" 
            value={sysStats.total_events || 0} 
            icon={<TimelineIcon />} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <KPIBox 
            title="Active Users" 
            value={data?.analytics?.users?.active || 0} 
            icon={<BarChartIcon />} 
          />
        </Grid>
      </Grid>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Detailed Analytics (Coming Soon)
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="body1" color="text.secondary">
            Charts and deeper analysis correlating notification metrics with trading performance will be available here in a future update.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Analytics;