import React from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Grid, 
  CircularProgress, 
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Divider
} from '@mui/material';
import { 
  CheckCircle as CheckCircleIcon, 
  Error as ErrorIcon,
  Storage as StorageIcon,
  Api as ApiIcon,
  NotificationsActive as NotificationsIcon,
  Computer as ComputerIcon
} from '@mui/icons-material';
import { useSystemHealth, useChannelsHealth } from '../../hooks/system/useSystemHealth';

const StatusIcon = ({ status }: { status: string }) => {
  return status === 'ok' ? 
    <CheckCircleIcon color="success" /> : 
    <ErrorIcon color="error" />;
};

const Administration: React.FC = () => {
  const { data: healthData, isLoading: healthLoading, error: healthError } = useSystemHealth();
  const { data: channelsData, isLoading: channelsLoading, error: channelsError } = useChannelsHealth();

  if (healthLoading || channelsLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        System Administration
      </Typography>

      {(healthError || channelsError) && (
        <Alert severity="error" sx={{ mb: 3 }}>
          Failed to fetch system data. Please ensure the backend is running.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Core System Health */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <ComputerIcon sx={{ mr: 1 }} /> Core System Health
              </Typography>
              <Divider sx={{ mb: 2 }} />
              {healthData ? (
                <List dense>
                  <ListItem>
                    <ListItemIcon><ApiIcon /></ListItemIcon>
                    <ListItemText primary="API Status" />
                    <StatusIcon status={healthData.api} />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><StorageIcon /></ListItemIcon>
                    <ListItemText primary="Database Status" />
                    <StatusIcon status={healthData.database} />
                  </ListItem>
                  <ListItem>
                    <ListItemText primary="Overall Status" />
                    <Chip 
                      label={healthData.status?.toUpperCase() || 'UNKNOWN'} 
                      color={healthData.status === 'ok' ? 'success' : 'error'} 
                      size="small" 
                    />
                  </ListItem>
                  {healthData.timestamp && (
                    <ListItem>
                      <ListItemText primary="Last Updated" secondary={new Date(healthData.timestamp).toLocaleString()} />
                    </ListItem>
                  )}
                </List>
              ) : (
                <Typography color="text.secondary">Data unavailable</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Channels Health */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <NotificationsIcon sx={{ mr: 1 }} /> Notification Channels
              </Typography>
              <Divider sx={{ mb: 2 }} />
              {channelsData?.services ? (
                <List dense>
                  {Object.entries(channelsData.services).map(([service, status]: [string, any]) => (
                    <ListItem key={service}>
                      <ListItemText 
                        primary={service.replace(/_/g, ' ').toUpperCase()} 
                        secondary={status.error || 'System Operational'}
                      />
                      <StatusIcon status={status.status} />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography color="text.secondary">Data unavailable</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Administration;