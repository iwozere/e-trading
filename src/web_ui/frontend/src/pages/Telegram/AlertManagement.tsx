/**
 * Alert Management Page
 * 
 * Interface for managing Telegram bot price alerts including
 * creation, modification, and monitoring of alert configurations.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  Dialog
} from '@mui/material';
import {
  PlayArrow as ActivateIcon,
  Pause as DeactivateIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Add as AddIcon
} from '@mui/icons-material';
import TelegramBreadcrumbs from '../../components/Telegram/TelegramBreadcrumbs';
import { useTelegramAlerts } from '../../hooks/telegram/useTelegramAlerts';
import ConfigBuilder from '../../components/Telegram/ConfigBuilder';

const flattenConditionList = (alert: any) => {
  if (alert.condition) return alert.condition;
  if (!alert.config || !alert.config.rule) return 'Unknown';
  
  const rule = alert.config.rule;
  if (rule.operator && rule.conditions) {
    return `${rule.operator.toUpperCase()} (${rule.conditions.length} conditions)`;
  }
  
  return `${rule.indicator} ${rule.comparison} ${rule.value}`;
};

const AlertManagement: React.FC = () => {
  const [filter, setFilter] = useState<string>('all');
  const [builderOpen, setBuilderOpen] = useState(false);

  // Query for alerts
  const { data: alertsResponse, isLoading, isError, error, refetch } = useTelegramAlerts({ 
    status: filter === 'all' ? undefined : filter as any
  });

  const alerts = alertsResponse?.data || [];

  const getStatusChip = (alert: any) => {
    if (alert.active) {
      return <Chip label="Active" color="success" size="small" />;
    }
    return <Chip label="Inactive" color="default" size="small" />;
  };

  const handleSaveConfig = (config: any, mode: 'alert' | 'schedule') => {
    // TODO: Connect to backend API `useCreateTelegramAlert` mutation
    console.log('Saving config to API:', { mode, config });
    alert('Config generated successfully. Ready to link to backend API.');
    setBuilderOpen(false);
  };

  if (isError) {
    return (
      <Box sx={{ p: 3 }}>
        <TelegramBreadcrumbs />
        <Typography variant="h4" gutterBottom>
          Alert Management
        </Typography>
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to load alerts: {error?.message || 'Unknown error'}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <TelegramBreadcrumbs />
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          Alert Management
        </Typography>
        <Box display="flex" gap={2} alignItems="center">
          <Button 
            variant="contained" 
            startIcon={<AddIcon />}
            onClick={() => setBuilderOpen(true)}
          >
            Create Complex Alert
          </Button>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Filter</InputLabel>
            <Select
              value={filter}
              label="Filter"
              onChange={(e) => setFilter(e.target.value)}
            >
              <MenuItem value="all">All Alerts</MenuItem>
              <MenuItem value="active">Active</MenuItem>
              <MenuItem value="inactive">Inactive</MenuItem>
            </Select>
          </FormControl>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => refetch()}
            disabled={isLoading}
          >
            Refresh
          </Button>
        </Box>
      </Box>
      
      {isLoading ? (
        <Card>
          <CardContent>
            <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
              <CircularProgress />
              <Typography variant="body1" sx={{ ml: 2 }}>
                Loading alerts...
              </Typography>
            </Box>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent>
            <TableContainer component={Paper} elevation={0}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>ID</TableCell>
                    <TableCell>User</TableCell>
                    <TableCell>Ticker</TableCell>
                    <TableCell>Condition</TableCell>
                    <TableCell>Price</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {alerts.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={8} align="center">
                        <Typography variant="body2" color="text.secondary">
                          No alerts found
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    alerts.map((alert: any) => (
                      <TableRow key={alert.id}>
                        <TableCell>{alert.id}</TableCell>
                        <TableCell>{alert.user_id}</TableCell>
                        <TableCell>
                          <Chip label={alert.ticker} variant="outlined" size="small" />
                        </TableCell>
                        <TableCell>{flattenConditionList(alert)}</TableCell>
                        <TableCell>
                          {alert.price ? `$${alert.price}` : 'N/A'}
                        </TableCell>
                        <TableCell>{getStatusChip(alert)}</TableCell>
                        <TableCell>
                          {alert.created ? new Date(alert.created).toLocaleDateString() : 'N/A'}
                        </TableCell>
                        <TableCell align="center">
                          <Box display="flex" gap={1} justifyContent="center">
                            <Tooltip title={alert.active ? "Deactivate" : "Activate"}>
                              <IconButton
                                size="small"
                                color={alert.active ? "warning" : "success"}
                                onClick={() => {
                                  console.log('Toggle alert', alert.id);
                                }}
                              >
                                {alert.active ? <DeactivateIcon /> : <ActivateIcon />}
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Delete">
                              <IconButton
                                size="small"
                                color="error"
                                onClick={() => {
                                  console.log('Delete alert', alert.id);
                                }}
                              >
                                <DeleteIcon />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      <Dialog 
        open={builderOpen} 
        onClose={() => setBuilderOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <ConfigBuilder 
          onSave={handleSaveConfig} 
          onCancel={() => setBuilderOpen(false)} 
          initialMode="alert"
        />
      </Dialog>
    </Box>
  );
};

export default AlertManagement;