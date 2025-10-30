/**
 * Audit Logs Page
 * 
 * Interface for viewing and filtering Telegram bot command audit logs
 * and user activity monitoring.
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
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Search as SearchIcon
} from '@mui/icons-material';
import TelegramBreadcrumbs from '../../components/Telegram/TelegramBreadcrumbs';
import { useTelegramAuditLogs } from '../../hooks/telegram/useTelegramAudit';

const AuditLogs: React.FC = () => {
  const [filters, setFilters] = useState({
    user_id: '',
    command: '',
    success_only: '',
    page: 1,
    page_size: 50
  });

  // Query for audit logs
  const { data: auditResponse, isLoading, isError, error, refetch } = useTelegramAuditLogs(filters);

  const logs = auditResponse?.data || [];

  const handleFilterChange = (field: string, value: string) => {
    setFilters(prev => ({
      ...prev,
      [field]: value,
      page: 1 // Reset to first page when filtering
    }));
  };

  const getSuccessChip = (success: boolean) => {
    if (success) {
      return <Chip label="Success" color="success" size="small" />;
    }
    return <Chip label="Failed" color="error" size="small" />;
  };

  if (isError) {
    return (
      <Box sx={{ p: 3 }}>
        <TelegramBreadcrumbs />
        <Typography variant="h4" gutterBottom>
          Audit Logs
        </Typography>
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to load audit logs: {error?.message || 'Unknown error'}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <TelegramBreadcrumbs />
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          Audit Logs
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={() => refetch()}
          disabled={isLoading}
        >
          Refresh
        </Button>
      </Box>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Filters
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                size="small"
                label="User ID"
                value={filters.user_id}
                onChange={(e) => handleFilterChange('user_id', e.target.value)}
                placeholder="Filter by user ID"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                size="small"
                label="Command"
                value={filters.command}
                onChange={(e) => handleFilterChange('command', e.target.value)}
                placeholder="Filter by command"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Status</InputLabel>
                <Select
                  value={filters.success_only}
                  label="Status"
                  onChange={(e) => handleFilterChange('success_only', e.target.value)}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="true">Success Only</MenuItem>
                  <MenuItem value="false">Failed Only</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Button
                fullWidth
                variant="contained"
                startIcon={<SearchIcon />}
                onClick={() => refetch()}
                disabled={isLoading}
              >
                Search
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
      
      {isLoading ? (
        <Card>
          <CardContent>
            <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
              <CircularProgress />
              <Typography variant="body1" sx={{ ml: 2 }}>
                Loading audit logs...
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
                    <TableCell>User ID</TableCell>
                    <TableCell>Command</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Response Time</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Error</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {logs.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7} align="center">
                        <Typography variant="body2" color="text.secondary">
                          No audit logs found
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    logs.map((log: any) => (
                      <TableRow key={log.id}>
                        <TableCell>{log.id}</TableCell>
                        <TableCell>{log.telegram_user_id}</TableCell>
                        <TableCell>
                          <Chip label={log.command} variant="outlined" size="small" />
                        </TableCell>
                        <TableCell>{getSuccessChip(log.success)}</TableCell>
                        <TableCell>
                          {log.response_time_ms ? `${log.response_time_ms}ms` : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {log.created ? new Date(log.created).toLocaleString() : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {log.error_message ? (
                            <Typography variant="caption" color="error">
                              {log.error_message.length > 50 
                                ? `${log.error_message.substring(0, 50)}...`
                                : log.error_message
                              }
                            </Typography>
                          ) : (
                            '-'
                          )}
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
    </Box>
  );
};

export default AuditLogs;