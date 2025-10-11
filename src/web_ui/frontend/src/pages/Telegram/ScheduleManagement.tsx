/**
 * Schedule Management Page
 * 
 * Interface for managing Telegram bot scheduled reports including
 * daily and weekly report configurations.
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
  Tooltip
} from '@mui/material';
import {
  PlayArrow as ActivateIcon,
  Pause as DeactivateIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Edit as EditIcon
} from '@mui/icons-material';
import TelegramBreadcrumbs from '../../components/Telegram/TelegramBreadcrumbs';
import { useTelegramSchedules } from '../../hooks/telegram/useTelegramSchedules';

const ScheduleManagement: React.FC = () => {
  const [filter, setFilter] = useState<string>('all');

  // Query for schedules
  const { data: schedulesResponse, isLoading, isError, error, refetch } = useTelegramSchedules({ 
    status: filter === 'all' ? undefined : filter as any
  });

  const schedules = schedulesResponse?.data || [];

  const getStatusChip = (schedule: any) => {
    if (schedule.active) {
      return <Chip label="Active" color="success" size="small" />;
    }
    return <Chip label="Inactive" color="default" size="small" />;
  };

  if (isError) {
    return (
      <Box sx={{ p: 3 }}>
        <TelegramBreadcrumbs />
        <Typography variant="h4" gutterBottom>
          Schedule Management
        </Typography>
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to load schedules: {error?.message || 'Unknown error'}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <TelegramBreadcrumbs />
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          Schedule Management
        </Typography>
        <Box display="flex" gap={2} alignItems="center">
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Filter</InputLabel>
            <Select
              value={filter}
              label="Filter"
              onChange={(e) => setFilter(e.target.value)}
            >
              <MenuItem value="all">All Schedules</MenuItem>
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
                Loading schedules...
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
                    <TableCell>Schedule Time</TableCell>
                    <TableCell>Period</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {schedules.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={8} align="center">
                        <Typography variant="body2" color="text.secondary">
                          No schedules found
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    schedules.map((schedule: any) => (
                      <TableRow key={schedule.id}>
                        <TableCell>{schedule.id}</TableCell>
                        <TableCell>{schedule.user_id}</TableCell>
                        <TableCell>
                          <Chip label={schedule.ticker} variant="outlined" size="small" />
                        </TableCell>
                        <TableCell>{schedule.scheduled_time}</TableCell>
                        <TableCell>
                          {schedule.period ? (
                            <Chip label={schedule.period} size="small" />
                          ) : (
                            'One-time'
                          )}
                        </TableCell>
                        <TableCell>{getStatusChip(schedule)}</TableCell>
                        <TableCell>
                          {schedule.created ? new Date(schedule.created).toLocaleDateString() : 'N/A'}
                        </TableCell>
                        <TableCell align="center">
                          <Box display="flex" gap={1} justifyContent="center">
                            <Tooltip title={schedule.active ? "Deactivate" : "Activate"}>
                              <IconButton
                                size="small"
                                color={schedule.active ? "warning" : "success"}
                                onClick={() => {
                                  // TODO: Implement toggle functionality
                                  console.log('Toggle schedule', schedule.id);
                                }}
                              >
                                {schedule.active ? <DeactivateIcon /> : <ActivateIcon />}
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Edit">
                              <IconButton
                                size="small"
                                color="info"
                                onClick={() => {
                                  // TODO: Implement edit functionality
                                  console.log('Edit schedule', schedule.id);
                                }}
                              >
                                <EditIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Delete">
                              <IconButton
                                size="small"
                                color="error"
                                onClick={() => {
                                  // TODO: Implement delete functionality
                                  console.log('Delete schedule', schedule.id);
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
    </Box>
  );
};

export default ScheduleManagement;