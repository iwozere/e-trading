/**
 * RecentActivityTable Component
 * 
 * Displays recent command activity with live updates and filtering.
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  CircularProgress,
  Alert,
  TextField,
  InputAdornment,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Search as SearchIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { useTelegramAuditLogs } from '../../hooks/telegram/useTelegramAudit';
import { CommandAudit } from '../../types/telegram';

const RecentActivityTable: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const { data, isLoading, isError, error, refetch } = useTelegramAuditLogs({
    limit: 10,
    command: searchQuery || undefined
  });

  const activities = data?.data || [];

  if (isError) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Activity
          </Typography>
          <Alert severity="error">
            Failed to load activity log: {error?.message || 'Unknown error'}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Recent Activity
          </Typography>
          <Tooltip title="Refresh">
            <IconButton size="small" onClick={() => refetch()}>
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Search */}
        <TextField
          fullWidth
          size="small"
          placeholder="Filter by command..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          sx={{ mb: 2 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon fontSize="small" />
              </InputAdornment>
            ),
          }}
        />

        {isLoading ? (
          <Box display="flex" justifyContent="center" py={4}>
            <CircularProgress size={32} />
          </Box>
        ) : activities.length === 0 ? (
          <Box py={4} textAlign="center">
            <Typography variant="body2" color="textSecondary">
              {searchQuery ? 'No matching activity found' : 'No recent activity'}
            </Typography>
          </Box>
        ) : (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Time</TableCell>
                  <TableCell>User</TableCell>
                  <TableCell>Command</TableCell>
                  <TableCell align="center">Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {activities.map((activity: CommandAudit) => (
                  <TableRow key={activity.id} hover>
                    <TableCell>
                      <Typography variant="body2" color="textSecondary">
                        {new Date(activity.timestamp).toLocaleTimeString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" fontFamily="monospace">
                        {activity.telegram_user_id}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {activity.command}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Tooltip title={activity.success ? 'Success' : activity.error_message || 'Failed'}>
                        {activity.success ? (
                          <SuccessIcon color="success" fontSize="small" />
                        ) : (
                          <ErrorIcon color="error" fontSize="small" />
                        )}
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </CardContent>
    </Card>
  );
};

export default RecentActivityTable;