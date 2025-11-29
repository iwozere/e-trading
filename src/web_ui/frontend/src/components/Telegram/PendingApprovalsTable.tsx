/**
 * PendingApprovalsTable Component
 * 
 * Displays users pending approval with quick action buttons
 * for approve/reject operations.
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
  Button,
  Chip,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip,
  TextField,
  InputAdornment
} from '@mui/material';
import {
  CheckCircle as ApproveIcon,
  Cancel as RejectIcon,
  Search as SearchIcon,
  Email as EmailIcon
} from '@mui/icons-material';
import { useTelegramUsers, useApproveTelegramUser } from '../../hooks/telegram/useTelegramUsers';
import { TelegramUser } from '../../types/telegram';

const PendingApprovalsTable: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const { data: users, isLoading, isError, error } = useTelegramUsers('pending');
  const approveMutation = useApproveTelegramUser();

  const handleApprove = async (userId: string) => {
    try {
      await approveMutation.mutateAsync(userId);
    } catch (err) {
      console.error('Failed to approve user:', err);
    }
  };

  const handleReject = async (userId: string) => {
    // For now, rejection is not implemented in the backend
    console.log('Reject user:', userId);
  };

  const filteredUsers = users?.filter((user: TelegramUser) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      user.telegram_user_id.toLowerCase().includes(query) ||
      user.email?.toLowerCase().includes(query)
    );
  }) || [];

  if (isError) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Pending Approvals
          </Typography>
          <Alert severity="error">
            Failed to load pending approvals: {error?.message || 'Unknown error'}
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
            Pending Approvals
          </Typography>
          <Chip
            label={`${filteredUsers.length} pending`}
            color="warning"
            size="small"
          />
        </Box>

        {/* Search */}
        <TextField
          fullWidth
          size="small"
          placeholder="Search by Telegram ID or email..."
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
        ) : filteredUsers.length === 0 ? (
          <Box py={4} textAlign="center">
            <Typography variant="body2" color="textSecondary">
              {searchQuery ? 'No matching users found' : 'No pending approvals'}
            </Typography>
          </Box>
        ) : (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Telegram ID</TableCell>
                  <TableCell>Email</TableCell>
                  <TableCell align="center">Verified</TableCell>
                  <TableCell align="center">Language</TableCell>
                  <TableCell align="center">Registered</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredUsers.map((user: TelegramUser) => (
                  <TableRow
                    key={user.telegram_user_id}
                    hover
                    sx={{
                      '&:last-child td, &:last-child th': { border: 0 }
                    }}
                  >
                    <TableCell>
                      <Typography variant="body2" fontFamily="monospace">
                        {user.telegram_user_id}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <EmailIcon fontSize="small" sx={{ mr: 0.5, color: 'text.secondary' }} />
                        <Typography variant="body2">
                          {user.email || '-'}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={user.verified ? 'Yes' : 'No'}
                        color={user.verified ? 'success' : 'default'}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Typography variant="body2">
                        {user.language.toUpperCase()}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Typography variant="body2" color="textSecondary">
                        {new Date(user.created_at).toLocaleDateString()}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Box display="flex" justifyContent="flex-end" gap={1}>
                        <Tooltip title="Approve user">
                          <IconButton
                            size="small"
                            color="success"
                            onClick={() => handleApprove(user.telegram_user_id)}
                            disabled={approveMutation.isLoading}
                          >
                            <ApproveIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Reject user">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleReject(user.telegram_user_id)}
                          >
                            <RejectIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
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

export default PendingApprovalsTable;
