/**
 * User Management Page
 * 
 * Interface for managing Telegram bot users including verification,
 * approval, and user settings management.
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  CheckCircle as ApproveIcon,
  Cancel as RejectIcon,
  Email as EmailIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import TelegramBreadcrumbs from '../../components/Telegram/TelegramBreadcrumbs';
import { 
  useTelegramUsers, 
  useApproveTelegramUser, 
  useRejectTelegramUser,
  useVerifyTelegramUser,
  useResetTelegramUserEmail,
  useDeleteTelegramUser
} from '../../hooks/telegram/useTelegramUsers';

const UserManagement: React.FC = () => {
  const [filter, setFilter] = useState<string>('all');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [userToDelete, setUserToDelete] = useState<string | null>(null);

  // Queries
  const { data: usersResponse, isLoading, isError, error, refetch } = useTelegramUsers({ 
    status: filter === 'all' ? undefined : filter as any
  });

  // Mutations
  const approveMutation = useApproveTelegramUser();
  const rejectMutation = useRejectTelegramUser();
  const verifyMutation = useVerifyTelegramUser();
  const resetEmailMutation = useResetTelegramUserEmail();
  const deleteMutation = useDeleteTelegramUser();

  const users = usersResponse?.data || [];

  const handleApprove = (userId: string) => {
    approveMutation.mutate(userId);
  };

  const handleReject = (userId: string) => {
    rejectMutation.mutate(userId);
  };

  const handleVerify = (userId: string) => {
    verifyMutation.mutate(userId);
  };

  const handleResetEmail = (userId: string) => {
    resetEmailMutation.mutate(userId);
  };

  const handleDeleteClick = (userId: string) => {
    setUserToDelete(userId);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (userToDelete) {
      deleteMutation.mutate(userToDelete);
      setDeleteDialogOpen(false);
      setUserToDelete(null);
    }
  };

  const getStatusChip = (user: any) => {
    if (!user.verified) {
      return <Chip label="Unverified" color="error" size="small" />;
    }
    if (!user.approved) {
      return <Chip label="Pending" color="warning" size="small" />;
    }
    return <Chip label="Approved" color="success" size="small" />;
  };

  if (isError) {
    return (
      <Box sx={{ p: 3 }}>
        <TelegramBreadcrumbs />
        <Typography variant="h4" gutterBottom>
          User Management
        </Typography>
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to load users: {error?.message || 'Unknown error'}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <TelegramBreadcrumbs />
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          User Management
        </Typography>
        <Box display="flex" gap={2} alignItems="center">
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Filter</InputLabel>
            <Select
              value={filter}
              label="Filter"
              onChange={(e) => setFilter(e.target.value)}
            >
              <MenuItem value="all">All Users</MenuItem>
              <MenuItem value="verified">Verified</MenuItem>
              <MenuItem value="approved">Approved</MenuItem>
              <MenuItem value="pending">Pending</MenuItem>
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
                Loading users...
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
                    <TableCell>User ID</TableCell>
                    <TableCell>Email</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Role</TableCell>
                    <TableCell>Language</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {users.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <Typography variant="body2" color="text.secondary">
                          No users found
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    users.map((user: any) => (
                      <TableRow key={user.telegram_user_id}>
                        <TableCell>{user.telegram_user_id}</TableCell>
                        <TableCell>{user.email || 'Not provided'}</TableCell>
                        <TableCell>{getStatusChip(user)}</TableCell>
                        <TableCell>
                          <Chip 
                            label={user.is_admin ? 'Admin' : 'User'} 
                            color={user.is_admin ? 'primary' : 'default'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{user.language}</TableCell>
                        <TableCell align="center">
                          <Box display="flex" gap={1} justifyContent="center">
                            {!user.verified && (
                              <Tooltip title="Verify User">
                                <IconButton
                                  size="small"
                                  color="info"
                                  onClick={() => handleVerify(user.telegram_user_id)}
                                  disabled={verifyMutation.isPending}
                                >
                                  <EmailIcon />
                                </IconButton>
                              </Tooltip>
                            )}
                            {user.verified && !user.approved && (
                              <Tooltip title="Approve User">
                                <IconButton
                                  size="small"
                                  color="success"
                                  onClick={() => handleApprove(user.telegram_user_id)}
                                  disabled={approveMutation.isPending}
                                >
                                  <ApproveIcon />
                                </IconButton>
                              </Tooltip>
                            )}
                            {user.approved && (
                              <Tooltip title="Reject User">
                                <IconButton
                                  size="small"
                                  color="warning"
                                  onClick={() => handleReject(user.telegram_user_id)}
                                  disabled={rejectMutation.isPending}
                                >
                                  <RejectIcon />
                                </IconButton>
                              </Tooltip>
                            )}
                            <Tooltip title="Reset Email">
                              <IconButton
                                size="small"
                                color="secondary"
                                onClick={() => handleResetEmail(user.telegram_user_id)}
                                disabled={resetEmailMutation.isPending}
                              >
                                <RefreshIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Delete User">
                              <IconButton
                                size="small"
                                color="error"
                                onClick={() => handleDeleteClick(user.telegram_user_id)}
                                disabled={deleteMutation.isPending}
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

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this user? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleDeleteConfirm} 
            color="error" 
            variant="contained"
            disabled={deleteMutation.isPending}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default UserManagement;