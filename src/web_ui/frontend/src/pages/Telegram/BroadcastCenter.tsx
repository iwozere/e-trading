/**
 * Broadcast Center Page
 * 
 * Interface for sending broadcast messages to Telegram bot users
 * and monitoring message delivery status.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  TextField,
  Button,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Divider,
  LinearProgress
} from '@mui/material';
import {
  Send as SendIcon,
  Refresh as RefreshIcon,
  History as HistoryIcon
} from '@mui/icons-material';
import TelegramBreadcrumbs from '../../components/Telegram/TelegramBreadcrumbs';
import { 
  useSendBroadcast, 
  useBroadcastHistory 
} from '../../hooks/telegram/useTelegramBroadcast';

const BroadcastCenter: React.FC = () => {
  const [message, setMessage] = useState('');
  const [page, setPage] = useState(1);

  // Queries and mutations
  const sendBroadcastMutation = useSendBroadcast();
  const { data: historyResponse, isLoading: historyLoading, error: historyError, refetch } = useBroadcastHistory(page, 20);

  const broadcasts = historyResponse?.data || [];

  const handleSendBroadcast = () => {
    if (!message.trim()) {
      return;
    }

    sendBroadcastMutation.mutate({
      message: message.trim()
    });

    // Clear message on successful send
    if (!sendBroadcastMutation.isError) {
      setMessage('');
    }
  };

  const getDeliveryStatusChip = (status: string) => {
    switch (status) {
      case 'completed':
        return <Chip label="Completed" color="success" size="small" />;
      case 'in_progress':
        return <Chip label="In Progress" color="info" size="small" />;
      case 'failed':
        return <Chip label="Failed" color="error" size="small" />;
      case 'pending':
        return <Chip label="Pending" color="warning" size="small" />;
      default:
        return <Chip label="Unknown" color="default" size="small" />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <TelegramBreadcrumbs />
      <Typography variant="h4" gutterBottom>
        Broadcast Center
      </Typography>

      <Grid container spacing={3}>
        {/* Send Broadcast Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Send Broadcast Message
              </Typography>
              <Box sx={{ mb: 2 }}>
                <TextField
                  fullWidth
                  multiline
                  rows={4}
                  label="Message"
                  placeholder="Enter your broadcast message here..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  disabled={sendBroadcastMutation.isPending}
                />
              </Box>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="caption" color="text.secondary">
                  {message.length} characters
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<SendIcon />}
                  onClick={handleSendBroadcast}
                  disabled={!message.trim() || sendBroadcastMutation.isPending}
                >
                  {sendBroadcastMutation.isPending ? 'Sending...' : 'Send Broadcast'}
                </Button>
              </Box>
              
              {sendBroadcastMutation.isPending && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                    Sending broadcast message...
                  </Typography>
                </Box>
              )}

              {sendBroadcastMutation.isError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  Failed to send broadcast: {sendBroadcastMutation.error?.message}
                </Alert>
              )}

              {sendBroadcastMutation.isSuccess && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  Broadcast sent successfully to {sendBroadcastMutation.data?.total_recipients} recipients!
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Broadcast Statistics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Broadcast Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="primary">
                      {broadcasts.length}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Total Broadcasts
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="success.main">
                      {broadcasts.filter((b: any) => b.delivery_status === 'completed').length}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Successful
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Broadcast History */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6">
                  <HistoryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Broadcast History
                </Typography>
                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={() => refetch()}
                  disabled={historyLoading}
                  size="small"
                >
                  Refresh
                </Button>
              </Box>

              {historyError && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  Failed to load broadcast history: {historyError?.message}
                </Alert>
              )}

              {historyLoading ? (
                <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
                  <CircularProgress />
                  <Typography variant="body1" sx={{ ml: 2 }}>
                    Loading broadcast history...
                  </Typography>
                </Box>
              ) : (
                <TableContainer component={Paper} elevation={0}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>ID</TableCell>
                        <TableCell>Message</TableCell>
                        <TableCell>Recipients</TableCell>
                        <TableCell>Delivered</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Sent At</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {broadcasts.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={6} align="center">
                            <Typography variant="body2" color="text.secondary">
                              No broadcasts found
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ) : (
                        broadcasts.map((broadcast: any) => (
                          <TableRow key={broadcast.id}>
                            <TableCell>{broadcast.id}</TableCell>
                            <TableCell>
                              <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                                {broadcast.message.length > 50 
                                  ? `${broadcast.message.substring(0, 50)}...`
                                  : broadcast.message
                                }
                              </Typography>
                            </TableCell>
                            <TableCell>{broadcast.total_recipients || 0}</TableCell>
                            <TableCell>{broadcast.successful_deliveries || 0}</TableCell>
                            <TableCell>
                              {getDeliveryStatusChip(broadcast.delivery_status || 'unknown')}
                            </TableCell>
                            <TableCell>
                              {broadcast.sent_at 
                                ? new Date(broadcast.sent_at).toLocaleString()
                                : 'N/A'
                              }
                            </TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default BroadcastCenter;