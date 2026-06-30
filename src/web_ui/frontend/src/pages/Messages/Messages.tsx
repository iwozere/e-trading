/**
 * Messages Page
 *
 * Admin view for browsing notification messages sent from the platform
 * (msg_messages table). Supports filtering by recipient, free-text content
 * search, delivery channel, status and a created-at date range. Defaults to
 * the last 30 days.
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
  TablePagination,
  Paper,
  Chip,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Search as SearchIcon,
  Visibility as VisibilityIcon,
} from '@mui/icons-material';
import { useMessages } from '../../hooks/notifications/useMessages';
import { MessageItem } from '../../api/notificationsApi';

// Default the "from" date to 30 days ago (YYYY-MM-DD for the date input).
const isoDateDaysAgo = (days: number): string => {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
};

const STATUS_OPTIONS = ['PENDING', 'PROCESSING', 'DELIVERED', 'FAILED', 'CANCELLED'];
const CHANNEL_OPTIONS = ['telegram', 'email', 'sms', 'slack', 'discord'];

const statusColor = (status: string): 'default' | 'success' | 'error' | 'warning' | 'info' => {
  switch (status?.toUpperCase()) {
    case 'DELIVERED':
      return 'success';
    case 'FAILED':
      return 'error';
    case 'CANCELLED':
      return 'warning';
    case 'PROCESSING':
      return 'info';
    default:
      return 'default';
  }
};

const Messages: React.FC = () => {
  // Draft filters (form inputs) vs applied filters (sent to the query).
  const [draft, setDraft] = useState({
    recipient_id: '',
    search: '',
    status: '',
    channel: '',
    start_date: isoDateDaysAgo(30),
    end_date: '',
  });
  const [applied, setApplied] = useState(draft);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(50);
  const [selected, setSelected] = useState<MessageItem | null>(null);

  const { data, isLoading, isError, error, refetch, isFetching } = useMessages({
    recipient_id: applied.recipient_id || undefined,
    search: applied.search || undefined,
    status: applied.status || undefined,
    channel: applied.channel || undefined,
    // Send the date boundaries as full ISO timestamps (inclusive end of day).
    start_date: applied.start_date ? `${applied.start_date}T00:00:00Z` : undefined,
    end_date: applied.end_date ? `${applied.end_date}T23:59:59Z` : undefined,
    limit: rowsPerPage,
    offset: page * rowsPerPage,
  });

  const messages = data?.items || [];
  const total = data?.total || 0;

  const handleDraftChange = (field: string, value: string) => {
    setDraft((prev) => ({ ...prev, [field]: value }));
  };

  const handleSearch = () => {
    setPage(0);
    setApplied(draft);
  };

  const handleChangePage = (_event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const contentPreview = (content: Record<string, any>): string => {
    if (!content) return '-';
    const text =
      content.message || content.text || content.title || content.body || JSON.stringify(content);
    return text.length > 80 ? `${text.substring(0, 80)}…` : text;
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Messages
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={() => refetch()}
          disabled={isFetching}
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
                label="Recipient"
                value={draft.recipient_id}
                onChange={(e) => handleDraftChange('recipient_id', e.target.value)}
                placeholder="Recipient ID contains…"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                size="small"
                label="Search text"
                value={draft.search}
                onChange={(e) => handleDraftChange('search', e.target.value)}
                placeholder="Content / type / template"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                size="small"
                type="date"
                label="From"
                value={draft.start_date}
                onChange={(e) => handleDraftChange('start_date', e.target.value)}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                size="small"
                type="date"
                label="To"
                value={draft.end_date}
                onChange={(e) => handleDraftChange('end_date', e.target.value)}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Status</InputLabel>
                <Select
                  value={draft.status}
                  label="Status"
                  onChange={(e) => handleDraftChange('status', e.target.value)}
                >
                  <MenuItem value="">All</MenuItem>
                  {STATUS_OPTIONS.map((s) => (
                    <MenuItem key={s} value={s}>
                      {s}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Channel</InputLabel>
                <Select
                  value={draft.channel}
                  label="Channel"
                  onChange={(e) => handleDraftChange('channel', e.target.value)}
                >
                  <MenuItem value="">All</MenuItem>
                  {CHANNEL_OPTIONS.map((c) => (
                    <MenuItem key={c} value={c}>
                      {c}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3} display="flex" alignItems="center">
              <Button
                fullWidth
                variant="contained"
                startIcon={<SearchIcon />}
                onClick={handleSearch}
                disabled={isFetching}
              >
                Search
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {isError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load messages: {(error as Error)?.message || 'Unknown error'}
        </Alert>
      )}

      {isLoading ? (
        <Card>
          <CardContent>
            <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
              <CircularProgress />
              <Typography variant="body1" sx={{ ml: 2 }}>
                Loading messages…
              </Typography>
            </Box>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent>
            <TableContainer component={Paper} elevation={0}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>ID</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Recipient</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Channels</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Content</TableCell>
                    <TableCell align="center">View</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {messages.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={8} align="center">
                        <Typography variant="body2" color="text.secondary" sx={{ py: 3 }}>
                          No messages found for the selected filters.
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    messages.map((msg) => (
                      <TableRow key={msg.id} hover>
                        <TableCell>{msg.id}</TableCell>
                        <TableCell>
                          {msg.created_at ? new Date(msg.created_at).toLocaleString() : 'N/A'}
                        </TableCell>
                        <TableCell>{msg.recipient_id || '-'}</TableCell>
                        <TableCell>{msg.message_type}</TableCell>
                        <TableCell>
                          {(msg.channels || []).map((c) => (
                            <Chip key={c} label={c} size="small" variant="outlined" sx={{ mr: 0.5 }} />
                          ))}
                        </TableCell>
                        <TableCell>
                          <Chip label={msg.status} size="small" color={statusColor(msg.status)} />
                        </TableCell>
                        <TableCell sx={{ maxWidth: 320 }}>
                          <Typography variant="caption" color="text.secondary">
                            {contentPreview(msg.content)}
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Tooltip title="View full message">
                            <IconButton size="small" onClick={() => setSelected(msg)}>
                              <VisibilityIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
            <TablePagination
              component="div"
              count={total}
              page={page}
              onPageChange={handleChangePage}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={handleChangeRowsPerPage}
              rowsPerPageOptions={[25, 50, 100, 200]}
            />
          </CardContent>
        </Card>
      )}

      {/* Detail dialog */}
      <Dialog open={Boolean(selected)} onClose={() => setSelected(null)} maxWidth="md" fullWidth>
        <DialogTitle>Message #{selected?.id}</DialogTitle>
        <DialogContent dividers>
          {selected && (
            <Box>
              <Grid container spacing={1} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Recipient</Typography>
                  <Typography variant="body2">{selected.recipient_id || '-'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Type</Typography>
                  <Typography variant="body2">{selected.message_type}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Status</Typography>
                  <Typography variant="body2">{selected.status}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Priority</Typography>
                  <Typography variant="body2">{selected.priority}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Channels</Typography>
                  <Typography variant="body2">{(selected.channels || []).join(', ') || '-'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Template</Typography>
                  <Typography variant="body2">{selected.template_name || '-'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Created</Typography>
                  <Typography variant="body2">
                    {selected.created_at ? new Date(selected.created_at).toLocaleString() : 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Processed</Typography>
                  <Typography variant="body2">
                    {selected.processed_at ? new Date(selected.processed_at).toLocaleString() : '-'}
                  </Typography>
                </Grid>
              </Grid>

              {selected.last_error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {selected.last_error}
                </Alert>
              )}

              <Typography variant="caption" color="text.secondary">Content</Typography>
              <Paper
                variant="outlined"
                sx={{ p: 2, mt: 0.5, overflowX: 'auto', backgroundColor: 'rgba(0,0,0,0.2)' }}
              >
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {JSON.stringify(selected.content, null, 2)}
                </pre>
              </Paper>
            </Box>
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default Messages;
