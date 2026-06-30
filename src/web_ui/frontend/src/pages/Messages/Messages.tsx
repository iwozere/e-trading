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
  DialogActions,
  Divider,
  Stack,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Search as SearchIcon,
  Visibility as VisibilityIcon,
  AttachFile as AttachFileIcon,
} from '@mui/icons-material';
import { useMessages } from '../../hooks/notifications/useMessages';
import { MessageItem } from '../../api/notificationsApi';

// ---- Email-style extraction helpers -------------------------------------
// Messages are persisted as a generic content dict; map them onto the email
// fields the notification service uses when actually sending (see EmailChannel
// and the processor's MessageContent mapping).

const emailSubject = (m: MessageItem): string =>
  m.content?.title || m.content?.subject || '(no subject)';

const emailTo = (m: MessageItem): string =>
  m.metadata?.email_receiver || m.recipient_id || '(unknown)';

const asList = (v: any): string[] => {
  if (!v) return [];
  if (Array.isArray(v)) return v.map(String);
  return String(v)
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
};

const emailHtml = (m: MessageItem): string | null => m.content?.html || null;

const emailText = (m: MessageItem): string =>
  m.content?.message || m.content?.text || m.content?.body || '';

const baseName = (p: string): string => String(p).split(/[\\/]/).pop() || String(p);

// Attachments may be stored as {filename: data}, {files: [paths]}, or a list.
const attachmentNames = (m: MessageItem): string[] => {
  const a = m.content?.attachments;
  if (!a) return [];
  if (Array.isArray(a)) {
    return a.map((x) => (typeof x === 'string' ? baseName(x) : x?.name || 'attachment'));
  }
  if (typeof a === 'object') {
    if (Array.isArray(a.files)) return a.files.map((f: any) => baseName(String(f)));
    return Object.keys(a);
  }
  return [];
};

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
  const [showRaw, setShowRaw] = useState(false);

  const openMessage = (m: MessageItem) => {
    setShowRaw(false);
    setSelected(m);
  };

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
                            <IconButton size="small" onClick={() => openMessage(msg)}>
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

      {/* Detail dialog — rendered as a normal email */}
      <Dialog open={Boolean(selected)} onClose={() => setSelected(null)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ pb: 1 }}>
          {selected ? emailSubject(selected) : ''}
        </DialogTitle>
        <DialogContent dividers>
          {selected && (
            <Box>
              {/* Email header */}
              <Box sx={{ mb: 1.5 }}>
                <EmailHeaderRow label="To" value={emailTo(selected)} />
                {asList(selected.metadata?.cc).length > 0 && (
                  <EmailHeaderRow label="Cc" value={asList(selected.metadata?.cc).join(', ')} />
                )}
                {asList(selected.metadata?.bcc).length > 0 && (
                  <EmailHeaderRow label="Bcc" value={asList(selected.metadata?.bcc).join(', ')} />
                )}
                <EmailHeaderRow label="Subject" value={emailSubject(selected)} />
                <EmailHeaderRow
                  label="Date"
                  value={selected.created_at ? new Date(selected.created_at).toLocaleString() : 'N/A'}
                />
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                  <Box sx={{ width: 70, flexShrink: 0 }}>
                    <Typography variant="caption" color="text.secondary">Via</Typography>
                  </Box>
                  <Stack direction="row" spacing={0.5} flexWrap="wrap">
                    {(selected.channels || []).map((c) => (
                      <Chip key={c} label={c} size="small" variant="outlined" />
                    ))}
                    <Chip label={selected.status} size="small" color={statusColor(selected.status)} />
                  </Stack>
                </Box>
              </Box>

              {/* Attachments */}
              {attachmentNames(selected).length > 0 && (
                <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mb: 1.5 }}>
                  {attachmentNames(selected).map((name) => (
                    <Chip
                      key={name}
                      icon={<AttachFileIcon />}
                      label={name}
                      size="small"
                      variant="outlined"
                    />
                  ))}
                </Stack>
              )}

              {selected.last_error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {selected.last_error}
                </Alert>
              )}

              <Divider sx={{ mb: 2 }} />

              {/* Body — HTML rendered in a sandboxed iframe (no scripts), or
                  plain text. Toggle to inspect the raw stored payload. */}
              {showRaw ? (
                <Paper
                  variant="outlined"
                  sx={{ p: 2, overflowX: 'auto', backgroundColor: 'rgba(0,0,0,0.2)' }}
                >
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {JSON.stringify({ content: selected.content, metadata: selected.metadata }, null, 2)}
                  </pre>
                </Paper>
              ) : emailHtml(selected) ? (
                <Box
                  component="iframe"
                  title="message-body"
                  sandbox=""
                  srcDoc={emailHtml(selected) as string}
                  sx={{
                    width: '100%',
                    minHeight: 320,
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 1,
                    backgroundColor: '#fff',
                  }}
                />
              ) : (
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {emailText(selected) || <em>(empty body)</em>}
                </Typography>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowRaw((v) => !v)}>
            {showRaw ? 'Show email' : 'Show raw'}
          </Button>
          <Button onClick={() => setSelected(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Small label/value row used in the email header block.
const EmailHeaderRow: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <Box sx={{ display: 'flex', alignItems: 'baseline', mt: 0.25 }}>
    <Box sx={{ width: 70, flexShrink: 0 }}>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
    </Box>
    <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>{value}</Typography>
  </Box>
);

export default Messages;
