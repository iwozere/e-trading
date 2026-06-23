import React, { useState, useRef, useEffect } from 'react';
import cronstrue from 'cronstrue';
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Collapse,
  Divider,
  Drawer,
  Grid,
  IconButton,
  Paper,
  Snackbar,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  Article as ArticleIcon,
  CheckCircle as CheckCircleIcon,
  Close as CloseIcon,
  ContentCopy as ContentCopyIcon,
  Error as ErrorIcon,
  ExpandLess,
  ExpandMore,
  HelpOutline as UnknownIcon,
  Memory as MemoryIcon,
  PlayArrow as PlayArrowIcon,
  Refresh as RefreshIcon,
  Speed as SpeedIcon,
  Timeline as TimelineIcon,
  Timer as TimerIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { useSystemMetrics, useSystemStatus, useServicesStatus, usePipelinesStatus, useTriggerPipeline, useRunLogs } from '../../hooks/system/useSystemHealth';

// ── shared types ─────────────────────────────────────────────────────────────

interface RecentRun {
  id: number;
  status: string | null;
  started_at: string | null;
  finished_at: string | null;
  duration_s: number | null;
  error: string | null;
}

interface Pipeline {
  id: number;
  name: string;
  job_type: string;
  target: string;
  enabled: boolean;
  cron: string;
  next_run_at: string | null;
  last_status: string;
  last_run_at: string | null;
  last_duration_s: number | null;
  success_rate_10: number | null;
  recent_runs: RecentRun[];
}

interface ServiceEntry {
  name: string;
  display_name: string;
  status: 'active' | 'inactive' | 'unknown';
  has_errors: boolean;
}

interface ChannelEntry {
  status: string;
}

// ── helpers ───────────────────────────────────────────────────────────────────

function statusColor(status: string): 'success' | 'error' | 'warning' | 'default' {
  if (['active', 'healthy', 'completed'].includes(status)) return 'success';
  if (['failed', 'inactive'].includes(status)) return 'error';
  if (['running', 'degraded'].includes(status)) return 'warning';
  return 'default';
}

function statusIcon(status: string) {
  if (['active', 'healthy', 'completed'].includes(status))
    return <CheckCircleIcon fontSize="small" color="success" />;
  if (['failed', 'inactive'].includes(status))
    return <ErrorIcon fontSize="small" color="error" />;
  if (['running', 'degraded'].includes(status))
    return <WarningIcon fontSize="small" color="warning" />;
  return <UnknownIcon fontSize="small" color="disabled" />;
}

function cronToHuman(cron: string): string {
  try {
    return cronstrue.toString(cron, { use24HourTimeFormat: true });
  } catch {
    return cron;
  }
}

function fmtDuration(seconds: number | null): string {
  if (seconds === null) return '—';
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function fmtDate(iso: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  const dd = String(d.getDate()).padStart(2, '0');
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const yyyy = d.getFullYear();
  const hh = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${dd}.${mm}.${yyyy} ${hh}:${min}:${ss}`;
}

function fmtDateShort(iso: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  const dd = String(d.getDate()).padStart(2, '0');
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const yyyy = d.getFullYear();
  const hh = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${dd}.${mm}.${yyyy} ${hh}:${min}`;
}

// ── TabPanel ──────────────────────────────────────────────────────────────────

interface TabPanelProps {
  children: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div role="tabpanel" hidden={value !== index}>
    {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
  </div>
);

// ── MetricBox (unchanged from original) ───────────────────────────────────────

interface MetricBoxProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  subtitle?: string;
}

const MetricBox: React.FC<MetricBoxProps> = ({ title, value, icon, subtitle }) => (
  <Paper elevation={0} variant="outlined" sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, color: 'text.secondary' }}>
      {icon}
      <Typography variant="body2" sx={{ ml: 1, fontWeight: 'medium' }}>
        {title}
      </Typography>
    </Box>
    <Typography variant="h4" component="div" sx={{ mb: 1 }}>
      {value}
    </Typography>
    {subtitle && (
      <Typography variant="caption" color="text.secondary">
        {subtitle}
      </Typography>
    )}
  </Paper>
);

// ── SystemTab (original content) ──────────────────────────────────────────────

const SystemTab: React.FC = () => {
  const { data: statusData, error: statusError } = useSystemStatus();
  const { data: metricsData, error: metricsError } = useSystemMetrics();

  const formatUptime = (seconds?: number) => {
    if (!seconds) return 'N/A';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  };

  const cpuUsage = metricsData?.cpu?.usage_percent || statusData?.system_metrics?.cpu_percent || 0;
  const memUsage = metricsData?.memory?.usage_percent || statusData?.system_metrics?.memory_percent || 0;

  return (
    <Box>
      {(statusError || metricsError) && (
        <Alert severity="error" sx={{ mb: 3 }}>
          Failed to fetch system metrics. The backend may be offline.
        </Alert>
      )}

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox
            title="Service Version"
            value={statusData?.version || 'N/A'}
            icon={<TimelineIcon />}
            subtitle={statusData?.service_name || 'Alkotrader'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox
            title="Uptime"
            value={formatUptime(statusData?.uptime_seconds)}
            icon={<TimerIcon />}
            subtitle="Time since last restart"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox
            title="CPU Usage"
            value={`${cpuUsage.toFixed(1)}%`}
            icon={<SpeedIcon />}
            subtitle="Current system load"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox
            title="Memory Usage"
            value={`${memUsage.toFixed(1)}%`}
            icon={<MemoryIcon />}
            subtitle="Current RAM usage"
          />
        </Grid>
      </Grid>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Trading Strategies
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Box sx={{ display: 'flex', gap: 4 }}>
            <Box>
              <Typography variant="body2" color="text.secondary">Active Strategies</Typography>
              <Typography variant="h5">{statusData?.active_strategies || 0}</Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">Total Configured</Typography>
              <Typography variant="h5">{statusData?.total_strategies || 0}</Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

// ── ServicesTab ───────────────────────────────────────────────────────────────

const ServicesTab: React.FC = () => {
  const { data, isLoading, error } = useServicesStatus();

  if (isLoading) return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}><CircularProgress /></Box>;
  if (error) return <Alert severity="error">Failed to fetch service status. The backend may be offline.</Alert>;

  const services: ServiceEntry[] = data?.services || [];
  const channels: Record<string, ChannelEntry> = data?.channels || {};

  return (
    <Box>
      <Typography variant="h6" gutterBottom>System Services</Typography>
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {services.map((svc) => (
          <Grid item xs={12} sm={6} md={3} key={svc.name}>
            <Paper
              variant="outlined"
              sx={{
                p: 2,
                borderColor: svc.status === 'active' && !svc.has_errors
                  ? 'success.dark'
                  : svc.status === 'inactive'
                  ? 'error.dark'
                  : 'divider',
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                {statusIcon(svc.has_errors ? 'warning' : svc.status)}
                <Chip
                  label={svc.status.toUpperCase()}
                  color={statusColor(svc.has_errors ? 'warning' : svc.status)}
                  size="small"
                  sx={{ fontWeight: 700, fontSize: '0.65rem', height: 20 }}
                />
              </Box>
              <Typography variant="body1" fontWeight={600} sx={{ mt: 1 }}>
                {svc.display_name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {svc.name}
              </Typography>
              {svc.has_errors && (
                <Typography variant="caption" color="warning.main" display="block" sx={{ mt: 0.5 }}>
                  Errors detected in logs
                </Typography>
              )}
            </Paper>
          </Grid>
        ))}
      </Grid>

      <Typography variant="h6" gutterBottom>Notification Channels</Typography>
      <Grid container spacing={2}>
        {Object.entries(channels).map(([name, ch]) => (
          <Grid item xs={12} sm={6} md={4} key={name}>
            <Paper
              variant="outlined"
              sx={{
                p: 2,
                borderColor: ch.status === 'healthy' ? 'success.dark' : ch.status === 'degraded' ? 'warning.dark' : 'divider',
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                {statusIcon(ch.status)}
                <Chip
                  label={ch.status.toUpperCase()}
                  color={statusColor(ch.status)}
                  size="small"
                  sx={{ fontWeight: 700, fontSize: '0.65rem', height: 20 }}
                />
              </Box>
              <Typography variant="body1" fontWeight={600} sx={{ mt: 1, textTransform: 'capitalize' }}>
                {name}
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

// ── RunHistoryStrip ───────────────────────────────────────────────────────────

const RunHistoryStrip: React.FC<{ runs: RecentRun[] }> = ({ runs }) => {
  const dots = [...runs].reverse();
  return (
    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
      {dots.map((r, i) => {
        const color =
          r.status === 'completed' ? 'success.main'
          : r.status === 'failed' ? 'error.main'
          : r.status === 'running' ? 'warning.main'
          : 'text.disabled';
        const label = `${r.status ?? 'unknown'}${r.started_at ? ' · ' + fmtDateShort(r.started_at) : ''}${r.duration_s !== null ? ' · ' + fmtDuration(r.duration_s) : ''}${r.error ? '\n' + r.error : ''}`;
        return (
          <Tooltip key={i} title={<span style={{ whiteSpace: 'pre-line' }}>{label}</span>} arrow>
            <Box
              sx={{
                width: 14,
                height: 14,
                borderRadius: '3px',
                backgroundColor: color,
                cursor: 'default',
                opacity: 0.85,
                '&:hover': { opacity: 1 },
              }}
            />
          </Tooltip>
        );
      })}
      {runs.length === 0 && (
        <Typography variant="caption" color="text.disabled">No runs yet</Typography>
      )}
    </Box>
  );
};

// ── LogViewerDrawer ───────────────────────────────────────────────────────────

interface LogViewerDrawerProps {
  open: boolean;
  runId: number | null;
  pipelineName: string;
  onClose: () => void;
}

const LogViewerDrawer: React.FC<LogViewerDrawerProps> = ({ open, runId, pipelineName, onClose }) => {
  const { data, isLoading, error, refetch } = useRunLogs(open ? runId : null);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (data?.log_content && logEndRef.current) {
      logEndRef.current.scrollIntoView();
    }
  }, [data]);

  const handleCopy = () => {
    if (data?.log_content) {
      navigator.clipboard.writeText(data.log_content).catch(() => {});
    }
  };

  return (
    <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: { xs: '100%', sm: 680 } } }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <Box>
            <Typography variant="h6">Pipeline Logs</Typography>
            <Typography variant="caption" color="text.secondary">{pipelineName}</Typography>
            {data?.started_at && (
              <Typography variant="caption" color="text.secondary" display="block">{fmtDate(data.started_at)}</Typography>
            )}
          </Box>
          <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5 }}>
            <Tooltip title="Refresh">
              <IconButton onClick={() => refetch()} size="small"><RefreshIcon fontSize="small" /></IconButton>
            </Tooltip>
            <Tooltip title="Copy to clipboard">
              <span>
                <IconButton onClick={handleCopy} size="small" disabled={!data?.log_content}>
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
            <IconButton onClick={onClose} size="small"><CloseIcon fontSize="small" /></IconButton>
          </Box>
        </Box>

        {data?.source && data.source !== 'none' && (
          <Box sx={{ px: 2, py: 0.5, bgcolor: 'action.hover', borderBottom: 1, borderColor: 'divider' }}>
            <Typography variant="caption" color="text.secondary">
              {data.source === 'file' ? `file: ${data.log_path}` : 'source: database result'}
            </Typography>
          </Box>
        )}

        <Box sx={{ flex: 1, overflow: 'auto', p: 1.5 }}>
          {isLoading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}><CircularProgress /></Box>
          )}
          {error && <Alert severity="error">Failed to load logs.</Alert>}
          {!isLoading && data?.source === 'none' && (
            <Alert severity="info">No log file or result data available for this run.</Alert>
          )}
          {data?.log_content && (
            <Box
              component="pre"
              sx={{
                fontFamily: 'Consolas, "Courier New", monospace',
                fontSize: '0.72rem',
                lineHeight: 1.5,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
                m: 0,
                p: 1.5,
                bgcolor: 'grey.900',
                color: 'grey.100',
                borderRadius: 1,
                border: 1,
                borderColor: 'divider',
              }}
            >
              {data.log_content}
            </Box>
          )}
          <div ref={logEndRef} />
        </Box>
      </Box>
    </Drawer>
  );
};

// ── PipelineRow ───────────────────────────────────────────────────────────────

interface PipelineRowProps {
  pipeline: Pipeline;
  onTrigger: (scheduleId: number, name: string) => void;
  onViewLogs: (runId: number, pipelineName: string) => void;
  isTriggeringThis: boolean;
}

const PipelineRow: React.FC<PipelineRowProps> = ({ pipeline: p, onTrigger, onViewLogs, isTriggeringThis }) => {
  const [open, setOpen] = useState(false);
  const isRunning = p.last_status === 'running';
  const runButtonDisabled = !p.enabled || isRunning || isTriggeringThis;

  return (
    <>
      <TableRow sx={{ '& > *': { borderBottom: 'unset' }, opacity: p.enabled ? 1 : 0.5 }}>
        <TableCell padding="checkbox">
          <IconButton size="small" onClick={() => setOpen(!open)} disabled={p.recent_runs.length === 0}>
            {open ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </IconButton>
        </TableCell>
        <TableCell>
          <Typography variant="body2" fontWeight={600}>{p.name}</Typography>
          <Typography variant="caption" color="text.secondary">{p.job_type}</Typography>
        </TableCell>
        <TableCell>
          <Tooltip title={cronToHuman(p.cron)} arrow placement="top">
            <Typography variant="body2" sx={{ fontFamily: 'monospace', cursor: 'default', whiteSpace: 'nowrap' }}>
              {p.cron}
            </Typography>
          </Tooltip>
        </TableCell>
        <TableCell>
          <Chip
            label={p.last_status.toUpperCase()}
            color={statusColor(p.last_status)}
            size="small"
            sx={{ fontWeight: 700, fontSize: '0.65rem', height: 20 }}
          />
        </TableCell>
        <TableCell>
          <Typography variant="body2">{fmtDate(p.last_run_at)}</Typography>
        </TableCell>
        <TableCell>
          <Typography variant="body2">{fmtDuration(p.last_duration_s)}</Typography>
        </TableCell>
        <TableCell>
          {p.success_rate_10 !== null ? (
            <Typography
              variant="body2"
              color={p.success_rate_10 >= 0.8 ? 'success.main' : p.success_rate_10 >= 0.5 ? 'warning.main' : 'error.main'}
              fontWeight={600}
            >
              {Math.round(p.success_rate_10 * 100)}%
            </Typography>
          ) : (
            <Typography variant="body2" color="text.disabled">—</Typography>
          )}
        </TableCell>
        <TableCell>
          <Typography variant="body2">{fmtDate(p.next_run_at)}</Typography>
        </TableCell>
        <TableCell>
          <RunHistoryStrip runs={p.recent_runs} />
        </TableCell>
        <TableCell padding="none" sx={{ pr: 1 }}>
          <Box sx={{ display: 'flex' }}>
            <Tooltip title={!p.enabled ? 'Schedule disabled' : isRunning || isTriggeringThis ? 'Pipeline is running' : 'Run now'}>
              <span>
                <IconButton
                  size="small"
                  color="primary"
                  disabled={runButtonDisabled}
                  onClick={() => onTrigger(p.id, p.name)}
                >
                  {isRunning || isTriggeringThis
                    ? <CircularProgress size={16} color="inherit" />
                    : <PlayArrowIcon fontSize="small" />}
                </IconButton>
              </span>
            </Tooltip>
            <Tooltip title={p.recent_runs.length === 0 ? 'No runs yet' : 'View logs'}>
              <span>
                <IconButton
                  size="small"
                  disabled={p.recent_runs.length === 0}
                  onClick={() => p.recent_runs.length > 0 && onViewLogs(p.recent_runs[0].id, p.name)}
                >
                  <ArticleIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        </TableCell>
      </TableRow>
      <TableRow>
        <TableCell colSpan={10} sx={{ py: 0 }}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ py: 2, px: 4 }}>
              <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                Cron: <strong>{p.cron}</strong> · Target: <strong>{p.target}</strong>
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Run ID</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Started</TableCell>
                    <TableCell>Finished</TableCell>
                    <TableCell>Duration</TableCell>
                    <TableCell>Error</TableCell>
                    <TableCell padding="none" />
                  </TableRow>
                </TableHead>
                <TableBody>
                  {p.recent_runs.map((r) => (
                    <TableRow key={r.id}>
                      <TableCell>
                        <Typography variant="caption" color="text.secondary">#{r.id}</Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={(r.status ?? 'unknown').toUpperCase()}
                          color={statusColor(r.status ?? 'unknown')}
                          size="small"
                          sx={{ fontWeight: 700, fontSize: '0.6rem', height: 18 }}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">{fmtDate(r.started_at)}</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">{fmtDate(r.finished_at)}</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">{fmtDuration(r.duration_s)}</Typography>
                      </TableCell>
                      <TableCell sx={{ maxWidth: 300 }}>
                        {r.error ? (
                          <Tooltip title={r.error} arrow>
                            <Typography
                              variant="caption"
                              color="error.main"
                              sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'block', maxWidth: 260 }}
                            >
                              {r.error}
                            </Typography>
                          </Tooltip>
                        ) : (
                          <Typography variant="caption" color="text.disabled">—</Typography>
                        )}
                      </TableCell>
                      <TableCell padding="none">
                        <Tooltip title="View logs">
                          <IconButton size="small" onClick={() => onViewLogs(r.id, p.name)}>
                            <ArticleIcon sx={{ fontSize: 14 }} />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
};

// ── PipelinesTab ──────────────────────────────────────────────────────────────

const PipelinesTab: React.FC = () => {
  const { data, isLoading, error } = usePipelinesStatus();
  const triggerMutation = useTriggerPipeline();
  const [triggeringId, setTriggeringId] = useState<number | null>(null);
  const [logDrawer, setLogDrawer] = useState<{ open: boolean; runId: number | null; pipelineName: string }>({
    open: false, runId: null, pipelineName: '',
  });
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false, message: '', severity: 'success',
  });

  const handleTrigger = async (scheduleId: number, name: string) => {
    setTriggeringId(scheduleId);
    try {
      await triggerMutation.mutateAsync(scheduleId);
      setSnackbar({ open: true, message: `"${name}" triggered successfully`, severity: 'success' });
    } catch (err: any) {
      const detail = err?.response?.data?.detail ?? `Failed to trigger "${name}"`;
      setSnackbar({ open: true, message: detail, severity: 'error' });
    } finally {
      setTriggeringId(null);
    }
  };

  const handleViewLogs = (runId: number, pipelineName: string) => {
    setLogDrawer({ open: true, runId, pipelineName });
  };

  if (isLoading) return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}><CircularProgress /></Box>;
  if (error) return <Alert severity="error">Failed to fetch pipeline status. The backend may be offline.</Alert>;

  const pipelines: Pipeline[] = data?.pipelines || [];

  if (pipelines.length === 0) {
    return (
      <Alert severity="info">No pipelines configured. Add schedules via the Telegram Schedule Management page.</Alert>
    );
  }

  return (
    <>
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox" />
              <TableCell>Pipeline</TableCell>
              <TableCell>Schedule</TableCell>
              <TableCell>Last Status</TableCell>
              <TableCell>Last Run</TableCell>
              <TableCell>Duration</TableCell>
              <TableCell>Success (10 runs)</TableCell>
              <TableCell>Next Run</TableCell>
              <TableCell>History</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {pipelines.map((p) => (
              <PipelineRow
                key={p.id}
                pipeline={p}
                onTrigger={handleTrigger}
                onViewLogs={handleViewLogs}
                isTriggeringThis={triggeringId === p.id}
              />
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <LogViewerDrawer
        open={logDrawer.open}
        runId={logDrawer.runId}
        pipelineName={logDrawer.pipelineName}
        onClose={() => setLogDrawer((prev) => ({ ...prev, open: false }))}
      />

      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          severity={snackbar.severity}
          onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
};

// ── Main Monitoring page ──────────────────────────────────────────────────────

const Monitoring: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Monitoring
      </Typography>

      <Tabs
        value={activeTab}
        onChange={(_e, v) => setActiveTab(v)}
        sx={{ borderBottom: 1, borderColor: 'divider', mb: 0 }}
      >
        <Tab label="System" />
        <Tab label="Services" />
        <Tab label="Pipelines" />
      </Tabs>

      <TabPanel value={activeTab} index={0}>
        <SystemTab />
      </TabPanel>
      <TabPanel value={activeTab} index={1}>
        <ServicesTab />
      </TabPanel>
      <TabPanel value={activeTab} index={2}>
        <PipelinesTab />
      </TabPanel>
    </Box>
  );
};

export default Monitoring;
