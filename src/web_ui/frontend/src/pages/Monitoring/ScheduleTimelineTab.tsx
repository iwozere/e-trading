import React, { useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  FormControlLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material';
import { useScheduleTimeline } from '../../hooks/system/useSystemHealth';

// ── types (mirror /api/monitoring/schedule-timeline) ─────────────────────────

interface TimelineSlot {
  weekday: number;
  date: string;
  start: string;
  start_minute: number;
  duration_s: number | null;
  duration_source?: 'median' | 'unknown';
  run_id?: number;
  status?: string | null;
}

interface TimelineJob {
  schedule_id: number;
  name: string;
  cron: string;
  frequent: boolean;
  typical_duration_s: number | null;
  slots: TimelineSlot[];
}

interface CollisionSide {
  schedule_id: number;
  name: string;
  start: string;
}

interface Collision {
  weekday: number;
  date: string;
  start_minute: number;
  overlap_s: number;
  a: CollisionSide;
  b: CollisionSide;
}

interface TimelineView {
  window_start: string;
  window_end: string;
  jobs: TimelineJob[];
  collisions: Collision[];
}

interface TimelineResponse {
  timestamp: string;
  tz: string;
  days: number;
  planned: TimelineView;
  actual: TimelineView;
}

type Mode = 'planned' | 'actual';

// ── helpers ───────────────────────────────────────────────────────────────────

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const DAY_MINUTES = 24 * 60;
// Jobs without a known duration (never finished) are drawn this long.
const UNKNOWN_DURATION_MIN = 5;
const LABEL_WIDTH = 260;

function browserTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
  } catch {
    return 'UTC';
  }
}

function todayWeekday(): number {
  return (new Date().getDay() + 6) % 7; // JS: 0=Sun → Mon-based
}

function fmtClock(minuteOfDay: number): string {
  const m = Math.floor(minuteOfDay) % DAY_MINUTES;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
}

function fmtDur(seconds: number | null): string {
  if (seconds === null) return 'unknown';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.round(seconds / 60);
  return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

function slotColor(slot: TimelineSlot, colliding: boolean): string {
  if (slot.status === 'failed') return 'error.main';
  if (slot.status === 'running') return 'warning.main';
  if (colliding) return 'secondary.main';
  if (slot.duration_s === null) return 'grey.500';
  return 'primary.main';
}

// ── component ─────────────────────────────────────────────────────────────────

const ScheduleTimelineTab: React.FC = () => {
  const tz = useMemo(browserTimeZone, []);
  const [mode, setMode] = useState<Mode>('planned');
  const [weekday, setWeekday] = useState<number>(todayWeekday());
  const [days, setDays] = useState<number>(14);
  const [hideFrequent, setHideFrequent] = useState<boolean>(true);
  const [onlyColliding, setOnlyColliding] = useState<boolean>(false);

  const { data, isLoading, error } = useScheduleTimeline(tz, days);
  const view: TimelineView | undefined = (data as TimelineResponse | undefined)?.[mode];

  const dayCollisions = useMemo(
    () => (view?.collisions ?? []).filter((c) => c.weekday === weekday).sort((x, y) => x.start_minute - y.start_minute),
    [view, weekday],
  );

  const collidingIds = useMemo(() => {
    const ids = new Set<number>();
    dayCollisions.forEach((c) => {
      ids.add(c.a.schedule_id);
      ids.add(c.b.schedule_id);
    });
    return ids;
  }, [dayCollisions]);

  const collisionCountByDay = useMemo(() => {
    const counts = Array(7).fill(0) as number[];
    (view?.collisions ?? []).forEach((c) => (counts[c.weekday] += 1));
    return counts;
  }, [view]);

  const rows = useMemo(() => {
    if (!view) return [];
    return view.jobs
      .filter((j) => !(hideFrequent && j.frequent))
      .map((j) => ({ job: j, slots: j.slots.filter((s) => s.weekday === weekday) }))
      .filter((r) => r.slots.length > 0)
      .filter((r) => !onlyColliding || collidingIds.has(r.job.schedule_id))
      .sort((a, b) => a.slots[0].start_minute - b.slots[0].start_minute || a.job.name.localeCompare(b.job.name));
  }, [view, weekday, hideFrequent, onlyColliding, collidingIds]);

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={6}>
        <CircularProgress />
      </Box>
    );
  }
  if (error || !view) {
    return <Alert severity="error">Failed to load schedule timeline.</Alert>;
  }

  return (
    <Stack spacing={2}>
      <Card>
        <CardContent>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems={{ md: 'center' }} flexWrap="wrap" useFlexGap>
            <ToggleButtonGroup size="small" exclusive value={mode} onChange={(_e, v: Mode | null) => v && setMode(v)}>
              <ToggleButton value="planned">Planned (next 7 days)</ToggleButton>
              <ToggleButton value="actual">Actual runs</ToggleButton>
            </ToggleButtonGroup>

            {mode === 'actual' && (
              <Select size="small" value={days} onChange={(e) => setDays(Number(e.target.value))}>
                {[7, 14, 28, 56].map((d) => (
                  <MenuItem key={d} value={d}>
                    Last {d} days
                  </MenuItem>
                ))}
              </Select>
            )}

            <ToggleButtonGroup size="small" exclusive value={weekday} onChange={(_e, v: number | null) => v !== null && setWeekday(v)}>
              {WEEKDAYS.map((label, i) => (
                <ToggleButton key={label} value={i}>
                  {label}
                  {collisionCountByDay[i] > 0 && (
                    <Chip size="small" color="secondary" label={collisionCountByDay[i]} sx={{ ml: 0.75, height: 18 }} />
                  )}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>

            <FormControlLabel
              control={<Switch size="small" checked={hideFrequent} onChange={(e) => setHideFrequent(e.target.checked)} />}
              label="Hide frequent jobs"
            />
            <FormControlLabel
              control={<Switch size="small" checked={onlyColliding} onChange={(e) => setOnlyColliding(e.target.checked)} />}
              label="Only colliding"
            />
          </Stack>
          <Typography variant="caption" color="text.secondary" display="block" mt={1}>
            Times in {tz}. {mode === 'planned'
              ? 'Bars use each job’s median duration of recent runs (grey = never finished, drawn as 5 min).'
              : 'Bars are real runs; on a weekday that occurred several times in the window, runs from all those dates are overlaid.'}{' '}
            Jobs firing more than 6× a day count as frequent and are left out of collision checks.
          </Typography>
        </CardContent>
      </Card>

      <Paper variant="outlined" sx={{ overflowX: 'auto' }}>
        <Box sx={{ minWidth: 900 }}>
          {/* hour axis */}
          <Box display="flex" sx={{ borderBottom: 1, borderColor: 'divider', position: 'sticky', top: 0, bgcolor: 'background.paper' }}>
            <Box sx={{ width: LABEL_WIDTH, flexShrink: 0, px: 1, py: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                {rows.length} jobs
              </Typography>
            </Box>
            <Box sx={{ position: 'relative', flexGrow: 1, height: 24 }}>
              {Array.from({ length: 24 }, (_, h) => (
                <Typography
                  key={h}
                  variant="caption"
                  color="text.secondary"
                  sx={{ position: 'absolute', left: `${(h / 24) * 100}%`, top: 4, pl: 0.25 }}
                >
                  {String(h).padStart(2, '0')}
                </Typography>
              ))}
            </Box>
          </Box>

          {rows.length === 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>
              No jobs on {WEEKDAYS[weekday]}.
            </Typography>
          )}

          {rows.map(({ job, slots }) => {
            const colliding = collidingIds.has(job.schedule_id);
            return (
              <Box key={job.schedule_id} display="flex" sx={{ borderBottom: 1, borderColor: 'divider', '&:hover': { bgcolor: 'action.hover' } }}>
                <Tooltip title={`${job.cron} (UTC)${job.frequent ? ' — frequent' : ''}`} placement="right">
                  <Box sx={{ width: LABEL_WIDTH, flexShrink: 0, px: 1, py: 0.5, overflow: 'hidden' }}>
                    <Typography
                      variant="body2"
                      noWrap
                      fontWeight={colliding ? 600 : 400}
                      color={colliding ? 'secondary.main' : 'text.primary'}
                    >
                      {job.name}
                    </Typography>
                  </Box>
                </Tooltip>
                <Box sx={{ position: 'relative', flexGrow: 1, height: 28 }}>
                  {Array.from({ length: 23 }, (_, h) => (
                    <Box
                      key={h}
                      sx={{ position: 'absolute', left: `${((h + 1) / 24) * 100}%`, top: 0, bottom: 0, borderLeft: 1, borderColor: 'divider' }}
                    />
                  ))}
                  {slots.map((s) => {
                    const durMin = s.duration_s !== null ? s.duration_s / 60 : UNKNOWN_DURATION_MIN;
                    const widthMin = Math.min(Math.max(durMin, 0), DAY_MINUTES - s.start_minute);
                    return (
                      <Tooltip
                        key={`${s.start}-${s.run_id ?? ''}`}
                        title={
                          <>
                            <b>{job.name}</b>
                            <br />
                            {s.date} {fmtClock(s.start_minute)} → {fmtClock(s.start_minute + durMin)}
                            <br />
                            Duration: {fmtDur(s.duration_s)}
                            {s.duration_source === 'median' && ' (median)'}
                            {s.status && (
                              <>
                                <br />
                                Status: {s.status}
                              </>
                            )}
                          </>
                        }
                      >
                        <Box
                          sx={{
                            position: 'absolute',
                            top: 5,
                            height: 18,
                            left: `${(s.start_minute / DAY_MINUTES) * 100}%`,
                            width: `max(3px, ${(widthMin / DAY_MINUTES) * 100}%)`,
                            bgcolor: slotColor(s, colliding),
                            opacity: mode === 'actual' ? 0.6 : 0.85,
                            borderRadius: 0.5,
                          }}
                        />
                      </Tooltip>
                    );
                  })}
                </Box>
              </Box>
            );
          })}
        </Box>
      </Paper>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Collisions on {WEEKDAYS[weekday]} ({dayCollisions.length})
          </Typography>
          {dayCollisions.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No overlapping jobs.
            </Typography>
          ) : (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {mode === 'actual' && <TableCell>Date</TableCell>}
                    <TableCell>Overlap starts</TableCell>
                    <TableCell>Running job</TableCell>
                    <TableCell>Starts during it</TableCell>
                    <TableCell align="right">Overlap</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {dayCollisions.map((c) => (
                    <TableRow key={`${c.a.start}-${c.a.schedule_id}-${c.b.start}-${c.b.schedule_id}`}>
                      {mode === 'actual' && <TableCell>{c.date}</TableCell>}
                      <TableCell>{fmtClock(c.start_minute)}</TableCell>
                      <TableCell>{c.a.name}</TableCell>
                      <TableCell>{c.b.name}</TableCell>
                      <TableCell align="right">{fmtDur(c.overlap_s)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>
    </Stack>
  );
};

export default ScheduleTimelineTab;
