import React, { useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Paper,
  Stack,
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
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from 'recharts';
import { useRecentSignals, useSignalBookNav, useSignalScorecard } from '../../hooks/system/useSystemHealth';

// ── types (mirror /api/signals/*) ────────────────────────────────────────────

interface HorizonStats {
  horizon: number;
  signals: number;
  cohorts: number;
  mean_raw: number;
  mean_excess: number;
  median_excess: number;
  hit_rate: number;
  cohort_mean_excess: number;
  nw_t: number | null;
  verdict: 'edge' | 'negative' | 'no_edge' | 'insufficient';
}

interface Stream {
  stream: string;
  source: string;
  signals: number;
  first_date: string;
  last_date: string;
  cost_bps: number;
  benchmark: string;
  horizons: HorizonStats[];
}

interface BookSummary {
  total_return?: number;
  benchmark_return?: number;
  excess_return?: number;
  max_drawdown?: number;
  avg_invested_pct?: number;
  positions?: number;
  start_date?: string;
}

interface Scorecard {
  benchmark: string;
  streams: Stream[];
  books: Record<string, BookSummary>;
  status_counts: Record<string, Record<string, number>>;
}

interface NavRow {
  date: string;
  nav: number;
  benchmark_nav: number;
  positions: number;
}

interface SignalRow {
  id: number;
  source: string;
  signal_date: string;
  ticker: string;
  side: string;
  rank: number | null;
  score: number | null;
  status: string;
  entry_date: string | null;
  entry_price: number | null;
  data_flag: string | null;
  [key: string]: unknown;
}

const HORIZONS = [1, 5, 10, 20];
const BENCHMARKS = ['vt', 'spy', 'iwm'];

const VERDICT_CHIP: Record<HorizonStats['verdict'], { label: string; color: 'success' | 'error' | 'default' | 'warning' }> = {
  edge: { label: 'Edge', color: 'success' },
  negative: { label: 'Negative', color: 'error' },
  no_edge: { label: 'No edge', color: 'default' },
  insufficient: { label: 'Too few dates', color: 'warning' },
};

const pct = (value: number | null | undefined, digits = 1): string =>
  value === null || value === undefined || Number.isNaN(value) ? '—' : `${(value * 100).toFixed(digits)}%`;

const signedColor = (value: number | null | undefined): string | undefined =>
  value === null || value === undefined ? undefined : value > 0 ? 'success.main' : value < 0 ? 'error.main' : undefined;

// ── sub-components ───────────────────────────────────────────────────────────

const BookChart: React.FC<{ source: string; benchmark: string }> = ({ source, benchmark }) => {
  const { data, isLoading, error } = useSignalBookNav(source);
  const rows: NavRow[] = data?.nav ?? [];

  if (isLoading) return <CircularProgress size={24} />;
  if (error || rows.length === 0) return <Alert severity="info">No paper book yet for {source}.</Alert>;

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" minTickGap={40} />
        <YAxis domain={['auto', 'auto']} tickFormatter={(v: number) => `${Math.round(v / 1000)}k`} />
        <ChartTooltip formatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })} />
        <Legend />
        <Line type="monotone" dataKey="nav" name={`${source} book`} dot={false} strokeWidth={2} stroke="#1976d2" />
        <Line type="monotone" dataKey="benchmark_nav" name={`100% ${benchmark}`} dot={false} stroke="#9e9e9e" />
      </LineChart>
    </ResponsiveContainer>
  );
};

const RecentSignals: React.FC<{ source: string; horizon: number; benchmark: string }> = ({ source, horizon, benchmark }) => {
  const { data, isLoading } = useRecentSignals(source, 100);
  const rows: SignalRow[] = data?.signals ?? [];
  const retKey = `ret_${horizon}d`;
  const benchKey = `${benchmark}_${horizon}d`;

  if (isLoading) return <CircularProgress size={24} />;
  return (
    <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 420 }}>
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Date</TableCell>
            <TableCell>Ticker</TableCell>
            <TableCell>Side</TableCell>
            <TableCell align="right">Rank</TableCell>
            <TableCell>Entry</TableCell>
            <TableCell align="right">Entry px</TableCell>
            <TableCell align="right">{horizon}d return</TableCell>
            <TableCell align="right">vs {benchmark.toUpperCase()}</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((r) => {
            const ret = r[retKey] as number | null;
            const bench = r[benchKey] as number | null;
            const excess = ret !== null && bench !== null && ret !== undefined && bench !== undefined
              ? (r.side === 'short' ? -1 : 1) * (ret - bench) : null;
            return (
              <TableRow key={r.id} hover>
                <TableCell>{r.signal_date}</TableCell>
                <TableCell>{r.ticker}</TableCell>
                <TableCell>{r.side}</TableCell>
                <TableCell align="right">{r.rank ?? '—'}</TableCell>
                <TableCell>{r.entry_date ?? '—'}</TableCell>
                <TableCell align="right">{r.entry_price?.toFixed(2) ?? '—'}</TableCell>
                <TableCell align="right" sx={{ color: signedColor(ret) }}>{pct(ret)}</TableCell>
                <TableCell align="right" sx={{ color: signedColor(excess) }}>{pct(excess)}</TableCell>
                <TableCell>
                  {r.status}
                  {r.data_flag && (
                    <Tooltip title="Suspected unadjusted split — excluded from scorecard and book">
                      <Chip size="small" color="warning" label={r.data_flag} sx={{ ml: 1 }} />
                    </Tooltip>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

// ── main tab ─────────────────────────────────────────────────────────────────

const SignalsTab: React.FC = () => {
  const [benchmark, setBenchmark] = useState('vt');
  const [horizon, setHorizon] = useState(20);
  const [selected, setSelected] = useState<string | null>(null);
  const { data, isLoading, error } = useSignalScorecard(benchmark);
  const scorecard = data as Scorecard | undefined;

  const rows = useMemo(
    () =>
      (scorecard?.streams ?? []).map((s) => ({
        stream: s,
        stats: s.horizons.find((h) => h.horizon === horizon),
        book: scorecard?.books[s.stream],
      })),
    [scorecard, horizon],
  );
  const selectedStream = rows.find((r) => r.stream.stream === selected)?.stream;

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">Failed to load the signal scorecard.</Alert>;
  if (!rows.length) return <Alert severity="info">No signals in the ledger yet — run the Signal Ledger Nightly job.</Alert>;

  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        <ToggleButtonGroup size="small" exclusive value={benchmark} onChange={(_e, v) => v && setBenchmark(v)}>
          {BENCHMARKS.map((b) => (
            <ToggleButton key={b} value={b}>vs {b.toUpperCase()}</ToggleButton>
          ))}
        </ToggleButtonGroup>
        <ToggleButtonGroup size="small" exclusive value={horizon} onChange={(_e, v) => v && setHorizon(v)}>
          {HORIZONS.map((h) => (
            <ToggleButton key={h} value={h}>{h}d</ToggleButton>
          ))}
        </ToggleButtonGroup>
        <Typography variant="body2" color="text.secondary">
          Entry at the next session's open, exit at the close {horizon} sessions later; excess return is after the stream's
          round-trip cost. t = Newey-West t-stat of daily cohort means (|t| ≥ 2 ≈ significant).
        </Typography>
      </Stack>

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Stream</TableCell>
              <TableCell align="right">Signals</TableCell>
              <TableCell align="right">Dates</TableCell>
              <TableCell>Since</TableCell>
              <TableCell align="right">Mean excess</TableCell>
              <TableCell align="right">Median excess</TableCell>
              <TableCell align="right">Hit rate</TableCell>
              <TableCell align="right">t</TableCell>
              <TableCell>Verdict</TableCell>
              <TableCell align="right">Book return</TableCell>
              <TableCell align="right">Book vs VT</TableCell>
              <TableCell align="right">Max DD</TableCell>
              <TableCell align="right">Invested</TableCell>
              <TableCell align="right">Open</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map(({ stream, stats, book }) => (
              <TableRow
                key={stream.stream}
                hover
                selected={stream.stream === selected}
                onClick={() => setSelected(stream.stream === selected ? null : stream.stream)}
                sx={{ cursor: 'pointer' }}
              >
                <TableCell>
                  <Tooltip title={`cost ${stream.cost_bps} bps round trip`}>
                    <span>{stream.stream}</span>
                  </Tooltip>
                </TableCell>
                <TableCell align="right">{stats?.signals ?? 0} / {stream.signals}</TableCell>
                <TableCell align="right">{stats?.cohorts ?? 0}</TableCell>
                <TableCell>{stream.first_date}</TableCell>
                <TableCell align="right" sx={{ color: signedColor(stats?.mean_excess) }}>{pct(stats?.mean_excess, 2)}</TableCell>
                <TableCell align="right" sx={{ color: signedColor(stats?.median_excess) }}>{pct(stats?.median_excess, 2)}</TableCell>
                <TableCell align="right">{pct(stats?.hit_rate, 0)}</TableCell>
                <TableCell align="right">{stats?.nw_t?.toFixed(2) ?? '—'}</TableCell>
                <TableCell>
                  {stats ? <Chip size="small" label={VERDICT_CHIP[stats.verdict].label} color={VERDICT_CHIP[stats.verdict].color} /> : '—'}
                </TableCell>
                <TableCell align="right" sx={{ color: signedColor(book?.total_return) }}>{pct(book?.total_return)}</TableCell>
                <TableCell align="right" sx={{ color: signedColor(book?.excess_return) }}>{pct(book?.excess_return)}</TableCell>
                <TableCell align="right">{pct(book?.max_drawdown)}</TableCell>
                <TableCell align="right">{pct(book?.avg_invested_pct, 0)}</TableCell>
                <TableCell align="right">{book?.positions ?? '—'}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <Typography variant="caption" color="text.secondary">
        Book: $200k start in VT, $2,000 per signal at the next open (best rank first), max 40 open positions, sold after the
        stream's holding period, cash back into VT. Sub-streams (e.g. p17_tier_ab) have no book of their own. Click a row for details.
      </Typography>

      {selectedStream && (
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>{selectedStream.stream}</Typography>
            <Box sx={{ mb: 2 }}>
              <BookChart source={selectedStream.source} benchmark="VT" />
            </Box>
            <Typography variant="subtitle2" gutterBottom>Latest signals ({selectedStream.source})</Typography>
            <RecentSignals source={selectedStream.source} horizon={horizon} benchmark={benchmark} />
          </CardContent>
        </Card>
      )}
    </Stack>
  );
};

export default SignalsTab;
