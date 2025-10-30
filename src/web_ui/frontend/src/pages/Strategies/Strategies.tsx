import React from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText,
} from '@mui/material';
import {
  Add,
  PlayArrow,
  Stop,
  Refresh,
  MoreVert,
  Edit,
  Delete,
  Warning,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';

// API functions
import { 
  getStrategies, 
  startStrategy, 
  stopStrategy, 
  restartStrategy, 
  deleteStrategy 
} from '../../api/tradingApi';
import type { StrategyStatus } from '../../api/tradingApi';

const Strategies: React.FC = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [selectedStrategy, setSelectedStrategy] = React.useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = React.useState(false);
  const [liveConfirmDialogOpen, setLiveConfirmDialogOpen] = React.useState(false);
  const [pendingAction, setPendingAction] = React.useState<{
    action: 'start' | 'restart';
    strategyId: string;
  } | null>(null);

  // Fetch strategies
  const {
    data: strategies = [],
    isLoading,
    refetch,
  } = useQuery<StrategyStatus[]>({
    queryKey: ['strategies'],
    queryFn: getStrategies,
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  // Mutations
  const startMutation = useMutation({
    mutationFn: ({ id, confirmLive }: { id: string; confirmLive: boolean }) =>
      startStrategy(id, confirmLive),
    onSuccess: () => {
      toast.success('Strategy started successfully');
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to start strategy');
    },
  });

  const stopMutation = useMutation({
    mutationFn: stopStrategy,
    onSuccess: () => {
      toast.success('Strategy stopped successfully');
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to stop strategy');
    },
  });

  const restartMutation = useMutation({
    mutationFn: ({ id, confirmLive }: { id: string; confirmLive: boolean }) =>
      restartStrategy(id, confirmLive),
    onSuccess: () => {
      toast.success('Strategy restarted successfully');
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to restart strategy');
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteStrategy,
    onSuccess: () => {
      toast.success('Strategy deleted successfully');
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to delete strategy');
    },
  });

  // Event handlers
  const handleMenuClick = (event: React.MouseEvent<HTMLElement>, strategyId: string) => {
    setAnchorEl(event.currentTarget);
    setSelectedStrategy(strategyId);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedStrategy(null);
  };

  const handleStart = (strategyId: string) => {
    const strategy = strategies.find(s => s.instance_id === strategyId);
    if (strategy?.trading_mode === 'live') {
      setPendingAction({ action: 'start', strategyId });
      setLiveConfirmDialogOpen(true);
    } else {
      startMutation.mutate({ id: strategyId, confirmLive: false });
    }
  };

  const handleStop = (strategyId: string) => {
    stopMutation.mutate(strategyId);
  };

  const handleRestart = (strategyId: string) => {
    const strategy = strategies.find(s => s.instance_id === strategyId);
    if (strategy?.trading_mode === 'live') {
      setPendingAction({ action: 'restart', strategyId });
      setLiveConfirmDialogOpen(true);
    } else {
      restartMutation.mutate({ id: strategyId, confirmLive: false });
    }
  };

  const handleEdit = (strategyId: string) => {
    navigate(`/strategies/${strategyId}/edit`);
    handleMenuClose();
  };

  const handleDelete = (strategyId: string) => {
    setSelectedStrategy(strategyId);
    setDeleteDialogOpen(true);
    handleMenuClose();
  };

  const confirmDelete = () => {
    if (selectedStrategy) {
      deleteMutation.mutate(selectedStrategy);
    }
    setDeleteDialogOpen(false);
    setSelectedStrategy(null);
  };

  const confirmLiveTrading = () => {
    if (pendingAction) {
      if (pendingAction.action === 'start') {
        startMutation.mutate({ id: pendingAction.strategyId, confirmLive: true });
      } else {
        restartMutation.mutate({ id: pendingAction.strategyId, confirmLive: true });
      }
    }
    setLiveConfirmDialogOpen(false);
    setPendingAction(null);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'stopped':
        return 'default';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Trading Strategies
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={() => refetch()}
            disabled={isLoading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => navigate('/strategies/new')}
          >
            New Strategy
          </Button>
        </Box>
      </Box>

      {/* Strategy Grid */}
      <Grid container spacing={3}>
        {strategies.map((strategy) => (
          <Grid item xs={12} sm={6} md={4} key={strategy.instance_id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Typography variant="h6" component="div">
                    {strategy.name}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Chip
                      label={strategy.status}
                      color={getStatusColor(strategy.status) as any}
                      size="small"
                    />
                    <IconButton
                      size="small"
                      onClick={(e) => handleMenuClick(e, strategy.instance_id)}
                    >
                      <MoreVert />
                    </IconButton>
                  </Box>
                </Box>

                <Typography variant="body2" color="textSecondary" gutterBottom>
                  {strategy.symbol} • {strategy.broker_type} • {strategy.trading_mode}
                  {strategy.trading_mode === 'live' && (
                    <Chip
                      label="LIVE"
                      color="warning"
                      size="small"
                      sx={{ ml: 1 }}
                    />
                  )}
                </Typography>

                <Typography variant="body2" gutterBottom>
                  Uptime: {formatUptime(strategy.uptime_seconds)}
                </Typography>

                {strategy.error_count > 0 && (
                  <Typography variant="body2" color="error" gutterBottom>
                    Errors: {strategy.error_count}
                  </Typography>
                )}

                {strategy.last_error && (
                  <Typography variant="body2" color="error" gutterBottom>
                    Last Error: {strategy.last_error}
                  </Typography>
                )}

                <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                  <Button
                    size="small"
                    startIcon={<PlayArrow />}
                    onClick={() => handleStart(strategy.instance_id)}
                    disabled={strategy.status === 'running' || startMutation.isPending}
                    variant="contained"
                    color="success"
                  >
                    Start
                  </Button>
                  <Button
                    size="small"
                    startIcon={<Stop />}
                    onClick={() => handleStop(strategy.instance_id)}
                    disabled={strategy.status !== 'running' || stopMutation.isPending}
                    variant="contained"
                    color="error"
                  >
                    Stop
                  </Button>
                  <Button
                    size="small"
                    startIcon={<Refresh />}
                    onClick={() => handleRestart(strategy.instance_id)}
                    disabled={restartMutation.isPending}
                    variant="outlined"
                  >
                    Restart
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => selectedStrategy && handleEdit(selectedStrategy)}>
          <Edit sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem onClick={() => selectedStrategy && handleDelete(selectedStrategy)}>
          <Delete sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Delete Strategy</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete this strategy? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={confirmDelete} color="error" autoFocus>
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Live Trading Confirmation Dialog */}
      <Dialog
        open={liveConfirmDialogOpen}
        onClose={() => setLiveConfirmDialogOpen(false)}
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Warning color="warning" sx={{ mr: 1 }} />
            Live Trading Confirmation
          </Box>
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            <strong>WARNING:</strong> You are about to {pendingAction?.action} a strategy in LIVE TRADING mode. 
            This will use real money. Please confirm that you want to proceed.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLiveConfirmDialogOpen(false)}>Cancel</Button>
          <Button onClick={confirmLiveTrading} color="warning" autoFocus>
            Confirm Live Trading
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Strategies;