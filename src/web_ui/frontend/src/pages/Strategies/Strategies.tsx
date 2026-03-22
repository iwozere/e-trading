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
  Skeleton,
  alpha,
  useTheme,
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
  RocketLaunch,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
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

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { type: 'spring' as const, stiffness: 100 }
  }
};

const Strategies: React.FC = () => {
  const navigate = useNavigate();
  const theme = useTheme();
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
    refetchInterval: 10000,
  });

  // Mutations
  const startMutation = useMutation({
    mutationFn: ({ id, confirmLive }: { id: string; confirmLive: boolean }) =>
      startStrategy(id, confirmLive),
    onSuccess: () => {
      toast.success('Strategy ignited');
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Ignition failed');
    },
  });

  const stopMutation = useMutation({
    mutationFn: stopStrategy,
    onSuccess: () => {
      toast.success('Strategy suspended');
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
  });

  const restartMutation = useMutation({
    mutationFn: ({ id, confirmLive }: { id: string; confirmLive: boolean }) =>
      restartStrategy(id, confirmLive),
    onSuccess: () => {
      toast.success('Strategy rebooted');
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteStrategy,
    onSuccess: () => {
      toast.success('Fleet data purged');
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
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
    if (selectedStrategy) deleteMutation.mutate(selectedStrategy);
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
      case 'running': return theme.palette.success.main;
      case 'stopped': return theme.palette.text.disabled;
      case 'error': return theme.palette.error.main;
      default: return theme.palette.text.disabled;
    }
  };

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  if (isLoading) {
    return (
      <Box sx={{ p: 4, maxWidth: 1600, mx: 'auto' }}>
        <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between' }}>
           <Skeleton variant="text" width={300} height={60} />
           <Skeleton variant="rectangular" width={150} height={40} />
        </Box>
        <Grid container spacing={4}>
          {[1, 2, 3, 4, 5, 6].map(i => (
            <Grid item xs={12} sm={6} md={4} key={i}>
              <Skeleton variant="rectangular" height={220} sx={{ borderRadius: 3 }} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  return (
    <Box 
        component={motion.div}
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        sx={{ p: 4, maxWidth: 1600, mx: 'auto' }}
    >
      {/* Header */}
      <Box sx={{ mb: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 800, mb: 1, fontFamily: 'Outfit' }}>
            Active Fleet
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Command and control your trading strategies
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <IconButton onClick={() => refetch()} sx={{ border: '1px solid rgba(255,255,255,0.1)' }}>
            <Refresh />
          </IconButton>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => navigate('/strategies/new')}
            sx={{ px: 3, fontWeight: 700 }}
          >
            Deploy New Unit
          </Button>
        </Box>
      </Box>

      {/* Strategy Grid */}
      <Grid container spacing={4}>
        <AnimatePresence>
          {strategies.map((strategy) => (
            <Grid item xs={12} sm={6} md={4} key={strategy.instance_id} component={motion.div} variants={itemVariants} layout>
              <Card sx={{ 
                  height: '100%', 
                  position: 'relative', 
                  overflow: 'hidden',
                  transition: 'transform 0.2s',
                  '&:hover': { transform: 'translateY(-4px)' }
              }}>
                <Box sx={{ 
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    width: 4, 
                    height: '100%', 
                    bgcolor: getStatusColor(strategy.status) 
                }} />
                
                <CardContent sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Box>
                        <Typography variant="h6" sx={{ fontWeight: 700, mb: 0.5 }}>
                        {strategy.name}
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 600, letterSpacing: 1 }}>
                            {strategy.symbol || 'N/A'} • {strategy.strategy_type || 'Custom'}
                        </Typography>
                    </Box>
                    <IconButton size="small" onClick={(e) => handleMenuClick(e, strategy.instance_id)}>
                      <MoreVert />
                    </IconButton>
                  </Box>

                  <Box sx={{ mb: 3, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip
                      label={strategy.status.toUpperCase()}
                      sx={{ 
                          bgcolor: alpha(getStatusColor(strategy.status), 0.1), 
                          color: getStatusColor(strategy.status),
                          fontWeight: 800,
                          fontSize: '0.65rem'
                      }}
                      size="small"
                    />
                    <Chip
                      label={strategy.trading_mode?.toUpperCase() || 'SANDBOX'}
                      color={strategy.trading_mode === 'live' ? 'warning' : 'default'}
                      size="small"
                      sx={{ fontWeight: 800, fontSize: '0.65rem' }}
                    />
                  </Box>

                  <Grid container spacing={2} sx={{ mb: 3 }}>
                      <Grid item xs={6}>
                          <Typography variant="caption" color="text.secondary" display="block">UPTIME</Typography>
                          <Typography variant="body2" sx={{ fontWeight: 700 }}>{formatUptime(strategy.uptime_seconds)}</Typography>
                      </Grid>
                      <Grid item xs={6}>
                          <Typography variant="caption" color="text.secondary" display="block">INSTABILITY</Typography>
                          <Typography variant="body2" sx={{ fontWeight: 700, color: strategy.error_count > 0 ? 'error.main' : 'text.primary' }}>
                              {strategy.error_count} Errors
                          </Typography>
                      </Grid>
                  </Grid>

                  <Box sx={{ display: 'flex', gap: 1.5, mt: 'auto' }}>
                    <Button
                      fullWidth
                      size="small"
                      startIcon={<RocketLaunch />}
                      onClick={() => handleStart(strategy.instance_id)}
                      disabled={strategy.status === 'running' || startMutation.isPending}
                      variant="contained"
                      sx={{ 
                          bgcolor: 'success.main', 
                          '&:hover': { bgcolor: 'success.dark' },
                          fontWeight: 800
                      }}
                    >
                      Launch
                    </Button>
                    <Button
                      fullWidth
                      size="small"
                      startIcon={<Stop />}
                      onClick={() => handleStop(strategy.instance_id)}
                      disabled={strategy.status !== 'running' || stopMutation.isPending}
                      variant="outlined"
                      color="error"
                      sx={{ fontWeight: 800 }}
                    >
                      Suspend
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </AnimatePresence>
      </Grid>

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        PaperProps={{
            sx: { 
                bgcolor: 'rgba(26, 29, 58, 0.9)', 
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255,255,255,0.1)',
                minWidth: 150
            }
        }}
      >
        <MenuItem onClick={() => selectedStrategy && handleEdit(selectedStrategy)}>
          <Edit sx={{ mr: 1.5, fontSize: '1.2rem' }} />
          <Typography variant="body2" sx={{ fontWeight: 600 }}>Modify Unit</Typography>
        </MenuItem>
        <MenuItem onClick={() => selectedStrategy && handleDelete(selectedStrategy)} sx={{ color: 'error.main' }}>
          <Delete sx={{ mr: 1.5, fontSize: '1.2rem' }} />
          <Typography variant="body2" sx={{ fontWeight: 600 }}>Purge Data</Typography>
        </MenuItem>
      </Menu>

      {/* Dialogs with Premium Styling */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)} PaperProps={{ sx: { borderRadius: 3, p: 1 } }}>
        <DialogTitle sx={{ fontWeight: 800 }}>Purge Strategy Unit?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This will permanently decommission the strategy unit and purge all local state. This action is irreversible.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button onClick={() => setDeleteDialogOpen(false)} color="inherit">Abort</Button>
          <Button onClick={confirmDelete} variant="contained" color="error" sx={{ fontWeight: 800 }}>Purge</Button>
        </DialogActions>
      </Dialog>

      <Dialog open={liveConfirmDialogOpen} onClose={() => setLiveConfirmDialogOpen(false)} PaperProps={{ sx: { borderRadius: 3, p: 1, border: '1px solid #ed6c02' } }}>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', fontWeight: 800, color: 'warning.main' }}>
            <Warning sx={{ mr: 1 }} /> LIVE IGNITION WARNING
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            You are about to engage real capital on live markets. Ensure all parameters have been validated.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button onClick={() => setLiveConfirmDialogOpen(false)} color="inherit">Abort Mission</Button>
          <Button onClick={confirmLiveTrading} variant="contained" color="warning" sx={{ fontWeight: 800 }}>Confirm Ignition</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Strategies;