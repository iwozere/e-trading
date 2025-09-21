import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Chip,
  Divider,
} from '@mui/material';
import {
  ExpandMore,
  Save,
  Cancel,
  Warning,
} from '@mui/icons-material';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';

// API functions
import { createStrategy, updateStrategy, getStrategy, getStrategyTemplates } from '../../api/tradingApi';

// Validation schema
const strategySchema = z.object({
  id: z.string().min(1, 'Strategy ID is required'),
  name: z.string().min(1, 'Strategy name is required'),
  enabled: z.boolean(),
  symbol: z.string().min(1, 'Symbol is required'),
  broker: z.object({
    type: z.enum(['binance', 'ibkr', 'mock']),
    trading_mode: z.enum(['paper', 'live']),
    name: z.string().min(1, 'Broker name is required'),
    cash: z.number().min(0, 'Cash must be positive'),
    paper_trading_config: z.object({
      mode: z.enum(['basic', 'realistic', 'advanced']),
      initial_balance: z.number().min(0),
      commission_rate: z.number().min(0).max(1),
      slippage_model: z.enum(['fixed', 'linear', 'sqrt']),
      base_slippage: z.number().min(0),
      latency_simulation: z.boolean(),
      min_latency_ms: z.number().min(0),
      max_latency_ms: z.number().min(0),
      market_impact_enabled: z.boolean(),
      realistic_fills: z.boolean(),
      partial_fill_probability: z.number().min(0).max(1),
      reject_probability: z.number().min(0).max(1),
    }).optional(),
    live_trading_confirmed: z.boolean().optional(),
  }),
  strategy: z.object({
    type: z.string().min(1, 'Strategy type is required'),
    parameters: z.object({
      entry_logic: z.object({
        name: z.string().min(1, 'Entry logic is required'),
        params: z.record(z.any()),
      }),
      exit_logic: z.object({
        name: z.string().min(1, 'Exit logic is required'),
        params: z.record(z.any()),
      }),
      position_size: z.number().min(0).max(1),
    }),
  }),
  risk_management: z.object({
    max_position_size: z.number().min(0),
    stop_loss_pct: z.number().min(0).max(100),
    take_profit_pct: z.number().min(0).max(100),
    max_daily_loss: z.number().min(0),
    max_daily_trades: z.number().min(1),
  }),
  notifications: z.object({
    position_opened: z.boolean(),
    position_closed: z.boolean(),
    email_enabled: z.boolean(),
    telegram_enabled: z.boolean(),
    error_notifications: z.boolean(),
  }),
});

type StrategyFormData = z.infer<typeof strategySchema>;

// Entry/Exit Logic Options
const entryLogicOptions = [
  { value: 'RSIBBVolumeEntryMixin', label: 'RSI + Bollinger Bands + Volume' },
  { value: 'RSIOrBBEntryMixin', label: 'RSI or Bollinger Bands' },
  { value: 'MACDEntryMixin', label: 'MACD Entry' },
  { value: 'SimpleEntryMixin', label: 'Simple Entry' },
];

const exitLogicOptions = [
  { value: 'ATRExitMixin', label: 'ATR-based Exit' },
  { value: 'RSIBBExitMixin', label: 'RSI + Bollinger Bands Exit' },
  { value: 'SimpleExitMixin', label: 'Simple Exit' },
  { value: 'TrailingStopExitMixin', label: 'Trailing Stop Exit' },
];

const StrategyForm: React.FC = () => {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();
  const isEditing = Boolean(id);

  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [showLiveWarning, setShowLiveWarning] = useState(false);

  // Form setup
  const {
    control,
    handleSubmit,
    watch,
    setValue,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<StrategyFormData>({
    resolver: zodResolver(strategySchema),
    defaultValues: {
      enabled: true,
      broker: {
        type: 'binance',
        trading_mode: 'paper',
        cash: 10000,
        paper_trading_config: {
          mode: 'realistic',
          initial_balance: 10000,
          commission_rate: 0.001,
          slippage_model: 'linear',
          base_slippage: 0.0005,
          latency_simulation: true,
          min_latency_ms: 20,
          max_latency_ms: 100,
          market_impact_enabled: true,
          realistic_fills: true,
          partial_fill_probability: 0.1,
          reject_probability: 0.02,
        },
      },
      strategy: {
        type: 'CustomStrategy',
        parameters: {
          entry_logic: {
            name: 'RSIBBVolumeEntryMixin',
            params: {
              e_rsi_period: 14,
              e_rsi_oversold: 30,
              e_bb_period: 20,
              e_bb_dev: 2.0,
              e_vol_ma_period: 20,
              e_min_volume_ratio: 1.1,
            },
          },
          exit_logic: {
            name: 'ATRExitMixin',
            params: {
              x_atr_period: 14,
              x_sl_multiplier: 1.5,
            },
          },
          position_size: 0.1,
        },
      },
      risk_management: {
        max_position_size: 1000,
        stop_loss_pct: 3.0,
        take_profit_pct: 6.0,
        max_daily_loss: 200,
        max_daily_trades: 5,
      },
      notifications: {
        position_opened: true,
        position_closed: true,
        email_enabled: false,
        telegram_enabled: true,
        error_notifications: true,
      },
    },
  });

  // Watch trading mode for live trading warning
  const tradingMode = watch('broker.trading_mode');
  useEffect(() => {
    setShowLiveWarning(tradingMode === 'live');
  }, [tradingMode]);

  // Fetch existing strategy for editing
  const { data: existingStrategy } = useQuery({
    queryKey: ['strategy', id],
    queryFn: () => getStrategy(id!),
    enabled: isEditing,
    onSuccess: (data) => {
      reset(data);
    },
  });

  // Fetch strategy templates
  const { data: templates } = useQuery({
    queryKey: ['strategy-templates'],
    queryFn: getStrategyTemplates,
  });

  // Create/Update mutations
  const createMutation = useMutation({
    mutationFn: createStrategy,
    onSuccess: () => {
      toast.success('Strategy created successfully');
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
      navigate('/strategies');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to create strategy');
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: StrategyFormData }) =>
      updateStrategy(id, data),
    onSuccess: () => {
      toast.success('Strategy updated successfully');
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
      navigate('/strategies');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to update strategy');
    },
  });

  // Handle form submission
  const onSubmit = (data: StrategyFormData) => {
    if (isEditing) {
      updateMutation.mutate({ id: id!, data });
    } else {
      createMutation.mutate(data);
    }
  };

  // Handle template selection
  const handleTemplateSelect = (templateKey: string) => {
    if (templates?.templates[templateKey]) {
      const template = templates.templates[templateKey];
      
      // Map template to form structure
      const formData: Partial<StrategyFormData> = {
        broker: template,
        // Keep existing strategy and other settings
      };
      
      Object.entries(formData).forEach(([key, value]) => {
        setValue(key as keyof StrategyFormData, value as any);
      });
      
      setSelectedTemplate(templateKey);
      toast.success(`Applied template: ${templateKey}`);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          {isEditing ? 'Edit Strategy' : 'Create New Strategy'}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<Cancel />}
            onClick={() => navigate('/strategies')}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            startIcon={<Save />}
            onClick={handleSubmit(onSubmit)}
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Saving...' : 'Save Strategy'}
          </Button>
        </Box>
      </Box>

      {/* Live Trading Warning */}
      {showLiveWarning && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Warning sx={{ mr: 1 }} />
            <Typography variant="body2">
              <strong>Live Trading Mode Selected:</strong> This strategy will use real money. 
              Please ensure you have thoroughly tested it in paper trading mode first.
            </Typography>
          </Box>
        </Alert>
      )}

      <form onSubmit={handleSubmit(onSubmit)}>
        <Grid container spacing={3}>
          {/* Basic Information */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Basic Information
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="id"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          label="Strategy ID"
                          fullWidth
                          error={!!errors.id}
                          helperText={errors.id?.message}
                          disabled={isEditing}
                        />
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="name"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          label="Strategy Name"
                          fullWidth
                          error={!!errors.name}
                          helperText={errors.name?.message}
                        />
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="symbol"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          label="Trading Symbol"
                          placeholder="e.g., BTCUSDT"
                          fullWidth
                          error={!!errors.symbol}
                          helperText={errors.symbol?.message}
                        />
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="enabled"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="Strategy Enabled"
                        />
                      )}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Template Selection */}
          {!isEditing && templates?.templates && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Strategy Templates
                  </Typography>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Select a template to pre-populate broker configuration
                  </Typography>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
                    {Object.entries(templates.templates).map(([key, template]: [string, any]) => (
                      <Chip
                        key={key}
                        label={template.description || key}
                        onClick={() => handleTemplateSelect(key)}
                        color={selectedTemplate === key ? 'primary' : 'default'}
                        variant={selectedTemplate === key ? 'filled' : 'outlined'}
                      />
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Broker Configuration */}
          <Grid item xs={12}>
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">Broker Configuration</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Controller
                      name="broker.type"
                      control={control}
                      render={({ field }) => (
                        <FormControl fullWidth>
                          <InputLabel>Broker Type</InputLabel>
                          <Select {...field} label="Broker Type">
                            <MenuItem value="binance">Binance</MenuItem>
                            <MenuItem value="ibkr">Interactive Brokers</MenuItem>
                            <MenuItem value="mock">Mock (Testing)</MenuItem>
                          </Select>
                        </FormControl>
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Controller
                      name="broker.trading_mode"
                      control={control}
                      render={({ field }) => (
                        <FormControl fullWidth>
                          <InputLabel>Trading Mode</InputLabel>
                          <Select {...field} label="Trading Mode">
                            <MenuItem value="paper">Paper Trading</MenuItem>
                            <MenuItem value="live">Live Trading</MenuItem>
                          </Select>
                        </FormControl>
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Controller
                      name="broker.cash"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          label="Initial Cash"
                          type="number"
                          fullWidth
                          onChange={(e) => field.onChange(parseFloat(e.target.value))}
                        />
                      )}
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Grid>

          {/* Strategy Configuration */}
          <Grid item xs={12}>
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">Strategy Configuration</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="strategy.parameters.entry_logic.name"
                      control={control}
                      render={({ field }) => (
                        <FormControl fullWidth>
                          <InputLabel>Entry Logic</InputLabel>
                          <Select {...field} label="Entry Logic">
                            {entryLogicOptions.map((option) => (
                              <MenuItem key={option.value} value={option.value}>
                                {option.label}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="strategy.parameters.exit_logic.name"
                      control={control}
                      render={({ field }) => (
                        <FormControl fullWidth>
                          <InputLabel>Exit Logic</InputLabel>
                          <Select {...field} label="Exit Logic">
                            {exitLogicOptions.map((option) => (
                              <MenuItem key={option.value} value={option.value}>
                                {option.label}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="strategy.parameters.position_size"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          label="Position Size (0-1)"
                          type="number"
                          fullWidth
                          inputProps={{ min: 0, max: 1, step: 0.01 }}
                          onChange={(e) => field.onChange(parseFloat(e.target.value))}
                        />
                      )}
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Grid>

          {/* Risk Management */}
          <Grid item xs={12}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">Risk Management</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="risk_management.max_position_size"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          label="Max Position Size ($)"
                          type="number"
                          fullWidth
                          onChange={(e) => field.onChange(parseFloat(e.target.value))}
                        />
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="risk_management.max_daily_loss"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          label="Max Daily Loss ($)"
                          type="number"
                          fullWidth
                          onChange={(e) => field.onChange(parseFloat(e.target.value))}
                        />
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Controller
                      name="risk_management.stop_loss_pct"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          label="Stop Loss (%)"
                          type="number"
                          fullWidth
                          onChange={(e) => field.onChange(parseFloat(e.target.value))}
                        />
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Controller
                      name="risk_management.take_profit_pct"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          label="Take Profit (%)"
                          type="number"
                          fullWidth
                          onChange={(e) => field.onChange(parseFloat(e.target.value))}
                        />
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Controller
                      name="risk_management.max_daily_trades"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          label="Max Daily Trades"
                          type="number"
                          fullWidth
                          onChange={(e) => field.onChange(parseInt(e.target.value))}
                        />
                      )}
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Grid>

          {/* Notifications */}
          <Grid item xs={12}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">Notifications</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="notifications.position_opened"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="Position Opened Notifications"
                        />
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="notifications.position_closed"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="Position Closed Notifications"
                        />
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="notifications.email_enabled"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="Email Notifications"
                        />
                      )}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Controller
                      name="notifications.telegram_enabled"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="Telegram Notifications"
                        />
                      )}
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Grid>
        </Grid>
      </form>
    </Box>
  );
};

export default StrategyForm;