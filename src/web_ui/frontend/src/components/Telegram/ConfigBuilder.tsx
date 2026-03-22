import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  MenuItem,
  Button,
  Grid,
  Divider,
  IconButton,
  Tabs,
  Tab,
  Stepper,
  Step,
  StepLabel,
  Paper,
  alpha,
  useTheme,
} from '@mui/material';
import { 
  Add as AddIcon, 
  Delete as DeleteIcon, 
  Code as CodeIcon,
  ChevronRight,
  ChevronLeft,
  AutoFixHigh,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { pluginRegistry } from './plugins/PluginRegistry';

interface ConfigBuilderProps {
  onSave: (config: any, mode: 'alert' | 'schedule') => void;
  onCancel: () => void;
  initialMode?: 'alert' | 'schedule';
}

interface Condition {
  id: string;
  indicator: string;
  comparison: string;
  targetType: 'value' | 'price' | 'indicator';
  value: string;
  targetIndicator?: string;
  pluginParams?: any;
}

const STANDARD_INDICATORS = ['rsi', 'sma', 'ema', 'macd', 'price'];
const COMPARISONS = [
  { value: 'gt', label: 'Greater Than (>)' },
  { value: 'lt', label: 'Less Than (<)' },
  { value: 'crosses_above', label: 'Crosses Above' },
  { value: 'crosses_below', label: 'Crosses Below' }
];

const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"];

const ConfigBuilder: React.FC<ConfigBuilderProps> = ({ onSave, onCancel, initialMode = 'alert' }) => {
  const theme = useTheme();
  const [mode, setMode] = useState<'alert' | 'schedule'>(initialMode);
  const [activeStep, setActiveStep] = useState(0);
  
  // Alert specific state
  const [ticker, setTicker] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');
  const [ruleOperator, setRuleOperator] = useState('and');
  const [conditions, setConditions] = useState<Condition[]>([
    { id: '1', indicator: 'rsi', comparison: 'lt', value: '30', targetType: 'value', targetIndicator: 'sma', pluginParams: {} }
  ]);

  // Schedule specific state
  const [scheduleType, setScheduleType] = useState('screener');
  const [scheduleTime, setScheduleTime] = useState('09:00');
  const [screenerList, setScreenerList] = useState('us_small_cap');

  const addCondition = () => {
    setConditions([
      ...conditions, 
      { id: Date.now().toString(), indicator: 'rsi', comparison: 'gt', value: '70', targetType: 'value', targetIndicator: 'sma', pluginParams: {} }
    ]);
  };

  const updateCondition = (id: string, field: keyof Condition, value: any) => {
    setConditions(conditions.map(c => {
      if (c.id === id) {
        const updated = { ...c, [field]: value };
        if (field === 'indicator') {
          const plugin = pluginRegistry.getPlugin(value);
          if (plugin) {
            updated.pluginParams = { ...plugin.defaultParams };
          }
        }
        return updated;
      }
      return c;
    }));
  };

  const removeCondition = (id: string) => {
    setConditions(conditions.filter(c => c.id !== id));
  };

  const getNaturalSummary = () => {
    if (mode === 'schedule') {
        return `Run ${scheduleType === 'screener' ? 'Stock Screener' : 'Daily Report'} on "${screenerList}" at ${scheduleTime}.`;
    }

    const sentences = conditions.map(c => {
      const ind = c.indicator.toUpperCase();
      const comp = COMPARISONS.find(comp => comp.value === c.comparison)?.label.toLowerCase();
      let target = '';
      if (c.targetType === 'value') target = c.value;
      else if (c.targetType === 'price') target = 'Price';
      else if (c.targetType === 'indicator') target = (c.targetIndicator || 'sma').toUpperCase();
      
      const plugin = pluginRegistry.getPlugin(c.indicator);
      if (plugin) return `[${plugin.label}] trigger detected`;
      
      return `${ind} ${comp} ${target}`;
    });

    const op = ruleOperator.toUpperCase();
    const summary = sentences.join(` ${op} `);
    return `Alert me for ${ticker} (${timeframe}) when ${summary}.`;
  };

  const generateAlertJson = () => {
    const rules = conditions.map(c => {
      const plugin = pluginRegistry.getPlugin(c.indicator);
      if (plugin) {
        return { plugin: plugin.name, params: c.pluginParams || {} };
      } else {
        const lhs = { indicator: { type: c.indicator.toUpperCase(), params: c.pluginParams || {} } };
        let rhs: any = {};
        if (c.targetType === 'value') rhs = { value: Number(c.value) || 0 };
        else if (c.targetType === 'price') rhs = { field: 'close' };
        else if (c.targetType === 'indicator') rhs = { indicator: { type: (c.targetIndicator || 'sma').toUpperCase() } };
        
        return { [c.comparison]: { lhs, rhs } };
      }
    });
    
    return {
      ticker,
      timeframe,
      rule: rules.length > 1 ? { operator: ruleOperator, conditions: rules } : rules[0]
    };
  };

  const generateScheduleJson = () => ({
    action: scheduleType === 'screener' ? 'report' : 'maintenance',
    parameters: { list_type: screenerList, scheduled_time: scheduleTime }
  });

  const handleSave = () => {
    const config = mode === 'alert' ? generateAlertJson() : generateScheduleJson();
    onSave(config, mode);
  };

  const steps = mode === 'alert' ? ['Asset Info', 'Build Rules', 'Finalize'] : ['Schedule Setup', 'Finalize'];

  return (
    <Card sx={{ border: '1px solid rgba(255,255,255,0.05)' }}>
      <CardContent sx={{ p: 4 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
          <Typography variant="h5" sx={{ fontWeight: 800, fontFamily: 'Outfit' }}>
            Logic Engine Builder
          </Typography>
          <Tabs 
            value={mode} 
            onChange={(_, val) => { setMode(val); setActiveStep(0); }}
            sx={{ 
                bgcolor: 'rgba(255,255,255,0.03)', 
                borderRadius: 2, 
                p: 0.5,
                '& .MuiTabs-indicator': { height: '100%', borderRadius: 1.5, zIndex: 0, opacity: 0.1 }
            }}
          >
            <Tab label="Alert Config" value="alert" sx={{ fontWeight: 700, zIndex: 1 }} />
            <Tab label="Schedule Config" value="schedule" sx={{ fontWeight: 700, zIndex: 1 }} />
          </Tabs>
        </Box>

        <Stepper activeStep={activeStep} sx={{ mb: 6 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel 
                StepIconProps={{ 
                   sx: { 
                     '&.Mui-active': { color: 'primary.main' },
                     '&.Mui-completed': { color: 'success.main' }
                   } 
                }}
              >
                <Typography variant="body2" sx={{ fontWeight: 700 }}>{label}</Typography>
              </StepLabel>
            </Step>
          ))}
        </Stepper>

        <AnimatePresence mode="wait">
          <motion.div
            key={`${mode}-${activeStep}`}
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -20, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {/* STEP 0: ASSET / BASIC SETUP */}
            {activeStep === 0 && (
              <Box>
                {mode === 'alert' ? (
                  <Grid container spacing={4}>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 700, color: 'text.secondary' }}>ASSET SYMBOL</Typography>
                      <TextField 
                        fullWidth 
                        placeholder="e.g. BTCUSDT"
                        value={ticker} 
                        onChange={e => setTicker(e.target.value.toUpperCase())} 
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 700, color: 'text.secondary' }}>TIMEFRAME</Typography>
                      <TextField 
                        select 
                        fullWidth 
                        value={timeframe} 
                        onChange={e => setTimeframe(e.target.value)}
                      >
                        {TIMEFRAMES.map(tf => <MenuItem key={tf} value={tf}>{tf}</MenuItem>)}
                      </TextField>
                    </Grid>
                  </Grid>
                ) : (
                  <Grid container spacing={4}>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 700, color: 'text.secondary' }}>TYPE</Typography>
                      <TextField select fullWidth value={scheduleType} onChange={e => setScheduleType(e.target.value)}>
                        <MenuItem value="screener">Stock Screener</MenuItem>
                        <MenuItem value="report">Daily Report</MenuItem>
                      </TextField>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 700, color: 'text.secondary' }}>LIST TYPE</Typography>
                      <TextField select fullWidth value={screenerList} onChange={e => setScreenerList(e.target.value)}>
                        <MenuItem value="us_small_cap">US Small Cap</MenuItem>
                        <MenuItem value="us_large_cap">US Large Cap</MenuItem>
                        <MenuItem value="swiss_shares">Swiss Shares</MenuItem>
                      </TextField>
                    </Grid>
                  </Grid>
                )}
              </Box>
            )}

            {/* STEP 1: LOGIC BUILDER */}
            {activeStep === 1 && mode === 'alert' && (
              <Box>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                  <Box>
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, color: 'text.secondary' }}>RULE SET OPERATOR</Typography>
                    <Typography variant="caption" color="text.disabled">How should we evaluate multiple conditions?</Typography>
                  </Box>
                  <TextField
                    select
                    size="small"
                    value={ruleOperator}
                    onChange={e => setRuleOperator(e.target.value)}
                    disabled={conditions.length <= 1}
                    sx={{ width: 250 }}
                  >
                    <MenuItem value="and">ALL of the following (AND)</MenuItem>
                    <MenuItem value="or">ANY of the following (OR)</MenuItem>
                  </TextField>
                </Box>

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mb: 3 }}>
                  {conditions.map((condition) => {
                    const plugin = pluginRegistry.getPlugin(condition.indicator);
                    return (
                      <Box 
                        key={condition.id} 
                        sx={{ 
                          p: 2, 
                          borderRadius: 2, 
                          bgcolor: 'rgba(255,255,255,0.02)',
                          border: '1px solid rgba(255,255,255,0.05)',
                          display: 'flex', 
                          gap: 2, 
                          alignItems: 'center' 
                        }}
                      >
                        <TextField
                          select
                          label="Indicator"
                          size="small"
                          value={condition.indicator}
                          onChange={e => updateCondition(condition.id, 'indicator', e.target.value)}
                          sx={{ minWidth: 160 }}
                        >
                          {STANDARD_INDICATORS.map(ind => <MenuItem key={ind} value={ind}>{ind.toUpperCase()}</MenuItem>)}
                          <Divider />
                          {pluginRegistry.getAllPlugins().map(p => (
                            <MenuItem key={p.name} value={p.name}>✨ {p.label}</MenuItem>
                          ))}
                        </TextField>

                        {plugin ? (
                          <Box flexGrow={1}>
                            <plugin.FormComponent 
                              params={condition.pluginParams || {}} 
                              onChange={(newParams) => updateCondition(condition.id, 'pluginParams', newParams)} 
                            />
                          </Box>
                        ) : (
                          <>
                            <TextField
                                select
                                label="Condition"
                                size="small"
                                value={condition.comparison}
                                onChange={e => updateCondition(condition.id, 'comparison', e.target.value)}
                                sx={{ minWidth: 150 }}
                            >
                                {COMPARISONS.map(cmp => <MenuItem key={cmp.value} value={cmp.value}>{cmp.label}</MenuItem>)}
                            </TextField>
                            <TextField
                                select
                                label="Against"
                                size="small"
                                value={condition.targetType}
                                onChange={e => updateCondition(condition.id, 'targetType', e.target.value)}
                                sx={{ minWidth: 140 }}
                            >
                                <MenuItem value="value">Static Value</MenuItem>
                                <MenuItem value="price">Price Action</MenuItem>
                                <MenuItem value="indicator">Other Indicator</MenuItem>
                            </TextField>
                            {condition.targetType === 'value' && (
                                <TextField
                                    label="Value"
                                    size="small"
                                    sx={{ width: 100 }}
                                    value={condition.value}
                                    onChange={e => updateCondition(condition.id, 'value', e.target.value)}
                                />
                            )}
                            {condition.targetType === 'indicator' && (
                                <TextField
                                    select
                                    label="Target"
                                    size="small"
                                    sx={{ width: 140 }}
                                    value={condition.targetIndicator || 'sma'}
                                    onChange={e => updateCondition(condition.id, 'targetIndicator', e.target.value)}
                                >
                                    {STANDARD_INDICATORS.map(ind => <MenuItem key={ind} value={ind}>{ind.toUpperCase()}</MenuItem>)}
                                </TextField>
                            )}
                          </>
                        )}
                        <IconButton color="error" onClick={() => removeCondition(condition.id)} disabled={conditions.length === 1}>
                          <DeleteIcon />
                        </IconButton>
                      </Box>
                    );
                  })}
                </Box>
                <Button variant="outlined" startIcon={<AddIcon />} onClick={addCondition} sx={{ borderStyle: 'dashed' }}>
                  Add Condition
                </Button>
              </Box>
            )}

            {/* FINAL STEP: REVIEW & SAVE */}
            {((activeStep === 2 && mode === 'alert') || (activeStep === 1 && mode === 'schedule')) && (
              <Box>
                <Box sx={{ 
                    p: 3, 
                    borderRadius: 2, 
                    bgcolor: 'primary.main', 
                    color: '#000',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 2,
                    mb: 4
                }}>
                    <AutoFixHigh />
                    <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>NATURAL LANGUAGE SUMMARY</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>{getNaturalSummary()}</Typography>
                    </Box>
                </Box>

                <Box p={2} bgcolor="rgba(0,0,0,0.2)" borderRadius={2} border="1px solid rgba(255,255,255,0.05)">
                  <Typography variant="caption" color="text.secondary" display="flex" alignItems="center" mb={1}>
                    <CodeIcon fontSize="small" sx={{ mr: 1 }} /> Generated Specification
                  </Typography>
                  <pre style={{ margin: 0, fontSize: '0.85rem', color: theme.palette.primary.main }}>
                    {JSON.stringify(mode === 'alert' ? generateAlertJson() : generateScheduleJson(), null, 2)}
                  </pre>
                </Box>
              </Box>
            )}
          </motion.div>
        </AnimatePresence>

        <Divider sx={{ my: 4 }} />

        <Box display="flex" justifyContent="space-between" mt={3}>
          <Button variant="outlined" color="inherit" onClick={onCancel}>
            Cancel
          </Button>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button 
                disabled={activeStep === 0} 
                onClick={() => setActiveStep(s => s - 1)}
                startIcon={<ChevronLeft />}
            >
                Back
            </Button>
            {activeStep === steps.length - 1 ? (
                <Button variant="contained" color="primary" onClick={handleSave} sx={{ px: 4, fontWeight: 800 }}>
                    DEPLOY CONFIG
                </Button>
            ) : (
                <Button 
                    variant="contained" 
                    onClick={() => setActiveStep(s => s + 1)}
                    endIcon={<ChevronRight />}
                    sx={{ px: 4 }}
                >
                    Continue
                </Button>
            )}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ConfigBuilder;
