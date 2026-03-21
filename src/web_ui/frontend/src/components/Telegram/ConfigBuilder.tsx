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
  Tab
} from '@mui/material';
import { Add as AddIcon, Delete as DeleteIcon, Code as CodeIcon } from '@mui/icons-material';
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
  const [mode, setMode] = useState<'alert' | 'schedule'>(initialMode);
  
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
        // Determine default params if switching to a plugin
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

  const generateAlertJson = () => {
    const rules = conditions.map(c => {
      const plugin = pluginRegistry.getPlugin(c.indicator);
      if (plugin) {
        return {
          plugin: plugin.name,
          params: c.pluginParams || {}
        };
      } else {
        const lhs = { indicator: { type: c.indicator.toUpperCase(), params: c.pluginParams || {} } };
        let rhs: any = {};
        if (!c.targetType || c.targetType === 'value') {
          rhs = { value: Number(c.value) || 0 };
        } else if (c.targetType === 'price') {
          rhs = { field: 'close' };
        } else if (c.targetType === 'indicator') {
          rhs = { indicator: { type: (c.targetIndicator || 'sma').toUpperCase() } };
        }
        
        return {
          [c.comparison]: { lhs, rhs }
        };
      }
    });
    
    return {
      ticker,
      timeframe,
      rule: rules.length > 1 ? { operator: ruleOperator, conditions: rules } : rules[0]
    };
  };

  const generateScheduleJson = () => {
    return {
      action: scheduleType === 'screener' ? 'report' : 'maintenance',
      parameters: {
        list_type: screenerList,
        scheduled_time: scheduleTime
      }
    };
  };

  const handleSave = () => {
    const config = mode === 'alert' ? generateAlertJson() : generateScheduleJson();
    onSave(config, mode);
  };

  return (
    <Card elevation={3}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h5">Configuration Builder</Typography>
          <Tabs value={mode} onChange={(_, val) => setMode(val)}>
            <Tab label="Alert Config" value="alert" />
            <Tab label="Schedule Config" value="schedule" />
          </Tabs>
        </Box>
        <Divider sx={{ mb: 3 }} />

        {mode === 'alert' && (
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <TextField 
                label="Ticker Symbol" 
                fullWidth 
                value={ticker} 
                onChange={e => setTicker(e.target.value.toUpperCase())} 
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField 
                select 
                label="Timeframe" 
                fullWidth 
                value={timeframe} 
                onChange={e => setTimeframe(e.target.value)}
              >
                {TIMEFRAMES.map(tf => <MenuItem key={tf} value={tf}>{tf}</MenuItem>)}
              </TextField>
            </Grid>

            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>Rule Logic</Typography>
              <Box mb={2}>
                <TextField
                  select
                  label="Match Operator"
                  size="small"
                  value={ruleOperator}
                  onChange={e => setRuleOperator(e.target.value)}
                  disabled={conditions.length <= 1}
                >
                  <MenuItem value="and">ALL of the following (AND)</MenuItem>
                  <MenuItem value="or">ANY of the following (OR)</MenuItem>
                </TextField>
              </Box>

              {conditions.map((condition, index) => {
                const plugin = pluginRegistry.getPlugin(condition.indicator);
                
                return (
                <Box key={condition.id} display="flex" gap={2} alignItems="center" mb={2} p={2} border="1px dashed #ccc" borderRadius={1}>
                  <TextField
                    select
                    label="Indicator / Plugin"
                    size="small"
                    value={condition.indicator}
                    onChange={e => updateCondition(condition.id, 'indicator', e.target.value)}
                    sx={{ minWidth: 150 }}
                  >
                    {/* Standard Indicators */}
                    {STANDARD_INDICATORS.map(ind => <MenuItem key={ind} value={ind}>{ind.toUpperCase()}</MenuItem>)}
                    
                    {/* UI Plugins */}
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
                        label="Target Type"
                        size="small"
                        value={condition.targetType || 'value'}
                        onChange={e => updateCondition(condition.id, 'targetType', e.target.value)}
                        sx={{ minWidth: 140 }}
                      >
                         <MenuItem value="value">Static Number</MenuItem>
                         <MenuItem value="price">Price Line</MenuItem>
                         <MenuItem value="indicator">Indicator</MenuItem>
                      </TextField>

                      {(!condition.targetType || condition.targetType === 'value') && (
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
                          label="Target Indicator"
                          size="small"
                          sx={{ width: 150 }}
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

              <Button startIcon={<AddIcon />} onClick={addCondition}>
                Add Condition
              </Button>
            </Grid>
          </Grid>
        )}

        {mode === 'schedule' && (
          <Grid container spacing={3}>
            <Grid item xs={12} sm={4}>
              <TextField 
                select 
                label="Type" 
                fullWidth 
                value={scheduleType} 
                onChange={e => setScheduleType(e.target.value)}
              >
                <MenuItem value="screener">Stock Screener</MenuItem>
                <MenuItem value="report">Daily Report</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField 
                label="Time (HH:MM)" 
                fullWidth 
                value={scheduleTime} 
                onChange={e => setScheduleTime(e.target.value)} 
              />
            </Grid>
            {scheduleType === 'screener' && (
              <Grid item xs={12} sm={4}>
                <TextField 
                  select 
                  label="Screener List" 
                  fullWidth 
                  value={screenerList} 
                  onChange={e => setScreenerList(e.target.value)}
                >
                  <MenuItem value="us_small_cap">US Small Cap</MenuItem>
                  <MenuItem value="us_large_cap">US Large Cap</MenuItem>
                  <MenuItem value="swiss_shares">Swiss Shares</MenuItem>
                </TextField>
              </Grid>
            )}
          </Grid>
        )}

        {/* Live Preview */}
        <Box mt={4} p={2} bgcolor="#f5f5f5" borderRadius={1} maxHeight={200} overflow="auto">
          <Typography variant="caption" color="text.secondary" display="flex" alignItems="center" mb={1}>
            <CodeIcon fontSize="small" sx={{ mr: 1 }} /> Generated JSON
          </Typography>
          <pre style={{ margin: 0, fontSize: '0.85rem' }}>
            {JSON.stringify(mode === 'alert' ? generateAlertJson() : generateScheduleJson(), null, 2)}
          </pre>
        </Box>

        <Box display="flex" justifyContent="flex-end" gap={2} mt={3}>
          <Button variant="outlined" color="inherit" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="contained" color="primary" onClick={handleSave}>
            Save {mode === 'alert' ? 'Alert' : 'Schedule'}
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ConfigBuilder;
