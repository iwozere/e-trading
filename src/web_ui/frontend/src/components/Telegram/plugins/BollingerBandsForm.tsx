import React from 'react';
import { Box, MenuItem, TextField } from '@mui/material';
import { FormComponentProps } from './types';

export const defaultBBandsParams = {
  type: 'cross_upper_up',
  period: 14,
  dev_up: 2.0,
  dev_down: 2.0
};

const SIGNAL_TYPES = [
  { value: 'cross_upper_up', label: 'Crosses Upper Band (Up)' },
  { value: 'cross_upper_down', label: 'Crosses Upper Band (Down)' },
  { value: 'cross_lower_up', label: 'Crosses Lower Band (Up)' },
  { value: 'cross_lower_down', label: 'Crosses Lower Band (Down)' },
  { value: 'touch_upper', label: 'Touches Upper Band' },
  { value: 'touch_lower', label: 'Touches Lower Band' }
];

const BollingerBandsForm: React.FC<FormComponentProps> = ({ params, onChange }) => {
  const handleChange = (field: string, value: any) => {
    onChange({
      ...params,
      [field]: value
    });
  };

  return (
    <Box display="flex" gap={2} alignItems="center">
      <TextField
        select
        label="Signal Type"
        size="small"
        value={params.type || 'cross_upper_up'}
        onChange={e => handleChange('type', e.target.value)}
        sx={{ minWidth: 220 }}
      >
        {SIGNAL_TYPES.map(sig => (
          <MenuItem key={sig.value} value={sig.value}>{sig.label}</MenuItem>
        ))}
      </TextField>

      <TextField
        label="Period"
        type="number"
        size="small"
        value={params.period || 14}
        onChange={e => handleChange('period', parseInt(e.target.value) || 14)}
        sx={{ minWidth: 80 }}
      />
      
      <TextField
        label="Dev Up"
        type="number"
        size="small"
        inputProps={{ step: 0.1 }}
        value={params.dev_up || 2.0}
        onChange={e => handleChange('dev_up', parseFloat(e.target.value) || 2.0)}
        sx={{ minWidth: 80 }}
      />

      <TextField
        label="Dev Down"
        type="number"
        size="small"
        inputProps={{ step: 0.1 }}
        value={params.dev_down || 2.0}
        onChange={e => handleChange('dev_down', parseFloat(e.target.value) || 2.0)}
        sx={{ minWidth: 80 }}
      />
    </Box>
  );
};

export default BollingerBandsForm;
