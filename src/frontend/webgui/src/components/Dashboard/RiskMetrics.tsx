import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

const RiskMetrics: React.FC = () => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Risk Metrics</Typography>
        <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
          <Box sx={{ flex: 1, textAlign: 'center' }}>
            <Typography variant="h4" color="primary">2.5%</Typography>
            <Typography variant="body2" color="text.secondary">Max Drawdown</Typography>
          </Box>
          <Box sx={{ flex: 1, textAlign: 'center' }}>
            <Typography variant="h4" color="primary">1.8</Typography>
            <Typography variant="body2" color="text.secondary">Sharpe Ratio</Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default RiskMetrics; 