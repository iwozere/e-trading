import React from 'react';
import { Card, CardContent, Typography, Grid } from '@mui/material';

const RiskMetrics: React.FC = () => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Risk Metrics</Typography>
        <Grid container spacing={2} mt={1}>
          {/* Placeholder risk metrics */}
          <Grid item xs={4}>
            <Typography variant="body2" color="text.secondary">Max Drawdown</Typography>
            <Typography variant="subtitle1">-12.5%</Typography>
          </Grid>
          <Grid item xs={4}>
            <Typography variant="body2" color="text.secondary">VaR (95%)</Typography>
            <Typography variant="subtitle1">$1,200</Typography>
          </Grid>
          <Grid item xs={4}>
            <Typography variant="body2" color="text.secondary">Sharpe Ratio</Typography>
            <Typography variant="subtitle1">1.45</Typography>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default RiskMetrics; 