import React from 'react';
import { Card, CardContent, Typography, Grid } from '@mui/material';

const RiskAnalysis: React.FC = () => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Portfolio Risk Analysis</Typography>
        <Grid container spacing={2} mt={1}>
          {/* Placeholder risk metrics */}
          <Grid item xs={4}>
            <Typography variant="body2" color="text.secondary">Portfolio VaR</Typography>
            <Typography variant="subtitle1">$2,500</Typography>
          </Grid>
          <Grid item xs={4}>
            <Typography variant="body2" color="text.secondary">Beta</Typography>
            <Typography variant="subtitle1">0.85</Typography>
          </Grid>
          <Grid item xs={4}>
            <Typography variant="body2" color="text.secondary">Diversification</Typography>
            <Typography variant="subtitle1">High</Typography>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default RiskAnalysis; 