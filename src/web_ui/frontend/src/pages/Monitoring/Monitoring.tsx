import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';

const Monitoring: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Real-Time Monitoring
      </Typography>
      
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Coming Soon
          </Typography>
          <Typography variant="body1">
            This page will show real-time monitoring of all trading strategies,
            including live P&L, position information, and system metrics.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Monitoring;