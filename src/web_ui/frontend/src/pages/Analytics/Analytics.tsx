import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';

const Analytics: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Performance Analytics
      </Typography>
      
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Coming Soon
          </Typography>
          <Typography variant="body1">
            This page will show detailed performance analytics, including
            charts, trade analysis, and comprehensive reporting.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Analytics;