import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

const PerformanceCharts: React.FC = () => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Performance Charts</Typography>
        <Box sx={{ mt: 2, height: 200, background: '#f5f5f5', borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography color="text.secondary">[Performance charts will appear here]</Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PerformanceCharts; 