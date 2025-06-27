import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const PerformanceCharts: React.FC = () => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Performance Charts</Typography>
        <div style={{ height: 200, background: '#f5f5f5', borderRadius: 8, marginTop: 16, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {/* Placeholder for chart */}
          <Typography color="text.secondary">[Chart will appear here]</Typography>
        </div>
      </CardContent>
    </Card>
  );
};

export default PerformanceCharts; 