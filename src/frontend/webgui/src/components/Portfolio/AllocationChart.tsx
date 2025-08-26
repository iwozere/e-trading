import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

const AllocationChart: React.FC = () => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Portfolio Allocation</Typography>
        <Box sx={{ height: 180, background: '#f5f5f5', borderRadius: 2, mt: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography color="text.secondary">[Pie/Bar chart will appear here]</Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default AllocationChart; 