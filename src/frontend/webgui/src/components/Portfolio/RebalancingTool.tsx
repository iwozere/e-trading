import React from 'react';
import { Card, CardContent, Typography, Button, Box } from '@mui/material';

const RebalancingTool: React.FC = () => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Rebalancing Tool</Typography>
        <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
          <Button variant="contained" color="primary">Rebalance</Button>
        </Box>
        <Box sx={{ mt: 3, height: 80, background: '#f5f5f5', borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography color="text.secondary">[Rebalancing status will appear here]</Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default RebalancingTool; 