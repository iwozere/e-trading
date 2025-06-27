import React from 'react';
import { Card, CardContent, Typography, Button, Box, List, ListItem, ListItemText } from '@mui/material';
import { useStrategyData } from '../../hooks/useStrategyData';

const BacktestRunner: React.FC = () => {
  const strategies = useStrategyData();
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Backtest Runner</Typography>
        <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
          <Button variant="contained" color="primary">Run Backtest</Button>
          <Button variant="outlined">Reset</Button>
        </Box>
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle2">Strategies</Typography>
          <List>
            {strategies.map((s) => (
              <ListItem key={s.id}>
                <ListItemText primary={s.name} secondary={`Status: ${s.status} | PnL: $${s.pnl}`} />
              </ListItem>
            ))}
          </List>
        </Box>
        <Box sx={{ mt: 3, height: 120, background: '#f5f5f5', borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography color="text.secondary">[Backtest results will appear here]</Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default BacktestRunner; 