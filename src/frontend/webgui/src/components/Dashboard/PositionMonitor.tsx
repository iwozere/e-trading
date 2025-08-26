"use client";

import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemText } from '@mui/material';
import { usePortfolioData } from '../../hooks/usePortfolioData';

const PositionMonitor: React.FC = () => {
  const positions = usePortfolioData();
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Position Monitor</Typography>
        <List>
          {positions.map((pos) => (
            <ListItem key={pos.symbol}>
              <ListItemText primary={pos.symbol} secondary={`Amount: ${pos.amount} | Value: $${pos.value.toLocaleString()}`} />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default PositionMonitor; 