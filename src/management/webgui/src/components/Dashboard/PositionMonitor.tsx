import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemText } from '@mui/material';

const PositionMonitor: React.FC = () => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Position Monitor</Typography>
        <List>
          {/* Placeholder positions */}
          <ListItem>
            <ListItemText primary="BTCUSD" secondary="Long 0.5 @ $30,000" />
          </ListItem>
          <ListItem>
            <ListItemText primary="ETHUSD" secondary="Short 1.0 @ $2,000" />
          </ListItem>
        </List>
      </CardContent>
    </Card>
  );
};

export default PositionMonitor; 