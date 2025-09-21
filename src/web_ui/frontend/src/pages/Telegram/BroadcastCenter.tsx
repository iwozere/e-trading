/**
 * Broadcast Center Page
 * 
 * Interface for sending broadcast messages to Telegram bot users
 * and monitoring message delivery status.
 */

import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress
} from '@mui/material';
import TelegramBreadcrumbs from '../../components/Telegram/TelegramBreadcrumbs';

const BroadcastCenter: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <TelegramBreadcrumbs />
      <Typography variant="h4" gutterBottom>
        Broadcast Center
      </Typography>
      
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
            <CircularProgress />
            <Typography variant="body1" sx={{ ml: 2 }}>
              Loading broadcast center...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default BroadcastCenter;