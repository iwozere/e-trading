/**
 * Telegram Bot Dashboard Page
 * 
 * Main dashboard for Telegram bot management showing statistics,
 * pending approvals, and recent activity.
 */

import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CircularProgress
} from '@mui/material';
import TelegramBreadcrumbs from '../../components/Telegram/TelegramBreadcrumbs';

const TelegramDashboard: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <TelegramBreadcrumbs />
      <Typography variant="h4" gutterBottom>
        Telegram Bot Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
                <CircularProgress />
                <Typography variant="body1" sx={{ ml: 2 }}>
                  Loading dashboard...
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TelegramDashboard;