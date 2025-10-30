import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';

const Administration: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        System Administration
      </Typography>
      
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Coming Soon
          </Typography>
          <Typography variant="body1">
            This page will provide system administration tools, including
            service management, configuration, logs, and backups.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Administration;