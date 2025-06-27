import React from 'react';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';

const LiveTradingPage: React.FC = () => (
  <Container maxWidth="md" sx={{ mt: 4 }}>
    <Typography variant="h4" gutterBottom>Live Trading</Typography>
    <Typography color="text.secondary">Monitor and manage your live trading bots here.</Typography>
  </Container>
);

export default LiveTradingPage; 