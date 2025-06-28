import React from 'react';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';

const BacktestingPage: React.FC = () => (
  <Container maxWidth="md" sx={{ mt: 4 }}>
    <Typography variant="h4" gutterBottom>Backtesting</Typography>
    <Typography color="text.secondary">Run and analyze backtests for your strategies here.</Typography>
  </Container>
);

export default BacktestingPage; 