import React from 'react';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';

const StrategiesPage: React.FC = () => (
  <Container maxWidth="md" sx={{ mt: 4 }}>
    <Typography variant="h4" gutterBottom>Strategies</Typography>
    <Typography color="text.secondary">Manage and review your trading strategies here.</Typography>
  </Container>
);

export default StrategiesPage; 