import React from 'react';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';

const AnalyticsPage: React.FC = () => (
  <Container maxWidth="md" sx={{ mt: 4 }}>
    <Typography variant="h4" gutterBottom>Analytics</Typography>
    <Typography color="text.secondary">View advanced analytics and reports here.</Typography>
  </Container>
);

export default AnalyticsPage; 