import React from 'react';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import { PerformanceCharts, PositionMonitor, RiskMetrics } from '../components/Dashboard';

export default function Home() {
  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Box>
        <PerformanceCharts />
        <PositionMonitor />
        <RiskMetrics />
      </Box>
    </Container>
  );
}
