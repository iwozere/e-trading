import { useState, useEffect } from 'react';
import { useWebSocket } from './useWebSocket';

export interface Strategy {
  id: string;
  name: string;
  status: 'active' | 'inactive';
  pnl: number;
}

export function useStrategyData() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  // For now, use a mock WebSocket URL
  useWebSocket('ws://localhost:8080/strategies', (data) => {
    setStrategies(data.strategies || []);
  });

  useEffect(() => {
    // Initial mock data
    setStrategies([
      { id: '1', name: 'Momentum', status: 'active', pnl: 1200 },
      { id: '2', name: 'Mean Reversion', status: 'inactive', pnl: -300 },
    ]);
  }, []);

  return strategies;
} 