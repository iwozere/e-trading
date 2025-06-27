import { useState, useEffect } from 'react';
import { useWebSocket } from './useWebSocket';

export interface PortfolioPosition {
  symbol: string;
  amount: number;
  value: number;
}

export function usePortfolioData() {
  const [positions, setPositions] = useState<PortfolioPosition[]>([]);
  // For now, use a mock WebSocket URL
  useWebSocket('ws://localhost:8080/portfolio', (data) => {
    setPositions(data.positions || []);
  });

  useEffect(() => {
    // Initial mock data
    setPositions([
      { symbol: 'BTCUSD', amount: 0.5, value: 15000 },
      { symbol: 'ETHUSD', amount: 2, value: 7000 },
    ]);
  }, []);

  return positions;
} 