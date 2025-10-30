// Simple mock WebSocket server for development
const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws, req) => {
  const url = req.url;
  let interval;
  if (url === '/strategies') {
    interval = setInterval(() => {
      ws.send(JSON.stringify({
        strategies: [
          { id: '1', name: 'Momentum', status: 'active', pnl: Math.round(1000 + Math.random() * 500) },
          { id: '2', name: 'Mean Reversion', status: 'inactive', pnl: Math.round(-500 + Math.random() * 200) },
        ],
      }));
    }, 2000);
  } else if (url === '/portfolio') {
    interval = setInterval(() => {
      ws.send(JSON.stringify({
        positions: [
          { symbol: 'BTCUSD', amount: 0.5, value: Math.round(15000 + Math.random() * 1000) },
          { symbol: 'ETHUSD', amount: 2, value: Math.round(7000 + Math.random() * 500) },
        ],
      }));
    }, 2000);
  }
  ws.on('close', () => clearInterval(interval));
});

console.log('Mock WebSocket server running on ws://localhost:8080'); 