# WebGUI – Professional Trading System Management

## Overview
A modern, enterprise-grade web application for managing trading systems, built with Next.js, TypeScript, Material-UI, NextAuth.js, and real-time WebSocket data.

## Structure
```
management/webgui/
├── src/
│   ├── app/                # Next.js app directory (routing, layouts, pages)
│   ├── components/         # UI components (Dashboard, StrategyBuilder, Portfolio, ...)
│   ├── hooks/              # Custom React hooks (WebSocket, data fetching, ...)
├── public/                 # Static assets
├── Dockerfile              # Docker support
├── docker-compose.yml      # Docker Compose for dev/prod
├── package.json            # Project config
└── ...
```

## Features
- Dashboard: Performance charts, position monitor, risk metrics
- Strategy Builder: Visual editor, backtest runner, parameter optimizer
- Portfolio: Allocation chart, rebalancing tool, risk analysis
- Real-time updates via WebSocket (mock server included)
- Authentication (Google, GitHub, Facebook, credentials)
- Docker deployment

## Setup & Usage
### Local Development
```bash
cd management/webgui
npm install
npm run dev
# In another terminal:
node src/mock-websocket-server.js
```
App: http://localhost:3000  |  WebSocket: ws://localhost:8080

### Docker
```bash
docker-compose up --build
```

## Authentication
- Uses NextAuth.js with Google, GitHub, Facebook, and credentials providers.
- Configure provider secrets in `.env.local` (see NextAuth.js docs).
- Default credentials: `admin` / `admin` (for mock login).

## Real-Time Data
- WebSocket endpoints:
  - `ws://localhost:8080/strategies` (mock strategies)
  - `ws://localhost:8080/portfolio` (mock portfolio)
- Hooks: `useWebSocket`, `useStrategyData`, `usePortfolioData`

## Customization
- Add new pages/components in `src/app` and `src/components`.
- Extend authentication by adding more NextAuth.js providers.
- Replace mock WebSocket server with your real backend as needed.

---
For questions or advanced customization, see the code comments and Next.js/NextAuth.js documentation. 