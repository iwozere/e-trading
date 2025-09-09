# Trading System Manager - Web GUI

A modern, professional web application for managing trading bots, strategies, and monitoring live trading performance. Built with Next.js, React, and Material-UI.

## Overview

The Trading System Manager is a comprehensive web interface that provides:

- **Real-time Dashboard**: Monitor trading performance, positions, and risk metrics
- **Strategy Management**: Create, configure, and optimize trading strategies
- **Backtesting Interface**: Test strategies with historical data
- **Live Trading Control**: Start, stop, and monitor live trading bots
- **Analytics & Reporting**: Detailed performance analysis and reporting

## Key Features

### 🎯 **Dashboard**
- Real-time performance charts
- Position monitoring
- Risk metrics visualization
- System health status

### 📊 **Strategy Management**
- Visual strategy builder
- Parameter optimization
- Strategy backtesting
- Performance comparison

### 🔄 **Live Trading**
- Bot status monitoring
- Real-time trade execution
- Risk management controls
- Emergency stop functionality

### 📈 **Analytics**
- Performance metrics
- Drawdown analysis
- Win/loss statistics
- Portfolio allocation

## Technology Stack

- **Frontend**: Next.js 15, React 19, TypeScript
- **UI Framework**: Material-UI (MUI) v7
- **Authentication**: NextAuth.js
- **Real-time**: WebSocket connections
- **Database**: Prisma ORM with SQLite/PostgreSQL
- **Styling**: Emotion (CSS-in-JS)

## Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Python trading system backend

### Installation
```bash
cd src/frontend/webgui
npm install
```

### Development
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

### Production Build
```bash
npm run build
npm start
```

## Project Structure

```
src/frontend/webgui/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── analytics/          # Analytics dashboard
│   │   ├── backtesting/        # Backtesting interface
│   │   ├── live-trading/       # Live trading control
│   │   ├── strategies/         # Strategy management
│   │   └── login/              # Authentication
│   ├── components/             # React components
│   │   ├── Dashboard/          # Dashboard components
│   │   ├── Portfolio/          # Portfolio management
│   │   └── StrategyBuilder/    # Strategy creation tools
│   ├── hooks/                  # Custom React hooks
│   └── pages/                  # Legacy pages (if any)
├── public/                     # Static assets
├── docs/                       # Documentation
└── package.json               # Dependencies and scripts
```

## Architecture

### Frontend Architecture
- **Next.js App Router**: Modern routing with server components
- **Component-based**: Modular, reusable React components
- **TypeScript**: Type-safe development
- **Material-UI**: Consistent, professional UI components

### Backend Integration
- **REST API**: Communication with Python trading backend
- **WebSocket**: Real-time data updates
- **Authentication**: Secure user management
- **Database**: Persistent data storage

### Real-time Features
- Live trade updates
- Performance monitoring
- System alerts
- Position tracking

## Development Workflow

### Adding New Features
1. Create components in `src/components/`
2. Add pages in `src/app/`
3. Implement hooks in `src/hooks/`
4. Update navigation in `NavBar.tsx`

### Styling Guidelines
- Use Material-UI components
- Follow responsive design principles
- Maintain consistent spacing and typography
- Use theme colors and variants

### State Management
- Local state with React hooks
- Global state with Context API (if needed)
- Server state with Next.js server components

## API Integration

### Trading Backend API
- **Base URL**: `http://localhost:8000/api`
- **Authentication**: Bearer token
- **Endpoints**:
  - `/bots` - Bot management
  - `/strategies` - Strategy operations
  - `/trades` - Trade data
  - `/performance` - Performance metrics

### WebSocket Connection
- **URL**: `ws://localhost:8000/ws`
- **Events**:
  - `trade_update` - New trade execution
  - `position_update` - Position changes
  - `performance_update` - Performance metrics
  - `system_alert` - System notifications

## Security

### Authentication
- NextAuth.js integration
- Session management
- Protected routes
- Role-based access control

### Data Protection
- HTTPS in production
- Secure API communication
- Input validation
- XSS protection

## Deployment

### Development
```bash
npm run dev
```

### Production
```bash
npm run build
npm start
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Monitoring & Logging

### Application Monitoring
- Performance metrics
- Error tracking
- User analytics
- System health checks

### Logging
- Client-side error logging
- API request/response logging
- User action tracking
- Performance monitoring

## Contributing

### Code Standards
- TypeScript strict mode
- ESLint configuration
- Prettier formatting
- Component documentation

### Testing
- Unit tests with Jest
- Integration tests
- E2E tests with Playwright
- Component testing with React Testing Library

## Support

### Documentation
- [Requirements.md](Requirements.md) - System requirements
- [Design.md](Design.md) - Architecture and design
- [Tasks.md](Tasks.md) - Development tasks

### Troubleshooting
- Check browser console for errors
- Verify API connectivity
- Check authentication status
- Review network requests

## Roadmap

### Phase 1: Core Features ✅
- [x] Basic dashboard
- [x] Navigation structure
- [x] Authentication setup
- [x] Material-UI integration

### Phase 2: Trading Integration 🚧
- [ ] Real-time data integration
- [ ] Bot management interface
- [ ] Strategy builder
- [ ] Performance charts

### Phase 3: Advanced Features 📋
- [ ] Advanced analytics
- [ ] Risk management tools
- [ ] Portfolio optimization
- [ ] Mobile responsiveness

### Phase 4: Production Ready 🎯
- [ ] Performance optimization
- [ ] Security hardening
- [ ] Monitoring integration
- [ ] Documentation completion
