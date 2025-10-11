# Design Document

## Overview

This design outlines the migration of Telegram bot management functionality from the Flask-based admin panel to the modern React-based web UI system. The goal is to create a unified interface that combines trading operations and Telegram bot management using consistent technology patterns and user experience.

## Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Unified Web UI (React + Vite)           │
├────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │ Trading System  │  │   Telegram Bot Management       │  │
│  │   Management    │  │                                 │  │
│  │                 │  │ • Dashboard & Statistics        │  │
│  │ • Strategies    │  │ • User Management               │  │
│  │ • Backtesting   │  │ • Alert Management              │  │
│  │ • Live Trading  │  │ • Schedule Management           │  │
│  │ • Analytics     │  │ • Broadcast Messaging           │  │
│  │                 │  │ • Audit & Logging               │  │
│  └─────────────────┘  └─────────────────────────────────┘  │
├────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │   Shared UI     │  │      State Management           │  │
│  │   Components    │  │                                 │  │
│  │                 │  │ • React Query (Server State)    │  │
│  │ • Navigation    │  │ • Zustand (Client State)        │  │
│  │ • Auth          │  │ • Socket.io (Real-time)         │  │
│  │ • Layout        │  │ • Material-UI (Components)      │  │
│  └─────────────────┘  └─────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    Backend Services                        │
├────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │ Trading System  │  │   Telegram Bot Services         │  │
│  │   Backend       │  │                                 │  │
│  │                 │  │ • telegram_service (Database)   │  │
│  │ • FastAPI       │  │ • Bot API (HTTP Endpoints)      │  │
│  │ • WebSocket     │  │ • WebSocket Events              │  │
│  │ • Database      │  │ • Broadcast Service             │  │
│  └─────────────────┘  └─────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### Navigation Structure
```
src/web_ui/frontend/src/
├── pages/
│   ├── trading/                    # Existing trading pages
│   │   ├── Dashboard.tsx
│   │   ├── Strategies.tsx
│   │   └── ...
│   └── telegram/                   # New Telegram bot pages
│       ├── TelegramDashboard.tsx   # Statistics & overview
│       ├── UserManagement.tsx      # User CRUD operations
│       ├── AlertManagement.tsx     # Alert configuration
│       ├── ScheduleManagement.tsx  # Schedule configuration
│       ├── BroadcastCenter.tsx     # Message broadcasting
│       └── AuditLogs.tsx          # Command audit logs
├── components/
│   ├── telegram/                   # Telegram-specific components
│   │   ├── UserTable.tsx
│   │   ├── AlertTable.tsx
│   │   ├── ScheduleTable.tsx
│   │   ├── BroadcastForm.tsx
│   │   ├── AuditTable.tsx
│   │   └── StatisticsCards.tsx
│   └── shared/                     # Shared components
│       ├── Navigation.tsx          # Updated with Telegram sections
│       ├── Layout.tsx
│       └── ...
└── hooks/
    └── telegram/                   # Telegram-specific hooks
        ├── useTelegramUsers.ts
        ├── useTelegramAlerts.ts
        ├── useTelegramSchedules.ts
        ├── useBroadcast.ts
        └── useAuditLogs.ts
```

## Data Models

### User Management Data Models

```typescript
interface TelegramUser {
  telegram_user_id: string;
  email: string | null;
  verified: boolean;
  approved: boolean;
  language: string;
  is_admin: boolean;
  max_alerts: number;
  max_schedules: number;
  created_at: string;
  updated_at: string;
}

interface UserStats {
  total_users: number;
  verified_users: number;
  approved_users: number;
  pending_approvals: number;
  admin_users: number;
}
```

### Alert Management Data Models

```typescript
interface TelegramAlert {
  id: string;
  user_id: string;
  symbol: string;
  alert_type: 'price_above' | 'price_below' | 'percentage_change';
  target_value: number;
  current_value: number;
  is_active: boolean;
  rearm_config: RearmConfig;
  created_at: string;
  last_triggered: string | null;
  trigger_count: number;
}

interface RearmConfig {
  enabled: boolean;
  type: 'immediate' | 'time_based' | 'price_based';
  cooldown_minutes?: number;
  price_threshold?: number;
  hysteresis_percent?: number;
}

interface AlertStats {
  total_alerts: number;
  active_alerts: number;
  triggered_today: number;
  rearm_cycles: number;
}
```

### Schedule Management Data Models

```typescript
interface TelegramSchedule {
  id: string;
  user_id: string;
  schedule_type: 'daily' | 'weekly';
  time: string; // HH:MM format
  timezone: string;
  config: ScheduleConfig;
  is_active: boolean;
  created_at: string;
  last_executed: string | null;
}

interface ScheduleConfig {
  report_type: 'portfolio' | 'alerts' | 'screener';
  symbols?: string[];
  parameters?: Record<string, any>;
}

interface ScheduleStats {
  total_schedules: number;
  active_schedules: number;
  executed_today: number;
  failed_executions: number;
}
```

### Audit and Broadcast Data Models

```typescript
interface CommandAudit {
  id: string;
  telegram_user_id: string;
  command: string;
  full_message: string;
  success: boolean;
  error_message?: string;
  execution_time_ms: number;
  timestamp: string;
}

interface BroadcastMessage {
  id: string;
  message: string;
  sent_by: string;
  sent_at: string;
  total_recipients: number;
  successful_deliveries: number;
  failed_deliveries: number;
}

interface AuditStats {
  total_commands: number;
  successful_commands: number;
  failed_commands: number;
  recent_activity_24h: number;
  top_commands: Array<{ command: string; count: number }>;
}
```

## Components and Interfaces

### Core Page Components

#### TelegramDashboard Component
```typescript
interface TelegramDashboardProps {}

const TelegramDashboard: React.FC<TelegramDashboardProps> = () => {
  const { data: userStats } = useQuery(['telegram-user-stats'], getTelegramUserStats);
  const { data: alertStats } = useQuery(['telegram-alert-stats'], getTelegramAlertStats);
  const { data: scheduleStats } = useQuery(['telegram-schedule-stats'], getTelegramScheduleStats);
  const { data: auditStats } = useQuery(['telegram-audit-stats'], getTelegramAuditStats);

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4">Telegram Bot Dashboard</Typography>
      </Grid>
      
      {/* Statistics Cards */}
      <Grid item xs={12} md={3}>
        <StatCard title="Total Users" value={userStats?.total_users} />
      </Grid>
      <Grid item xs={12} md={3}>
        <StatCard title="Active Alerts" value={alertStats?.active_alerts} />
      </Grid>
      <Grid item xs={12} md={3}>
        <StatCard title="Active Schedules" value={scheduleStats?.active_schedules} />
      </Grid>
      <Grid item xs={12} md={3}>
        <StatCard title="Commands (24h)" value={auditStats?.recent_activity_24h} />
      </Grid>

      {/* Pending Approvals */}
      <Grid item xs={12}>
        <PendingApprovalsTable />
      </Grid>

      {/* Recent Activity */}
      <Grid item xs={12}>
        <RecentActivityTable />
      </Grid>
    </Grid>
  );
};
```

#### UserManagement Component
```typescript
interface UserManagementProps {}

const UserManagement: React.FC<UserManagementProps> = () => {
  const [filter, setFilter] = useState<'all' | 'verified' | 'approved' | 'pending'>('all');
  const { data: users, isLoading } = useQuery(
    ['telegram-users', filter], 
    () => getTelegramUsers(filter)
  );
  
  const verifyUserMutation = useMutation(verifyTelegramUser);
  const approveUserMutation = useMutation(approveTelegramUser);
  const resetEmailMutation = useMutation(resetTelegramUserEmail);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>User Management</Typography>
      
      {/* Filter Buttons */}
      <ButtonGroup sx={{ mb: 2 }}>
        <Button 
          variant={filter === 'all' ? 'contained' : 'outlined'}
          onClick={() => setFilter('all')}
        >
          All Users
        </Button>
        <Button 
          variant={filter === 'verified' ? 'contained' : 'outlined'}
          onClick={() => setFilter('verified')}
        >
          Verified
        </Button>
        <Button 
          variant={filter === 'approved' ? 'contained' : 'outlined'}
          onClick={() => setFilter('approved')}
        >
          Approved
        </Button>
        <Button 
          variant={filter === 'pending' ? 'contained' : 'outlined'}
          onClick={() => setFilter('pending')}
        >
          Pending
        </Button>
      </ButtonGroup>

      {/* User Table */}
      <UserTable 
        users={users || []}
        loading={isLoading}
        onVerify={verifyUserMutation.mutate}
        onApprove={approveUserMutation.mutate}
        onResetEmail={resetEmailMutation.mutate}
      />
    </Box>
  );
};
```

### Shared UI Components

#### UserTable Component
```typescript
interface UserTableProps {
  users: TelegramUser[];
  loading: boolean;
  onVerify: (userId: string) => void;
  onApprove: (userId: string) => void;
  onResetEmail: (userId: string) => void;
}

const UserTable: React.FC<UserTableProps> = ({
  users,
  loading,
  onVerify,
  onApprove,
  onResetEmail
}) => {
  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Telegram ID</TableCell>
            <TableCell>Email</TableCell>
            <TableCell align="center">Verified</TableCell>
            <TableCell align="center">Approved</TableCell>
            <TableCell align="center">Admin</TableCell>
            <TableCell align="center">Max Alerts</TableCell>
            <TableCell align="center">Max Schedules</TableCell>
            <TableCell align="center">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell colSpan={8} align="center">
                <CircularProgress />
              </TableCell>
            </TableRow>
          ) : (
            users.map((user) => (
              <UserTableRow
                key={user.telegram_user_id}
                user={user}
                onVerify={onVerify}
                onApprove={onApprove}
                onResetEmail={onResetEmail}
              />
            ))
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
};
```

#### StatCard Component
```typescript
interface StatCardProps {
  title: string;
  value: number | undefined;
  subtitle?: string;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  action?: React.ReactNode;
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  color = 'primary',
  action
}) => {
  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography color="textSecondary" gutterBottom variant="body2">
              {title}
            </Typography>
            <Typography variant="h4" color={color}>
              {value ?? '-'}
            </Typography>
            {subtitle && (
              <Typography color="textSecondary" variant="body2">
                {subtitle}
              </Typography>
            )}
          </Box>
          {action && (
            <Box>
              {action}
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};
```

## API Integration

### Backend Service Integration

The React interface will integrate with existing backend services through a new API layer that wraps the existing `telegram_service` functions:

#### API Service Layer
```typescript
// src/web_ui/frontend/src/services/telegramApi.ts

class TelegramApiService {
  private baseUrl = '/api/telegram';

  // User Management
  async getUsers(filter?: string): Promise<TelegramUser[]> {
    const response = await fetch(`${this.baseUrl}/users?filter=${filter || ''}`);
    return response.json();
  }

  async verifyUser(userId: string): Promise<void> {
    await fetch(`${this.baseUrl}/users/${userId}/verify`, { method: 'POST' });
  }

  async approveUser(userId: string): Promise<void> {
    await fetch(`${this.baseUrl}/users/${userId}/approve`, { method: 'POST' });
  }

  async resetUserEmail(userId: string): Promise<void> {
    await fetch(`${this.baseUrl}/users/${userId}/reset-email`, { method: 'POST' });
  }

  // Alert Management
  async getAlerts(filter?: string): Promise<TelegramAlert[]> {
    const response = await fetch(`${this.baseUrl}/alerts?filter=${filter || ''}`);
    return response.json();
  }

  async toggleAlert(alertId: string): Promise<void> {
    await fetch(`${this.baseUrl}/alerts/${alertId}/toggle`, { method: 'POST' });
  }

  async deleteAlert(alertId: string): Promise<void> {
    await fetch(`${this.baseUrl}/alerts/${alertId}`, { method: 'DELETE' });
  }

  // Schedule Management
  async getSchedules(filter?: string): Promise<TelegramSchedule[]> {
    const response = await fetch(`${this.baseUrl}/schedules?filter=${filter || ''}`);
    return response.json();
  }

  async toggleSchedule(scheduleId: string): Promise<void> {
    await fetch(`${this.baseUrl}/schedules/${scheduleId}/toggle`, { method: 'POST' });
  }

  // Broadcast
  async sendBroadcast(message: string): Promise<BroadcastResult> {
    const response = await fetch(`${this.baseUrl}/broadcast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    return response.json();
  }

  // Audit Logs
  async getAuditLogs(params: AuditLogParams): Promise<CommandAudit[]> {
    const queryString = new URLSearchParams(params).toString();
    const response = await fetch(`${this.baseUrl}/audit?${queryString}`);
    return response.json();
  }

  // Statistics
  async getUserStats(): Promise<UserStats> {
    const response = await fetch(`${this.baseUrl}/stats/users`);
    return response.json();
  }

  async getAlertStats(): Promise<AlertStats> {
    const response = await fetch(`${this.baseUrl}/stats/alerts`);
    return response.json();
  }

  async getScheduleStats(): Promise<ScheduleStats> {
    const response = await fetch(`${this.baseUrl}/stats/schedules`);
    return response.json();
  }

  async getAuditStats(): Promise<AuditStats> {
    const response = await fetch(`${this.baseUrl}/stats/audit`);
    return response.json();
  }
}

export const telegramApi = new TelegramApiService();
```

### React Query Hooks

```typescript
// src/web_ui/frontend/src/hooks/telegram/useTelegramUsers.ts

export const useTelegramUsers = (filter?: string) => {
  return useQuery({
    queryKey: ['telegram-users', filter],
    queryFn: () => telegramApi.getUsers(filter),
    staleTime: 30000, // 30 seconds
  });
};

export const useVerifyTelegramUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: telegramApi.verifyUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['telegram-users'] });
      queryClient.invalidateQueries({ queryKey: ['telegram-user-stats'] });
    },
  });
};

export const useApproveTelegramUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: telegramApi.approveUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['telegram-users'] });
      queryClient.invalidateQueries({ queryKey: ['telegram-user-stats'] });
    },
  });
};
```

## Real-time Communication

### WebSocket Integration

The system will use Socket.io for real-time updates, extending the existing WebSocket infrastructure:

```typescript
// src/web_ui/frontend/src/contexts/TelegramWebSocketContext.tsx

interface TelegramWebSocketContextType {
  connected: boolean;
  userStats: UserStats | null;
  alertStats: AlertStats | null;
  recentActivity: CommandAudit[];
}

export const TelegramWebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ 
  children 
}) => {
  const { socket } = useWebSocket();
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const [alertStats, setAlertStats] = useState<AlertStats | null>(null);
  const [recentActivity, setRecentActivity] = useState<CommandAudit[]>([]);

  useEffect(() => {
    if (!socket) return;

    // Subscribe to Telegram bot events
    socket.emit('subscribe', { channel: 'telegram_bot_events' });

    // Handle real-time updates
    socket.on('telegram_user_registered', (data) => {
      // Update user stats
      setUserStats(prev => prev ? { ...prev, total_users: prev.total_users + 1 } : null);
    });

    socket.on('telegram_user_verified', (data) => {
      setUserStats(prev => prev ? { ...prev, verified_users: prev.verified_users + 1 } : null);
    });

    socket.on('telegram_alert_triggered', (data) => {
      setAlertStats(prev => prev ? { ...prev, triggered_today: prev.triggered_today + 1 } : null);
    });

    socket.on('telegram_command_executed', (data: CommandAudit) => {
      setRecentActivity(prev => [data, ...prev.slice(0, 9)]); // Keep last 10
    });

    return () => {
      socket.off('telegram_user_registered');
      socket.off('telegram_user_verified');
      socket.off('telegram_alert_triggered');
      socket.off('telegram_command_executed');
    };
  }, [socket]);

  return (
    <TelegramWebSocketContext.Provider value={{
      connected: !!socket?.connected,
      userStats,
      alertStats,
      recentActivity
    }}>
      {children}
    </TelegramWebSocketContext.Provider>
  );
};
```

## Error Handling

### Centralized Error Handling

```typescript
// src/web_ui/frontend/src/utils/telegramErrorHandler.ts

export class TelegramApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public context?: string
  ) {
    super(message);
    this.name = 'TelegramApiError';
  }
}

export const handleTelegramApiError = (error: unknown, context: string) => {
  if (error instanceof TelegramApiError) {
    toast.error(`${context}: ${error.message}`);
  } else if (error instanceof Error) {
    toast.error(`${context}: ${error.message}`);
  } else {
    toast.error(`${context}: An unexpected error occurred`);
  }
  
  // Log to monitoring service
  console.error(`Telegram API Error in ${context}:`, error);
};
```

### Error Boundaries

```typescript
// src/web_ui/frontend/src/components/telegram/TelegramErrorBoundary.tsx

interface TelegramErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

export class TelegramErrorBoundary extends React.Component<
  React.PropsWithChildren<{}>,
  TelegramErrorBoundaryState
> {
  constructor(props: React.PropsWithChildren<{}>) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): TelegramErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Telegram component error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box p={3} textAlign="center">
          <Typography variant="h6" color="error" gutterBottom>
            Telegram Bot Management Error
          </Typography>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            Something went wrong with the Telegram bot interface.
          </Typography>
          <Button 
            variant="contained" 
            onClick={() => this.setState({ hasError: false })}
          >
            Try Again
          </Button>
        </Box>
      );
    }

    return this.props.children;
  }
}
```

## Testing Strategy

### Component Testing

```typescript
// src/web_ui/frontend/src/components/telegram/__tests__/UserTable.test.tsx

import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { UserTable } from '../UserTable';

const mockUsers: TelegramUser[] = [
  {
    telegram_user_id: '123456789',
    email: 'test@example.com',
    verified: true,
    approved: false,
    language: 'en',
    is_admin: false,
    max_alerts: 10,
    max_schedules: 5,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z'
  }
];

describe('UserTable', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    });
  });

  it('renders user data correctly', () => {
    const mockHandlers = {
      onVerify: jest.fn(),
      onApprove: jest.fn(),
      onResetEmail: jest.fn()
    };

    render(
      <QueryClientProvider client={queryClient}>
        <UserTable 
          users={mockUsers} 
          loading={false} 
          {...mockHandlers}
        />
      </QueryClientProvider>
    );

    expect(screen.getByText('123456789')).toBeInTheDocument();
    expect(screen.getByText('test@example.com')).toBeInTheDocument();
  });

  it('calls approve handler when approve button is clicked', () => {
    const mockHandlers = {
      onVerify: jest.fn(),
      onApprove: jest.fn(),
      onResetEmail: jest.fn()
    };

    render(
      <QueryClientProvider client={queryClient}>
        <UserTable 
          users={mockUsers} 
          loading={false} 
          {...mockHandlers}
        />
      </QueryClientProvider>
    );

    fireEvent.click(screen.getByText('Approve'));
    expect(mockHandlers.onApprove).toHaveBeenCalledWith('123456789');
  });
});
```

### API Integration Testing

```typescript
// src/web_ui/frontend/src/services/__tests__/telegramApi.test.ts

import { telegramApi } from '../telegramApi';

// Mock fetch
global.fetch = jest.fn();

describe('TelegramApiService', () => {
  beforeEach(() => {
    (fetch as jest.Mock).mockClear();
  });

  it('fetches users with correct parameters', async () => {
    const mockUsers = [{ telegram_user_id: '123', email: 'test@example.com' }];
    (fetch as jest.Mock).mockResolvedValueOnce({
      json: () => Promise.resolve(mockUsers)
    });

    const result = await telegramApi.getUsers('verified');

    expect(fetch).toHaveBeenCalledWith('/api/telegram/users?filter=verified');
    expect(result).toEqual(mockUsers);
  });

  it('verifies user with correct endpoint', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({ ok: true });

    await telegramApi.verifyUser('123456789');

    expect(fetch).toHaveBeenCalledWith(
      '/api/telegram/users/123456789/verify',
      { method: 'POST' }
    );
  });
});
```

## Performance Optimization

### Data Fetching Optimization

```typescript
// Implement pagination for large datasets
export const useTelegramUsersPaginated = (page: number, pageSize: number, filter?: string) => {
  return useQuery({
    queryKey: ['telegram-users-paginated', page, pageSize, filter],
    queryFn: () => telegramApi.getUsersPaginated(page, pageSize, filter),
    keepPreviousData: true, // Keep previous data while fetching new page
    staleTime: 30000,
  });
};

// Implement infinite scroll for audit logs
export const useTelegramAuditLogsInfinite = (filters: AuditLogFilters) => {
  return useInfiniteQuery({
    queryKey: ['telegram-audit-logs', filters],
    queryFn: ({ pageParam = 0 }) => 
      telegramApi.getAuditLogsPaginated(pageParam, 50, filters),
    getNextPageParam: (lastPage, pages) => 
      lastPage.hasMore ? pages.length : undefined,
  });
};
```

### Component Optimization

```typescript
// Memoize expensive components
export const UserTable = React.memo<UserTableProps>(({
  users,
  loading,
  onVerify,
  onApprove,
  onResetEmail
}) => {
  // Component implementation
});

// Virtualize large lists
import { FixedSizeList as List } from 'react-window';

export const VirtualizedAuditTable: React.FC<{ logs: CommandAudit[] }> = ({ logs }) => {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div style={style}>
      <AuditLogRow log={logs[index]} />
    </div>
  );

  return (
    <List
      height={600}
      itemCount={logs.length}
      itemSize={60}
      width="100%"
    >
      {Row}
    </List>
  );
};
```

## Security Considerations

### Authentication Integration

```typescript
// Extend existing auth context for Telegram bot permissions
interface AuthContextType {
  user: User | null;
  permissions: {
    trading: string[];
    telegram: string[];
  };
  hasPermission: (resource: string, action: string) => boolean;
}

export const useTelegramPermissions = () => {
  const { permissions, hasPermission } = useAuth();
  
  return {
    canManageUsers: hasPermission('telegram', 'manage_users'),
    canManageAlerts: hasPermission('telegram', 'manage_alerts'),
    canSendBroadcasts: hasPermission('telegram', 'send_broadcasts'),
    canViewAuditLogs: hasPermission('telegram', 'view_audit_logs'),
  };
};
```

### Input Validation

```typescript
// Validation schemas using Zod
import { z } from 'zod';

export const broadcastMessageSchema = z.object({
  message: z.string()
    .min(1, 'Message cannot be empty')
    .max(4096, 'Message too long (max 4096 characters)'),
  title: z.string().optional(),
});

export const alertConfigSchema = z.object({
  symbol: z.string().min(1, 'Symbol is required'),
  alert_type: z.enum(['price_above', 'price_below', 'percentage_change']),
  target_value: z.number().positive('Target value must be positive'),
  rearm_config: z.object({
    enabled: z.boolean(),
    type: z.enum(['immediate', 'time_based', 'price_based']),
    cooldown_minutes: z.number().optional(),
    price_threshold: z.number().optional(),
    hysteresis_percent: z.number().min(0).max(100).optional(),
  }),
});
```

This design provides a comprehensive foundation for migrating the Flask admin panel to a modern React-based interface while maintaining all existing functionality and improving user experience through better performance, real-time updates, and consistent UI patterns.