import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import toast from 'react-hot-toast';
import { useAuthStore } from '../stores/authStore';

// WebSocket Events
interface WebSocketEvents {
  strategy_update: (data: any) => void;
  system_update: (data: any) => void;
  trade_notification: (data: any) => void;
  alert: (data: any) => void;
}

interface WebSocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  connectionStats: {
    connectedAt?: Date;
    reconnectAttempts: number;
    lastError?: string;
  };
  subscribeToStrategy: (strategyId: string) => void;
  unsubscribeFromStrategy: (strategyId: string) => void;
  subscribeToSystem: () => void;
  unsubscribeFromSystem: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: React.ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStats, setConnectionStats] = useState({
    reconnectAttempts: 0,
    lastError: undefined as string | undefined,
  });

  const { user, token, isAuthenticated } = useAuthStore();

  // Initialize WebSocket connection
  useEffect(() => {
    if (!isAuthenticated || !token) {
      return;
    }

    const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:5003';

    const newSocket = io(WS_URL, {
      auth: {
        token,
        user_id: user?.username,
      },
      transports: ['websocket'],
      upgrade: true,
      rememberUpgrade: true,
    });

    // Connection event handlers
    newSocket.on('connect', () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setConnectionStats(prev => ({
        ...prev,
        connectedAt: new Date(),
        reconnectAttempts: 0,
        lastError: undefined,
      }));
      toast.success('Connected to trading system');
    });

    newSocket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      setIsConnected(false);
      toast.error('Disconnected from trading system');
    });

    newSocket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setConnectionStats(prev => ({
        ...prev,
        reconnectAttempts: prev.reconnectAttempts + 1,
        lastError: error.message,
      }));

      if (connectionStats.reconnectAttempts === 0) {
        toast.error('Failed to connect to trading system');
      }
    });

    newSocket.on('reconnect', (attemptNumber) => {
      console.log('WebSocket reconnected after', attemptNumber, 'attempts');
      toast.success('Reconnected to trading system');
    });

    // Message handlers
    newSocket.on('welcome', (data) => {
      console.log('WebSocket welcome:', data);
    });

    newSocket.on('ping', () => {
      newSocket.emit('pong');
    });

    newSocket.on('strategy_update', (data) => {
      console.log('Strategy update:', data);
      // Handle strategy status updates
      // This could trigger React Query cache updates
    });

    newSocket.on('system_update', (data) => {
      console.log('System update:', data);
      // Handle system status updates
    });

    newSocket.on('trade_notification', (data) => {
      console.log('Trade notification:', data);
      toast.success(`Trade executed: ${data.data?.symbol} ${data.data?.side} ${data.data?.quantity}`);
    });

    newSocket.on('alert', (data) => {
      console.log('Alert:', data);
      const { priority, data: alertData } = data;

      switch (priority) {
        case 'error':
          toast.error(alertData.message || 'System error occurred');
          break;
        case 'warning':
          toast.error(alertData.message || 'System warning', {
            duration: 6000,
          });
          break;
        case 'info':
        default:
          toast(alertData.message || 'System notification');
          break;
      }
    });

    setSocket(newSocket);

    // Cleanup on unmount
    return () => {
      newSocket.close();
      setSocket(null);
      setIsConnected(false);
    };
  }, [isAuthenticated, token, user?.username]);

  // Subscription functions
  const subscribeToStrategy = useCallback((strategyId: string) => {
    if (socket && isConnected) {
      socket.emit('subscribe', {
        type: 'subscribe',
        subscription_type: 'strategy',
        strategy_id: strategyId,
      });
      console.log('Subscribed to strategy:', strategyId);
    }
  }, [socket, isConnected]);

  const unsubscribeFromStrategy = useCallback((strategyId: string) => {
    if (socket && isConnected) {
      socket.emit('unsubscribe', {
        type: 'unsubscribe',
        subscription_type: 'strategy',
        strategy_id: strategyId,
      });
      console.log('Unsubscribed from strategy:', strategyId);
    }
  }, [socket, isConnected]);

  const subscribeToSystem = useCallback(() => {
    if (socket && isConnected) {
      socket.emit('subscribe', {
        type: 'subscribe',
        subscription_type: 'system',
      });
      console.log('Subscribed to system events');
    }
  }, [socket, isConnected]);

  const unsubscribeFromSystem = useCallback(() => {
    if (socket && isConnected) {
      socket.emit('unsubscribe', {
        type: 'unsubscribe',
        subscription_type: 'system',
      });
      console.log('Unsubscribed from system events');
    }
  }, [socket, isConnected]);

  // Auto-subscribe to system events when connected
  useEffect(() => {
    if (isConnected) {
      subscribeToSystem();
    }
  }, [isConnected, subscribeToSystem]);

  const contextValue: WebSocketContextType = {
    socket,
    isConnected,
    connectionStats,
    subscribeToStrategy,
    unsubscribeFromStrategy,
    subscribeToSystem,
    unsubscribeFromSystem,
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};