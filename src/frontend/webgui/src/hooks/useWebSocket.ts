import { useEffect, useRef, useState } from 'react';

export function useWebSocket<T = any>(url: string, onMessage?: (data: T) => void) {
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    // For mock: use a timeout to simulate connection
    ws.current = new window.WebSocket(url);
    ws.current.onopen = () => setConnected(true);
    ws.current.onclose = () => setConnected(false);
    ws.current.onerror = () => setConnected(false);
    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage && onMessage(data);
      } catch (e) {}
    };
    return () => {
      ws.current?.close();
    };
  }, [url, onMessage]);

  const send = (data: any) => {
    if (ws.current && connected) {
      ws.current.send(JSON.stringify(data));
    }
  };

  return { connected, send };
} 