import api from './tradingApi';

export interface MessageItem {
  id: number;
  message_type: string;
  priority: string;
  channels: string[];
  recipient_id: string | null;
  template_name: string | null;
  content: Record<string, any>;
  status: string;
  created_at: string | null;
  scheduled_for: string | null;
  processed_at: string | null;
  retry_count: number;
  last_error: string | null;
}

export interface MessageSearchResult {
  total: number;
  items: MessageItem[];
  limit: number;
  offset: number;
}

export interface MessageSearchParams {
  recipient_id?: string;
  search?: string;
  start_date?: string;
  end_date?: string;
  status?: string;
  channel?: string;
  days?: number;
  limit?: number;
  offset?: number;
}

export const notificationsApi = {
  searchMessages: async (params: MessageSearchParams): Promise<MessageSearchResult> => {
    // Drop empty values so we don't send blank query params.
    const query: Record<string, string | number> = {};
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        query[key] = value as string | number;
      }
    });

    const response = await api.get('/api/notifications/messages/search', { params: query });
    return response.data;
  },
};
