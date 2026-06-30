import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { notificationsApi, MessageSearchParams } from '../../api/notificationsApi';

export const useMessages = (params: MessageSearchParams) => {
  return useQuery({
    queryKey: ['notifications', 'messages', params],
    queryFn: () => notificationsApi.searchMessages(params),
    placeholderData: keepPreviousData,
    staleTime: 30000,
  });
};
