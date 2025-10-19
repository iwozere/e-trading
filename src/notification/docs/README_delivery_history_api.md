# Delivery History API

This document describes the delivery history API endpoints implemented as part of task 5.3.

## Overview

The delivery history API provides comprehensive endpoints for querying message and delivery history with advanced filtering, pagination, and export capabilities. This enables analytics, troubleshooting, and audit trail functionality for the notification service.

## Endpoints

### 1. Message History

**GET** `/api/v1/history/messages`

Query message history with filtering and pagination.

#### Parameters

- `user_id` (optional): Filter by recipient user ID
- `channel` (optional): Filter by specific channel
- `status` (optional): Filter by message status (PENDING, PROCESSING, DELIVERED, FAILED, CANCELLED)
- `message_type` (optional): Filter by message type
- `start_date` (optional): Filter messages created after this date (ISO format)
- `end_date` (optional): Filter messages created before this date (ISO format)
- `limit` (optional): Maximum number of results (1-1000, default: 100)
- `offset` (optional): Number of results to skip (default: 0)
- `order_by` (optional): Field to order by (created_at, scheduled_for, processed_at, default: created_at)
- `order_desc` (optional): Order in descending order (default: true)

#### Response

```json
{
  "messages": [
    {
      "id": 123,
      "message_type": "trade_alert",
      "priority": "HIGH",
      "channels": ["telegram", "email"],
      "recipient_id": "user123",
      "template_name": "alert_template",
      "content": {...},
      "message_metadata": {...},
      "created_at": "2023-10-19T10:30:00Z",
      "scheduled_for": "2023-10-19T10:30:00Z",
      "status": "DELIVERED",
      "retry_count": 0,
      "max_retries": 3,
      "last_error": null,
      "processed_at": "2023-10-19T10:30:05Z"
    }
  ],
  "pagination": {
    "total": 150,
    "limit": 100,
    "offset": 0,
    "has_more": true
  },
  "filters": {
    "user_id": "user123",
    "channel": "telegram",
    "status": "DELIVERED",
    "message_type": "trade_alert",
    "start_date": "2023-10-01T00:00:00Z",
    "end_date": "2023-10-31T23:59:59Z"
  }
}
```

### 2. Delivery History

**GET** `/api/v1/history/deliveries`

Query delivery status history with filtering and pagination.

#### Parameters

- `message_id` (optional): Filter by specific message ID
- `user_id` (optional): Filter by recipient user ID (requires join with messages)
- `channel` (optional): Filter by specific channel
- `status` (optional): Filter by delivery status (PENDING, SENT, DELIVERED, FAILED, BOUNCED)
- `start_date` (optional): Filter deliveries created after this date (ISO format)
- `end_date` (optional): Filter deliveries created before this date (ISO format)
- `limit` (optional): Maximum number of results (1-1000, default: 100)
- `offset` (optional): Number of results to skip (default: 0)
- `order_by` (optional): Field to order by (created_at, delivered_at, default: created_at)
- `order_desc` (optional): Order in descending order (default: true)

#### Response

```json
{
  "deliveries": [
    {
      "id": 456,
      "message_id": 123,
      "channel": "telegram",
      "status": "DELIVERED",
      "delivered_at": "2023-10-19T10:30:05Z",
      "response_time_ms": 150,
      "error_message": null,
      "external_id": "tg_msg_789",
      "created_at": "2023-10-19T10:30:00Z"
    }
  ],
  "pagination": {
    "total": 75,
    "limit": 100,
    "offset": 0,
    "has_more": false
  },
  "filters": {
    "message_id": null,
    "user_id": "user123",
    "channel": "telegram",
    "status": "DELIVERED",
    "start_date": "2023-10-01T00:00:00Z",
    "end_date": "2023-10-31T23:59:59Z"
  }
}
```

### 3. Export History

**GET** `/api/v1/history/export`

Export message and delivery history for analytics.

#### Parameters

- `format` (required): Export format ("json" or "csv")
- `user_id` (optional): Filter by recipient user ID
- `channel` (optional): Filter by specific channel
- `status` (optional): Filter by message status
- `message_type` (optional): Filter by message type
- `start_date` (optional): Filter messages created after this date (ISO format)
- `end_date` (optional): Filter messages created before this date (ISO format)
- `limit` (optional): Maximum number of records to export (1-50000, default: 10000)

#### Response (Small Export - Immediate)

For exports with limit ≤ 1000, data is returned immediately:

```json
{
  "export_type": "immediate",
  "format": "json",
  "record_count": 25,
  "data": [...],  // Array of message objects with nested deliveries
  "filters": {
    "user_id": "user123",
    "channel": "telegram",
    "status": "DELIVERED",
    "message_type": "trade_alert",
    "start_date": "2023-10-01T00:00:00Z",
    "end_date": "2023-10-31T23:59:59Z"
  }
}
```

#### Response (Large Export - Background)

For exports with limit > 1000, processing happens in background:

```json
{
  "export_type": "background",
  "export_id": "export_20231019_103000_user123",
  "status": "processing",
  "message": "Large export started in background. Check status with export_id.",
  "estimated_completion": "5-10 minutes"
}
```

#### CSV Format

When format is "csv", the data field contains CSV text with the following columns:

- message_id, message_type, priority, channels, recipient_id, template_name
- created_at, scheduled_for, status, retry_count
- delivery_id, delivery_channel, delivery_status, delivered_at
- response_time_ms, error_message, external_id

### 4. History Summary

**GET** `/api/v1/history/summary`

Get summary statistics for message and delivery history.

#### Parameters

- `user_id` (optional): Filter by recipient user ID
- `channel` (optional): Filter by specific channel
- `days` (optional): Number of days to analyze (1-365, default: 30)

#### Response

```json
{
  "period": {
    "days": 30,
    "start_date": "2023-09-19T10:30:00Z",
    "end_date": "2023-10-19T10:30:00Z"
  },
  "filters": {
    "user_id": "user123",
    "channel": null
  },
  "message_statistics": {
    "total": 150,
    "by_status": {
      "PENDING": 5,
      "PROCESSING": 2,
      "DELIVERED": 140,
      "FAILED": 3,
      "CANCELLED": 0
    },
    "success_rate": 0.933
  },
  "delivery_statistics": {
    "total": 300,
    "by_status": {
      "PENDING": 8,
      "SENT": 5,
      "DELIVERED": 280,
      "FAILED": 6,
      "BOUNCED": 1
    },
    "success_rate": 0.933
  },
  "channel_breakdown": {
    "telegram": {
      "total_deliveries": 150,
      "successful_deliveries": 145,
      "success_rate": 0.967
    },
    "email": {
      "total_deliveries": 150,
      "successful_deliveries": 135,
      "success_rate": 0.900
    }
  }
}
```

## Features

### Filtering

All endpoints support comprehensive filtering:

- **User-based**: Filter by recipient user ID
- **Channel-based**: Filter by specific notification channel
- **Status-based**: Filter by message or delivery status
- **Time-based**: Filter by date ranges
- **Type-based**: Filter by message type

### Pagination

- Configurable page size (limit) with reasonable maximums
- Offset-based pagination for large result sets
- Total count and "has_more" indicators
- Consistent pagination metadata across endpoints

### Sorting

- Configurable sort field and direction
- Support for relevant timestamp fields
- Consistent ordering behavior

### Export Capabilities

- **Multiple formats**: JSON and CSV export
- **Immediate export**: For small datasets (≤1000 records)
- **Background processing**: For large datasets (>1000 records)
- **Comprehensive data**: Includes both message and delivery data
- **Filtering support**: All filters apply to exports

### Performance Considerations

- **Query optimization**: Efficient database queries with proper indexing
- **Result limits**: Reasonable limits to prevent resource exhaustion
- **Background processing**: Large operations don't block API responses
- **SQLite compatibility**: Works with both SQLite and PostgreSQL

## Error Handling

All endpoints return appropriate HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **404**: Not Found (for specific resource queries)
- **500**: Internal Server Error

Error responses include descriptive messages:

```json
{
  "error": "Bad Request",
  "detail": "Limit must be between 1 and 1000"
}
```

## Usage Examples

### Get recent messages for a user

```bash
curl "http://localhost:8080/api/v1/history/messages?user_id=user123&limit=50&order_desc=true"
```

### Get failed deliveries for troubleshooting

```bash
curl "http://localhost:8080/api/v1/history/deliveries?status=FAILED&start_date=2023-10-01T00:00:00Z"
```

### Export telegram messages as CSV

```bash
curl "http://localhost:8080/api/v1/history/export?format=csv&channel=telegram&limit=500"
```

### Get weekly summary statistics

```bash
curl "http://localhost:8080/api/v1/history/summary?days=7"
```

## Integration

These endpoints integrate with:

- **Analytics systems**: For performance monitoring and reporting
- **Troubleshooting tools**: For debugging delivery issues
- **Audit systems**: For compliance and tracking
- **Dashboard applications**: For real-time monitoring

## Security

- All endpoints respect user access controls
- Sensitive data is properly filtered
- Rate limiting applies to prevent abuse
- Input validation prevents injection attacks

## Future Enhancements

Potential future improvements:

- **Real-time exports**: WebSocket-based streaming exports
- **Advanced analytics**: Built-in trend analysis and anomaly detection
- **Custom filters**: User-defined filter combinations
- **Scheduled exports**: Automated periodic exports
- **Data retention**: Automated archival and cleanup integration