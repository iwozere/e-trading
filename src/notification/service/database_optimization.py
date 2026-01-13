"""
Database Query Optimization for Notification Service

This module provides optimized database queries and indexing strategies
for the notification service to improve performance under high load.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import and_, or_, desc, asc, func, text, Index
from sqlalchemy.dialects.postgresql import insert

from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus, RateLimit, ChannelConfig,
    MessagePriority, MessageStatus, DeliveryStatus
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class OptimizedMessageRepository:
    """Optimized repository for message operations with performance improvements."""

    def __init__(self, session: Session):
        """
        Initialize the optimized repository with a database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def create_message(self, message_data: Dict[str, Any]) -> Message:
        """
        Create a new message.

        Args:
            message_data: Dictionary with message data

        Returns:
            Created Message object

        Raises:
            IntegrityError: If message creation fails
        """
        try:
            from sqlalchemy.exc import IntegrityError
            message = Message(**message_data)
            self.session.add(message)
            self.session.flush()  # Get the ID without committing
            _logger.info("Created message %s with type %s", message.id, message.message_type)
            return message
        except IntegrityError:
            self.session.rollback()
            _logger.exception("Failed to create message:")
            raise

    def get_message(self, message_id: int) -> Optional[Message]:
        """
        Get a message by ID.

        Args:
            message_id: Message ID

        Returns:
            Message object or None if not found
        """
        return self.session.query(Message).filter(Message.id == message_id).first()

    def list_messages(
        self,
        status: Optional[MessageStatus] = None,
        priority: Optional[MessagePriority] = None,
        recipient_id: Optional[str] = None,
        message_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> List[Message]:
        """
        List messages with optional filtering.

        Args:
            status: Filter by status
            priority: Filter by priority
            recipient_id: Filter by recipient ID
            message_type: Filter by message type
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Field to order by
            order_desc: Order in descending order

        Returns:
            List of Message objects
        """
        query = self.session.query(Message)

        if status is not None:
            query = query.filter(Message.status == status.value)

        if priority is not None:
            query = query.filter(Message.priority == priority.value)

        if recipient_id is not None:
            query = query.filter(Message.recipient_id == recipient_id)

        if message_type is not None:
            query = query.filter(Message.message_type == message_type)

        # Apply ordering
        order_field = getattr(Message, order_by, Message.created_at)
        if order_desc:
            query = query.order_by(desc(order_field))
        else:
            query = query.order_by(asc(order_field))

        return query.offset(offset).limit(limit).all()

    def get_pending_messages_with_lock(
        self,
        limit: int = 10,
        lock_instance_id: str = None,
        channels: Optional[List[str]] = None
    ) -> List[Message]:
        """
        Get pending messages with distributed locking for database-centric processing.

        This method atomically claims messages for processing by a specific instance,
        preventing multiple notification service instances from processing the same message.

        Args:
            limit: Maximum number of messages to claim
            lock_instance_id: Unique identifier for the processing instance

        Returns:
            List of Message objects claimed for processing
        """
        if not lock_instance_id:
            raise ValueError("lock_instance_id is required for distributed processing")

        from datetime import datetime, timezone, timedelta
        current_time = datetime.now(timezone.utc)
        stale_lock_threshold = current_time - timedelta(minutes=5)  # Locks older than 5 minutes are stale

        try:
            # Build dynamic channel condition for raw SQL
            channel_condition = ""
            if channels:
                # PostgreSQL array overlap operator && or contains @>
                # We want messages where message.channels contains at least one of the provided channels
                # SQL: AND channels && ARRAY['ch1', 'ch2']
                channel_condition = "AND channels && :channels"

            # Use raw SQL for atomic message claiming with PostgreSQL-specific features
            query = text(f"""
                UPDATE msg_messages
                SET locked_by = :lock_instance_id,
                    locked_at = :current_time
                WHERE id IN (
                    SELECT id FROM msg_messages
                    WHERE status = 'PENDING'
                    AND scheduled_for <= :current_time
                    {channel_condition}
                    AND (
                        locked_by IS NULL
                        OR locked_at < :stale_threshold
                    )
                    ORDER BY
                        CASE priority
                            WHEN 'CRITICAL' THEN 1
                            WHEN 'HIGH' THEN 2
                            WHEN 'NORMAL' THEN 3
                            WHEN 'LOW' THEN 4
                        END,
                        scheduled_for ASC
                    LIMIT :limit
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING *
            """)

            params = {
                'lock_instance_id': lock_instance_id,
                'current_time': current_time,
                'stale_threshold': stale_lock_threshold,
                'limit': limit
            }
            if channels:
                params['channels'] = channels

            result = self.session.execute(query, params)

            # Convert result rows to Message objects
            messages = []
            for row in result:
                message = Message()
                for column in row._mapping.keys():
                    setattr(message, column, row._mapping[column])
                messages.append(message)

            if messages:
                _logger.info(
                    "Claimed %s messages for processing by instance %s",
                    len(messages), lock_instance_id
                )

            return messages

        except Exception:
            _logger.exception("Error claiming messages with lock:")
            self.session.rollback()
            return []

    def get_pending_messages(
        self,
        current_time: datetime,
        priority: Optional[MessagePriority] = None,
        channels: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Message]:
        """
        Get pending messages ready for processing.

        This is a compatibility method that delegates to get_pending_messages_optimized.

        Args:
            current_time: Current timestamp
            priority: Filter by priority
            channels: Filter by specific channels
            limit: Maximum number of results

        Returns:
            List of pending Message objects
        """
        return self.get_pending_messages_optimized(current_time, priority, channels, limit)

    def get_pending_messages_optimized(
        self,
        current_time: datetime,
        priority: Optional[MessagePriority] = None,
        channels: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Message]:
        """
        Optimized query for pending messages with proper indexing.

        Uses composite index on (status, scheduled_for, priority) for optimal performance.

        Args:
            current_time: Current timestamp
            priority: Filter by priority
            channels: Filter by specific channels
            limit: Maximum number of results

        Returns:
            List of pending Message objects
        """
        # Use raw SQL for optimal performance with custom priority ordering
        priority_case = text(
            "CASE priority "
            "WHEN 'CRITICAL' THEN 1 "
            "WHEN 'HIGH' THEN 2 "
            "WHEN 'NORMAL' THEN 3 "
            "WHEN 'LOW' THEN 4 "
            "END"
        )

        query = self.session.query(Message).filter(
            and_(
                Message.status == MessageStatus.PENDING.value,
                Message.scheduled_for <= current_time
            )
        )

        if priority is not None:
            query = query.filter(Message.priority == priority.value)

        if channels:
            # Filter messages where ANY of the specified channels are present
            # For PostgreSQL ARRAY, we use overlap (&&) which is more robust
            # than contains (@>) in an OR loop.
            query = query.filter(Message.channels.overlap(channels))

        # Use the optimized ordering with index hint
        return query.order_by(
            priority_case,
            asc(Message.scheduled_for),
            asc(Message.id)  # Tie-breaker for consistent ordering
        ).limit(limit).all()

    def bulk_update_message_status(
        self,
        message_ids: List[int],
        status: MessageStatus,
        processed_at: Optional[datetime] = None
    ) -> int:
        """
        Bulk update message status for better performance.

        Args:
            message_ids: List of message IDs to update
            status: New status
            processed_at: Processing timestamp

        Returns:
            Number of messages updated
        """
        if not message_ids:
            return 0

        update_data = {"status": status.value}
        if processed_at:
            update_data["processed_at"] = processed_at

        updated_count = self.session.query(Message).filter(
            Message.id.in_(message_ids)
        ).update(update_data, synchronize_session=False)

        _logger.info("Bulk updated %s messages to status %s", updated_count, status.value)
        return updated_count

    def get_messages_with_delivery_status(
        self,
        message_ids: List[int]
    ) -> List[Message]:
        """
        Efficiently load messages with their delivery statuses using eager loading.

        Args:
            message_ids: List of message IDs

        Returns:
            List of Message objects with delivery_statuses loaded
        """
        return self.session.query(Message).options(
            selectinload(Message.delivery_statuses)
        ).filter(Message.id.in_(message_ids)).all()

    def get_message_statistics_optimized(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by_hour: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Optimized query for message statistics using aggregation.

        Args:
            start_date: Start date for statistics
            end_date: End date for statistics
            group_by_hour: Group by hour instead of day

        Returns:
            List of statistics dictionaries
        """
        if group_by_hour:
            date_trunc = func.date_trunc('hour', Message.created_at)
        else:
            date_trunc = func.date_trunc('day', Message.created_at)

        # Use a single query with aggregation for better performance
        stats_query = self.session.query(
            date_trunc.label('time_period'),
            Message.status,
            Message.priority,
            func.count(Message.id).label('message_count'),
            func.avg(Message.retry_count).label('avg_retry_count')
        ).filter(
            and_(
                Message.created_at >= start_date,
                Message.created_at <= end_date
            )
        ).group_by(
            date_trunc,
            Message.status,
            Message.priority
        ).order_by(date_trunc)

        results = []
        for row in stats_query.all():
            results.append({
                "time_period": row.time_period.isoformat(),
                "status": row.status,
                "priority": row.priority,
                "message_count": row.message_count,
                "avg_retry_count": float(row.avg_retry_count) if row.avg_retry_count else 0.0
            })

        return results

    def update_message(self, message_id: int, update_data: Dict[str, Any]) -> Optional[Message]:
        """
        Update a message.

        Args:
            message_id: Message ID
            update_data: Dictionary with fields to update

        Returns:
            Updated Message object or None if not found
        """
        message = self.get_message(message_id)
        if not message:
            return None

        try:
            for key, value in update_data.items():
                if hasattr(message, key):
                    setattr(message, key, value)

            self.session.flush()
            return message
        except Exception:
            self.session.rollback()
            raise

    def get_message(self, message_id: int) -> Optional[Message]:
        """
        Get a message by ID.

        Args:
            message_id: Message ID

        Returns:
            Message object or None if not found
        """
        return self.session.query(Message).filter(Message.id == message_id).first()

    def get_failed_messages_for_retry(
        self,
        current_time: datetime,
        retry_delay_minutes: int = 5,
        limit: int = 50
    ) -> List[Message]:
        """
        Get failed messages that can be retried.

        Args:
            current_time: Current timestamp
            retry_delay_minutes: Minimum delay before retry
            limit: Maximum number of results

        Returns:
            List of failed Message objects ready for retry
        """
        from datetime import timedelta
        retry_cutoff = current_time - timedelta(minutes=retry_delay_minutes)

        return self.session.query(Message).filter(
            and_(
                Message.status == MessageStatus.FAILED.value,
                Message.retry_count < Message.max_retries,
                or_(
                    Message.processed_at.is_(None),
                    Message.processed_at <= retry_cutoff
                )
            )
        ).order_by(asc(Message.processed_at)).limit(limit).all()


class OptimizedDeliveryStatusRepository:
    """Optimized repository for delivery status operations."""

    def __init__(self, session: Session):
        """
        Initialize the optimized repository with a database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def bulk_create_delivery_statuses(
        self,
        delivery_data: List[Dict[str, Any]]
    ) -> List[MessageDeliveryStatus]:
        """
        Bulk create delivery statuses for better performance.

        Args:
            delivery_data: List of delivery status data dictionaries

        Returns:
            List of created MessageDeliveryStatus objects
        """
        if not delivery_data:
            return []

        # Use PostgreSQL's INSERT ... RETURNING for efficiency
        stmt = insert(MessageDeliveryStatus).values(delivery_data)
        stmt = stmt.returning(MessageDeliveryStatus)

        result = self.session.execute(stmt)
        delivery_statuses = result.fetchall()

        _logger.info("Bulk created %s delivery statuses", len(delivery_statuses))
        return delivery_statuses

    def get_delivery_history_optimized(
        self,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
        status: Optional[DeliveryStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[MessageDeliveryStatus], int]:
        """
        Optimized delivery history query with proper indexing and joins.

        Args:
            user_id: Filter by user ID (requires join with messages)
            channel: Filter by channel
            status: Filter by delivery status
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Tuple of (delivery statuses, total count)
        """
        # Build base query with conditional join
        if user_id:
            # Use explicit join for better query planning
            query = self.session.query(MessageDeliveryStatus).join(
                Message,
                MessageDeliveryStatus.message_id == Message.id
            ).filter(Message.recipient_id == user_id)
        else:
            query = self.session.query(MessageDeliveryStatus)

        # Apply filters
        if channel:
            query = query.filter(MessageDeliveryStatus.channel == channel)

        if status:
            query = query.filter(MessageDeliveryStatus.status == status.value)

        if start_date:
            query = query.filter(MessageDeliveryStatus.created_at >= start_date)

        if end_date:
            query = query.filter(MessageDeliveryStatus.created_at <= end_date)

        # Get total count efficiently
        total_count = query.count()

        # Apply ordering and pagination
        deliveries = query.order_by(
            desc(MessageDeliveryStatus.created_at),
            desc(MessageDeliveryStatus.id)  # Tie-breaker
        ).offset(offset).limit(limit).all()

        return deliveries, total_count

    def get_channel_performance_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        channels: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimized query for channel performance metrics.

        Args:
            start_date: Start date for metrics
            end_date: End date for metrics
            channels: Optional list of channels to filter

        Returns:
            Dictionary with channel performance data
        """
        # Single aggregation query for all metrics
        query = self.session.query(
            MessageDeliveryStatus.channel,
            func.count(MessageDeliveryStatus.id).label('total_attempts'),
            func.sum(
                func.case(
                    [(MessageDeliveryStatus.status == DeliveryStatus.DELIVERED.value, 1)],
                    else_=0
                )
            ).label('successful_deliveries'),
            func.avg(
                func.case(
                    [(MessageDeliveryStatus.status == DeliveryStatus.DELIVERED.value,
                      MessageDeliveryStatus.response_time_ms)],
                    else_=None
                )
            ).label('avg_response_time'),
            func.percentile_cont(0.5).within_group(
                MessageDeliveryStatus.response_time_ms
            ).label('median_response_time'),
            func.percentile_cont(0.95).within_group(
                MessageDeliveryStatus.response_time_ms
            ).label('p95_response_time')
        ).filter(
            and_(
                MessageDeliveryStatus.created_at >= start_date,
                MessageDeliveryStatus.created_at <= end_date,
                MessageDeliveryStatus.response_time_ms.isnot(None)
            )
        ).group_by(MessageDeliveryStatus.channel)

        if channels:
            query = query.filter(MessageDeliveryStatus.channel.in_(channels))

        results = {}
        for row in query.all():
            success_rate = (row.successful_deliveries / row.total_attempts
                          if row.total_attempts > 0 else 0.0)

            results[row.channel] = {
                "total_attempts": row.total_attempts,
                "successful_deliveries": row.successful_deliveries,
                "failed_deliveries": row.total_attempts - row.successful_deliveries,
                "success_rate": success_rate,
                "avg_response_time_ms": float(row.avg_response_time) if row.avg_response_time else None,
                "median_response_time_ms": float(row.median_response_time) if row.median_response_time else None,
                "p95_response_time_ms": float(row.p95_response_time) if row.p95_response_time else None
            }

        return results


class OptimizedRateLimitRepository:
    """Optimized repository for rate limit operations."""

    def __init__(self, session: Session):
        """
        Initialize the optimized repository with a database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def bulk_refill_tokens(
        self,
        current_time: datetime,
        batch_size: int = 1000
    ) -> int:
        """
        Bulk refill tokens for all rate limits that need updating.

        Args:
            current_time: Current timestamp
            batch_size: Number of records to process at once

        Returns:
            Number of rate limits updated
        """
        # Find rate limits that need token refill (older than 1 minute)
        refill_cutoff = current_time - timedelta(minutes=1)

        # Use raw SQL for optimal performance
        update_sql = text("""
            UPDATE msg_rate_limits
            SET tokens = LEAST(max_tokens, tokens + FLOOR(EXTRACT(EPOCH FROM (:current_time - last_refill)) / 60.0) * refill_rate),
                last_refill = :current_time
            WHERE last_refill < :refill_cutoff
        """)

        result = self.session.execute(update_sql, {
            'current_time': current_time,
            'refill_cutoff': refill_cutoff
        })

        updated_count = result.rowcount
        _logger.info("Bulk refilled tokens for %s rate limits", updated_count)
        return updated_count

    def check_rate_limits_batch(
        self,
        user_channel_pairs: List[Tuple[str, str]],
        current_time: datetime
    ) -> Dict[Tuple[str, str], bool]:
        """
        Check rate limits for multiple user-channel pairs efficiently.

        Args:
            user_channel_pairs: List of (user_id, channel) tuples
            current_time: Current timestamp

        Returns:
            Dictionary mapping (user_id, channel) to availability
        """
        if not user_channel_pairs:
            return {}

        # Build conditions for all pairs
        conditions = []
        for user_id, channel in user_channel_pairs:
            conditions.append(
                and_(RateLimit.user_id == user_id, RateLimit.channel == channel)
            )

        # Single query to get all relevant rate limits
        rate_limits = self.session.query(RateLimit).filter(
            or_(*conditions)
        ).all()

        # Create lookup dictionary
        rate_limit_lookup = {
            (rl.user_id, rl.channel): rl for rl in rate_limits
        }

        results = {}
        for user_id, channel in user_channel_pairs:
            rate_limit = rate_limit_lookup.get((user_id, channel))
            if rate_limit:
                # Refill tokens if needed
                rate_limit.refill_tokens(current_time)
                results[(user_id, channel)] = rate_limit.is_available
            else:
                # No rate limit exists, assume available
                results[(user_id, channel)] = True

        return results


def create_optimized_indexes(engine):
    """
    Create additional database indexes for optimal query performance.

    Args:
        engine: SQLAlchemy engine
    """
    _logger.info("Creating optimized database indexes...")

    # Composite indexes for common query patterns
    indexes_to_create = [
        # Messages table optimizations
        Index(
            'idx_msg_messages_status_scheduled_priority',
            Message.status, Message.scheduled_for, Message.priority,
            postgresql_where=Message.status == 'PENDING'
        ),
        Index(
            'idx_msg_messages_recipient_created',
            Message.recipient_id, Message.created_at
        ),
        Index(
            'idx_msg_messages_type_status',
            Message.message_type, Message.status
        ),
        Index(
            'idx_msg_messages_retry_status',
            Message.retry_count, Message.max_retries, Message.status,
            postgresql_where=Message.status == 'FAILED'
        ),

        # Delivery status table optimizations
        Index(
            'idx_msg_delivery_status_channel_created',
            MessageDeliveryStatus.channel, MessageDeliveryStatus.created_at
        ),
        Index(
            'idx_msg_delivery_status_status_delivered',
            MessageDeliveryStatus.status, MessageDeliveryStatus.delivered_at
        ),
        Index(
            'idx_msg_delivery_status_message_channel',
            MessageDeliveryStatus.message_id, MessageDeliveryStatus.channel
        ),

        # Rate limits table optimizations
        Index(
            'idx_msg_rate_limits_refill_time',
            RateLimit.last_refill,
            postgresql_where=RateLimit.tokens < RateLimit.max_tokens
        ),

        # Channel health table optimizations
        Index(
            'idx_msg_channel_health_status_checked',
            ChannelHealth.status, ChannelHealth.checked_at
        ),

        # Channel configs table optimizations
        Index(
            'idx_msg_channel_configs_enabled_updated',
            ChannelConfig.enabled, ChannelConfig.updated_at
        )
    ]

    # Create indexes that don't already exist
    with engine.connect() as conn:
        for index in indexes_to_create:
            try:
                index.create(conn, checkfirst=True)
                _logger.info("Created index: %s", index.name)
            except Exception as e:
                _logger.warning("Failed to create index %s: %s", index.name, e)

    _logger.info("Finished creating optimized database indexes")


def analyze_query_performance(session: Session) -> Dict[str, Any]:
    """
    Analyze query performance and provide optimization recommendations.

    Args:
        session: SQLAlchemy database session

    Returns:
        Dictionary with performance analysis
    """
    _logger.info("Analyzing query performance...")

    analysis = {
        "table_sizes": {},
        "index_usage": {},
        "slow_queries": [],
        "recommendations": []
    }

    try:
        # Get table sizes
        table_size_query = text("""
            SELECT
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
            FROM pg_tables
            WHERE tablename LIKE 'msg_%'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """)

        result = session.execute(table_size_query)
        for row in result:
            analysis["table_sizes"][row.tablename] = {
                "size": row.size,
                "size_bytes": row.size_bytes
            }

        # Get index usage statistics
        index_usage_query = text("""
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE tablename LIKE 'msg_%'
            ORDER BY idx_scan DESC
        """)

        result = session.execute(index_usage_query)
        for row in result:
            table_key = f"{row.schemaname}.{row.tablename}"
            if table_key not in analysis["index_usage"]:
                analysis["index_usage"][table_key] = []

            analysis["index_usage"][table_key].append({
                "index_name": row.indexname,
                "scans": row.idx_scan,
                "tuples_read": row.idx_tup_read,
                "tuples_fetched": row.idx_tup_fetch
            })

        # Generate recommendations based on analysis
        recommendations = []

        # Check for large tables without proper indexing
        for table_name, table_info in analysis["table_sizes"].items():
            if table_info["size_bytes"] > 100 * 1024 * 1024:  # > 100MB
                recommendations.append(
                    f"Table {table_name} is large ({table_info['size']}). "
                    "Consider partitioning or archiving old data."
                )

        # Check for unused indexes
        for table_key, indexes in analysis["index_usage"].items():
            for index_info in indexes:
                if index_info["scans"] == 0:
                    recommendations.append(
                        f"Index {index_info['index_name']} on {table_key} is unused. "
                        "Consider dropping it to save space."
                    )

        analysis["recommendations"] = recommendations

    except Exception as e:
        _logger.exception("Failed to analyze query performance:")
        analysis["error"] = str(e)

    return analysis


def optimize_database_settings(engine):
    """
    Apply database-level optimizations for PostgreSQL.

    Args:
        engine: SQLAlchemy engine
    """
    _logger.info("Applying database optimizations...")

    optimizations = [
        # Increase work_mem for complex queries
        "SET work_mem = '256MB'",

        # Optimize for read-heavy workload
        "SET random_page_cost = 1.1",

        # Increase effective_cache_size
        "SET effective_cache_size = '4GB'",

        # Enable parallel query execution
        "SET max_parallel_workers_per_gather = 4",

        # Optimize checkpoint settings
        "SET checkpoint_completion_target = 0.9",

        # Increase shared_buffers if not already set
        "SET shared_buffers = '1GB'"
    ]

    with engine.connect() as conn:
        for optimization in optimizations:
            try:
                conn.execute(text(optimization))
                _logger.info("Applied optimization: %s", optimization)
            except Exception as e:
                _logger.warning("Failed to apply optimization '%s': %s", optimization, e)

    _logger.info("Finished applying database optimizations")