"""
Notification Service Repository

Repository layer for notification service operations.
Provides data access methods for messages, delivery status, channel health, rate limits, and channel configs.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, text
from sqlalchemy.exc import IntegrityError

from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus, RateLimit, ChannelConfig,
    MessagePriority, MessageStatus, DeliveryStatus
)
from src.notification.logger import setup_logger
from src.notification.service.database_optimization import (
    OptimizedMessageRepository,
    OptimizedDeliveryStatusRepository,
    OptimizedRateLimitRepository
)

_logger = setup_logger(__name__)


class MessageRepository:
    """Repository for message operations."""

    def __init__(self, session: Session):
        """
        Initialize the repository with a database session.

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
            _logger.info("Updated message %s", message.id)
            return message
        except Exception as e:
            self.session.rollback()
            _logger.error("Failed to update message %s: %s", message_id, e)
            raise

    def get_pending_messages(
        self,
        current_time: datetime,
        priority: Optional[MessagePriority] = None,
        limit: int = 100
    ) -> List[Message]:
        """
        Get pending messages that are ready for processing.

        Args:
            current_time: Current timestamp
            priority: Filter by priority
            limit: Maximum number of results

        Returns:
            List of pending Message objects
        """
        query = self.session.query(Message).filter(
            and_(
                Message.status == MessageStatus.PENDING.value,
                Message.scheduled_for <= current_time
            )
        )

        if priority is not None:
            query = query.filter(Message.priority == priority.value)

        # Order by priority (CRITICAL first) then by scheduled_for
        priority_order = text(
            "CASE priority "
            "WHEN 'CRITICAL' THEN 1 "
            "WHEN 'HIGH' THEN 2 "
            "WHEN 'NORMAL' THEN 3 "
            "WHEN 'LOW' THEN 4 "
            "END"
        )

        return query.order_by(priority_order, asc(Message.scheduled_for)).limit(limit).all()

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

    def cleanup_old_messages(self, days_to_keep: int = 30) -> int:
        """
        Clean up old delivered messages.

        Args:
            days_to_keep: Number of days of messages to keep

        Returns:
            Number of messages deleted
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

        deleted_count = self.session.query(Message).filter(
            and_(
                Message.created_at < cutoff_date,
                Message.status == MessageStatus.DELIVERED.value
            )
        ).delete()

        _logger.info("Cleaned up %s old messages", deleted_count)
        return deleted_count

    def get_queue_health_for_channels(self, channels: List[str]) -> Dict[str, Any]:
        """
        Get queue health metrics for specific channels.

        Args:
            channels: List of channel names to monitor (e.g., ["telegram"], ["email", "sms"])

        Returns:
            Dictionary with queue health metrics:
            - pending: Count of pending messages
            - processing: Count of messages currently processing
            - failed_last_hour: Count of failures in last hour
            - delivered_last_hour: Count of deliveries in last hour
            - stuck_messages: Count of messages stuck in processing
        """
        from sqlalchemy import func

        current_time = datetime.now(timezone.utc)
        one_hour_ago = current_time - timedelta(hours=1)
        five_min_ago = current_time - timedelta(minutes=5)

        # Build channel filters (messages where ANY of the specified channels are present)
        channel_filters = [Message.channels.contains([ch]) for ch in channels]

        if not channel_filters:
            # No channels specified, return empty metrics
            return {
                "pending": 0,
                "processing": 0,
                "failed_last_hour": 0,
                "delivered_last_hour": 0,
                "stuck_messages": 0
            }

        # Pending count
        pending_count = self.session.query(func.count(Message.id)).filter(
            Message.status == MessageStatus.PENDING.value,
            or_(*channel_filters)
        ).scalar() or 0

        # Processing count
        processing_count = self.session.query(func.count(Message.id)).filter(
            Message.status == MessageStatus.PROCESSING.value,
            or_(*channel_filters)
        ).scalar() or 0

        # Failed in last hour
        failed_last_hour = self.session.query(func.count(Message.id)).filter(
            Message.status == MessageStatus.FAILED.value,
            Message.updated_at >= one_hour_ago,
            or_(*channel_filters)
        ).scalar() or 0

        # Delivered in last hour
        delivered_last_hour = self.session.query(func.count(Message.id)).filter(
            Message.status == MessageStatus.DELIVERED.value,
            Message.delivered_at >= one_hour_ago,
            or_(*channel_filters)
        ).scalar() or 0

        # Stuck messages (processing for > 5 minutes)
        stuck_messages = self.session.query(func.count(Message.id)).filter(
            Message.status == MessageStatus.PROCESSING.value,
            Message.updated_at < five_min_ago,
            or_(*channel_filters)
        ).scalar() or 0

        return {
            "pending": pending_count,
            "processing": processing_count,
            "failed_last_hour": failed_last_hour,
            "delivered_last_hour": delivered_last_hour,
            "stuck_messages": stuck_messages
        }

    def get_pending_messages_with_lock(
        self,
        limit: int = 10,
        lock_instance_id: str = None
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

        current_time = datetime.now(timezone.utc)
        stale_lock_threshold = current_time - timedelta(minutes=5)  # Locks older than 5 minutes are stale

        try:
            # Use raw SQL for atomic message claiming with PostgreSQL-specific features
            query = text("""
                UPDATE msg_messages
                SET locked_by = :lock_instance_id,
                    locked_at = :current_time
                WHERE id IN (
                    SELECT id FROM msg_messages
                    WHERE status = 'PENDING'
                    AND scheduled_for <= :current_time
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

            result = self.session.execute(query, {
                'lock_instance_id': lock_instance_id,
                'current_time': current_time,
                'stale_threshold': stale_lock_threshold,
                'limit': limit
            })

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

    def release_message_lock(self, message_id: int, lock_instance_id: str) -> bool:
        """
        Release the lock on a message after processing.

        Args:
            message_id: Message ID to unlock
            lock_instance_id: Instance ID that owns the lock

        Returns:
            True if lock was released, False otherwise
        """
        try:
            updated_rows = self.session.query(Message).filter(
                and_(
                    Message.id == message_id,
                    Message.locked_by == lock_instance_id
                )
            ).update({
                'locked_by': None,
                'locked_at': None
            })

            if updated_rows > 0:
                _logger.debug("Released lock on message %s", message_id)
                return True
            else:
                _logger.warning("Could not release lock on message %s (not owned by %s)", message_id, lock_instance_id)
                return False

        except Exception as e:
            _logger.error("Error releasing lock on message %s: %s", message_id, e)
            return False

    def cleanup_stale_locks(self, stale_threshold_minutes: int = 5) -> int:
        """
        Clean up stale message locks from failed or crashed instances.

        Args:
            stale_threshold_minutes: Minutes after which a lock is considered stale

        Returns:
            Number of stale locks cleaned up
        """
        try:
            stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=stale_threshold_minutes)

            updated_rows = self.session.query(Message).filter(
                and_(
                    Message.locked_by.isnot(None),
                    Message.locked_at < stale_threshold
                )
            ).update({
                'locked_by': None,
                'locked_at': None
            })

            if updated_rows > 0:
                _logger.info("Cleaned up %s stale message locks", updated_rows)

            return updated_rows

        except Exception:
            _logger.exception("Error cleaning up stale locks:")
            return 0


class DeliveryStatusRepository:
    """Repository for delivery status operations."""

    def __init__(self, session: Session):
        """
        Initialize the repository with a database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def create_delivery_status(self, status_data: Dict[str, Any]) -> MessageDeliveryStatus:
        """
        Create a new delivery status.

        Args:
            status_data: Dictionary with delivery status data

        Returns:
            Created MessageDeliveryStatus object
        """
        try:
            delivery_status = MessageDeliveryStatus(**status_data)
            self.session.add(delivery_status)
            self.session.flush()
            _logger.info("Created delivery status %s for message %s", delivery_status.id, delivery_status.message_id)
            return delivery_status
        except IntegrityError:
            self.session.rollback()
            _logger.exception("Failed to create delivery status:")
            raise

    def get_delivery_status(self, status_id: int) -> Optional[MessageDeliveryStatus]:
        """
        Get a delivery status by ID.

        Args:
            status_id: Delivery status ID

        Returns:
            MessageDeliveryStatus object or None if not found
        """
        return self.session.query(MessageDeliveryStatus).filter(MessageDeliveryStatus.id == status_id).first()

    def get_delivery_statuses_by_message(self, message_id: int) -> List[MessageDeliveryStatus]:
        """
        Get all delivery statuses for a message.

        Args:
            message_id: Message ID

        Returns:
            List of MessageDeliveryStatus objects
        """
        return self.session.query(MessageDeliveryStatus).filter(
            MessageDeliveryStatus.message_id == message_id
        ).order_by(desc(MessageDeliveryStatus.created_at)).all()

    def update_delivery_status(self, status_id: int, update_data: Dict[str, Any]) -> Optional[MessageDeliveryStatus]:
        """
        Update a delivery status.

        Args:
            status_id: Delivery status ID
            update_data: Dictionary with fields to update

        Returns:
            Updated MessageDeliveryStatus object or None if not found
        """
        delivery_status = self.get_delivery_status(status_id)
        if not delivery_status:
            return None

        try:
            for key, value in update_data.items():
                if hasattr(delivery_status, key):
                    setattr(delivery_status, key, value)

            self.session.flush()
            _logger.info("Updated delivery status %s", delivery_status.id)
            return delivery_status
        except Exception as e:
            self.session.rollback()
            _logger.error("Failed to update delivery status %s: %s", status_id, e)
            raise

    def get_delivery_statistics(
        self,
        channel: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get delivery statistics for a time period.

        Args:
            channel: Filter by channel
            days: Number of days to look back

        Returns:
            Dictionary with statistics
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        query = self.session.query(MessageDeliveryStatus).filter(
            MessageDeliveryStatus.created_at >= cutoff_date
        )

        if channel is not None:
            query = query.filter(MessageDeliveryStatus.channel == channel)

        # Get counts by status
        status_counts = {}
        for status in DeliveryStatus:
            count = query.filter(MessageDeliveryStatus.status == status.value).count()
            status_counts[status.value] = count

        # Get total count
        total_count = query.count()

        # Get average response time for delivered messages
        delivered_statuses = query.filter(
            and_(
                MessageDeliveryStatus.status == DeliveryStatus.DELIVERED.value,
                MessageDeliveryStatus.response_time_ms.isnot(None)
            )
        ).all()

        avg_response_time = None
        if delivered_statuses:
            response_times = [ds.response_time_ms for ds in delivered_statuses if ds.response_time_ms is not None]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)

        return {
            "total_deliveries": total_count,
            "status_counts": status_counts,
            "average_response_time_ms": avg_response_time,
            "period_days": days,
            "channel": channel
        }

    def get_channel_delivery_rates(
        self,
        cutoff_date: datetime,
        channel: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get delivery rates grouped by channel.

        Args:
            cutoff_date: Only include deliveries after this date
            channel: Filter by specific channel

        Returns:
            Dictionary with channel delivery rates
        """
        query = self.session.query(MessageDeliveryStatus).filter(
            MessageDeliveryStatus.created_at >= cutoff_date
        )

        if channel:
            query = query.filter(MessageDeliveryStatus.channel == channel)

        # Group by channel and calculate rates
        from sqlalchemy import func

        channel_stats = self.session.query(
            MessageDeliveryStatus.channel,
            func.count(MessageDeliveryStatus.id).label('total_attempts'),
            func.sum(
                func.case(
                    [(MessageDeliveryStatus.status == DeliveryStatus.DELIVERED.value, 1)],
                    else_=0
                )
            ).label('successful_attempts'),
            func.avg(
                func.case(
                    [(MessageDeliveryStatus.status == DeliveryStatus.DELIVERED.value,
                      MessageDeliveryStatus.response_time_ms)],
                    else_=None
                )
            ).label('avg_response_time')
        ).filter(
            MessageDeliveryStatus.created_at >= cutoff_date
        ).group_by(MessageDeliveryStatus.channel)

        if channel:
            channel_stats = channel_stats.filter(MessageDeliveryStatus.channel == channel)

        results = {}
        for row in channel_stats.all():
            success_rate = (row.successful_attempts / row.total_attempts
                          if row.total_attempts > 0 else 0.0)

            results[row.channel] = {
                "total_attempts": row.total_attempts,
                "successful_attempts": row.successful_attempts,
                "failed_attempts": row.total_attempts - row.successful_attempts,
                "success_rate": success_rate,
                "avg_response_time_ms": float(row.avg_response_time) if row.avg_response_time else None
            }

        return results

    def get_user_delivery_rates(
        self,
        user_id: str,
        cutoff_date: datetime
    ) -> Dict[str, Any]:
        """
        Get delivery rates for a specific user.

        Args:
            user_id: User ID to analyze
            cutoff_date: Only include deliveries after this date

        Returns:
            Dictionary with user delivery statistics
        """
        # This would require joining with messages table to get user_id
        # For now, return placeholder implementation
        return {
            "user_id": user_id,
            "total_messages": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "success_rate": 0.0,
            "avg_response_time_ms": None
        }

    def get_response_time_data(
        self,
        cutoff_date: datetime,
        channel: Optional[str] = None
    ) -> List[int]:
        """
        Get response time data for analysis.

        Args:
            cutoff_date: Only include deliveries after this date
            channel: Filter by specific channel

        Returns:
            List of response times in milliseconds
        """
        query = self.session.query(MessageDeliveryStatus.response_time_ms).filter(
            and_(
                MessageDeliveryStatus.created_at >= cutoff_date,
                MessageDeliveryStatus.status == DeliveryStatus.DELIVERED.value,
                MessageDeliveryStatus.response_time_ms.isnot(None)
            )
        )

        if channel:
            query = query.filter(MessageDeliveryStatus.channel == channel)

        response_times = [row.response_time_ms for row in query.all()]
        return response_times

    def get_time_series_data(
        self,
        cutoff_date: datetime,
        granularity: str = "daily",
        channel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get time series data for trend analysis.

        Args:
            cutoff_date: Only include deliveries after this date
            granularity: Time granularity (hourly, daily, weekly, monthly)
            channel: Filter by specific channel

        Returns:
            List of time series data points
        """
        from sqlalchemy import func

        # Determine date truncation based on granularity
        if granularity == "hourly":
            date_trunc = func.date_trunc('hour', MessageDeliveryStatus.created_at)
        elif granularity == "daily":
            date_trunc = func.date_trunc('day', MessageDeliveryStatus.created_at)
        elif granularity == "weekly":
            date_trunc = func.date_trunc('week', MessageDeliveryStatus.created_at)
        else:  # monthly
            date_trunc = func.date_trunc('month', MessageDeliveryStatus.created_at)

        query = self.session.query(
            date_trunc.label('time_period'),
            func.count(MessageDeliveryStatus.id).label('total_attempts'),
            func.sum(
                func.case(
                    [(MessageDeliveryStatus.status == DeliveryStatus.DELIVERED.value, 1)],
                    else_=0
                )
            ).label('successful_attempts'),
            func.avg(
                func.case(
                    [(MessageDeliveryStatus.status == DeliveryStatus.DELIVERED.value,
                      MessageDeliveryStatus.response_time_ms)],
                    else_=None
                )
            ).label('avg_response_time')
        ).filter(
            MessageDeliveryStatus.created_at >= cutoff_date
        ).group_by(date_trunc).order_by(date_trunc)

        if channel:
            query = query.filter(MessageDeliveryStatus.channel == channel)

        results = []
        for row in query.all():
            success_rate = (row.successful_attempts / row.total_attempts
                          if row.total_attempts > 0 else 0.0)

            results.append({
                "timestamp": row.time_period.isoformat(),
                "total_attempts": row.total_attempts,
                "successful_attempts": row.successful_attempts,
                "success_rate": success_rate,
                "avg_response_time_ms": float(row.avg_response_time) if row.avg_response_time else None
            })

        return results

    def get_active_channels(self, cutoff_date: datetime) -> List[str]:
        """
        Get list of active channels.

        Args:
            cutoff_date: Only include channels active after this date

        Returns:
            List of channel names
        """
        channels = self.session.query(MessageDeliveryStatus.channel).filter(
            MessageDeliveryStatus.created_at >= cutoff_date
        ).distinct().all()

        return [row.channel for row in channels]


class RateLimitRepository:
    """Repository for rate limit operations."""

    def __init__(self, session: Session):
        """
        Initialize the repository with a database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def create_or_update_rate_limit(self, rate_limit_data: Dict[str, Any]) -> RateLimit:
        """
        Create or update rate limit.

        Args:
            rate_limit_data: Dictionary with rate limit data

        Returns:
            RateLimit object
        """
        user_id = rate_limit_data.get('user_id')
        channel = rate_limit_data.get('channel')

        if not user_id or not channel:
            raise ValueError("User ID and channel are required")

        # Try to get existing rate limit
        rate_limit = self.session.query(RateLimit).filter(
            and_(RateLimit.user_id == user_id, RateLimit.channel == channel)
        ).first()

        try:
            if rate_limit:
                # Update existing record
                for key, value in rate_limit_data.items():
                    if hasattr(rate_limit, key):
                        setattr(rate_limit, key, value)
            else:
                # Create new record
                rate_limit = RateLimit(**rate_limit_data)
                self.session.add(rate_limit)

            self.session.flush()
            _logger.info("Updated rate limit for user %s, channel %s", user_id, channel)
            return rate_limit
        except Exception as e:
            self.session.rollback()
            _logger.error("Failed to update rate limit for user %s, channel %s: %s", user_id, channel, e)
            raise

    def get_rate_limit(self, user_id: str, channel: str) -> Optional[RateLimit]:
        """
        Get rate limit for user and channel.

        Args:
            user_id: User ID
            channel: Channel name

        Returns:
            RateLimit object or None if not found
        """
        return self.session.query(RateLimit).filter(
            and_(RateLimit.user_id == user_id, RateLimit.channel == channel)
        ).first()

    def check_and_consume_token(self, user_id: str, channel: str, default_config: Dict[str, Any]) -> bool:
        """
        Check if user has tokens available and consume one if available.

        Args:
            user_id: User ID
            channel: Channel name
            default_config: Default rate limit configuration

        Returns:
            True if token was consumed, False if rate limited
        """
        current_time = datetime.now(timezone.utc)

        # Get or create rate limit
        rate_limit = self.get_rate_limit(user_id, channel)
        if not rate_limit:
            rate_limit = self.create_or_update_rate_limit({
                'user_id': user_id,
                'channel': channel,
                'tokens': default_config.get('max_tokens', 60),
                'max_tokens': default_config.get('max_tokens', 60),
                'refill_rate': default_config.get('refill_rate', 60),
                'last_refill': current_time
            })

        # Refill tokens based on time elapsed
        rate_limit.refill_tokens(current_time)

        # Try to consume a token
        if rate_limit.consume_token():
            self.session.flush()
            return True

        return False


class ChannelConfigRepository:
    """Repository for channel configuration operations."""

    def __init__(self, session: Session):
        """
        Initialize the repository with a database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def create_channel_config(self, config_data: Dict[str, Any]) -> ChannelConfig:
        """
        Create a new channel configuration.

        Args:
            config_data: Dictionary with channel config data

        Returns:
            Created ChannelConfig object
        """
        try:
            config = ChannelConfig(**config_data)
            self.session.add(config)
            self.session.flush()
            _logger.info("Created channel config for %s", config.channel)
            return config
        except IntegrityError:
            self.session.rollback()
            _logger.exception("Failed to create channel config:")
            raise

    def get_channel_config(self, channel: str) -> Optional[ChannelConfig]:
        """
        Get channel configuration by channel name.

        Args:
            channel: Channel name

        Returns:
            ChannelConfig object or None if not found
        """
        return self.session.query(ChannelConfig).filter(ChannelConfig.channel == channel).first()

    def list_channel_configs(self, enabled_only: bool = False) -> List[ChannelConfig]:
        """
        List channel configurations.

        Args:
            enabled_only: Only return enabled channels

        Returns:
            List of ChannelConfig objects
        """
        query = self.session.query(ChannelConfig)

        if enabled_only:
            query = query.filter(ChannelConfig.enabled == True)

        return query.order_by(asc(ChannelConfig.channel)).all()

    def update_channel_config(self, channel: str, update_data: Dict[str, Any]) -> Optional[ChannelConfig]:
        """
        Update channel configuration.

        Args:
            channel: Channel name
            update_data: Dictionary with fields to update

        Returns:
            Updated ChannelConfig object or None if not found
        """
        config = self.get_channel_config(channel)
        if not config:
            return None

        try:
            for key, value in update_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            self.session.flush()
            _logger.info("Updated channel config for %s", channel)
            return config
        except Exception as e:
            self.session.rollback()
            _logger.error("Failed to update channel config for %s: %s", channel, e)
            raise

    def delete_channel_config(self, channel: str) -> bool:
        """
        Delete channel configuration.

        Args:
            channel: Channel name

        Returns:
            True if deleted, False if not found
        """
        config = self.get_channel_config(channel)
        if not config:
            return False

        try:
            self.session.delete(config)
            _logger.info("Deleted channel config for %s", channel)
            return True
        except Exception as e:
            self.session.rollback()
            _logger.error("Failed to delete channel config for %s: %s", channel, e)
            raise

    def get_enabled_channels(self) -> List[str]:
        """
        Get list of enabled channel names.

        Returns:
            List of enabled channel names
        """
        channels = self.session.query(ChannelConfig.channel).filter(
            ChannelConfig.enabled == True
        ).all()

        return [channel[0] for channel in channels]


class NotificationRepository:
    """Unified repository for all notification service operations."""

    def __init__(self, session: Session, use_optimized: bool = True):
        """
        Initialize the unified repository with a database session.

        Args:
            session: SQLAlchemy database session
            use_optimized: Whether to use optimized repository implementations
        """
        self.session = session

        if use_optimized:
            # Use optimized implementations for better performance
            self.messages = OptimizedMessageRepository(session)
            self.delivery_status = OptimizedDeliveryStatusRepository(session)
            self.rate_limits = OptimizedRateLimitRepository(session)
            # Keep standard implementations for these
            self.channel_configs = ChannelConfigRepository(session)
        else:
            # Use standard implementations
            self.messages = MessageRepository(session)
            self.delivery_status = DeliveryStatusRepository(session)
            self.rate_limits = RateLimitRepository(session)
            self.channel_configs = ChannelConfigRepository(session)

    def commit(self):
        """Commit the current transaction."""
        self.session.commit()

    def rollback(self):
        """Rollback the current transaction."""
        self.session.rollback()

    def close(self):
        """Close the session."""
        self.session.close()