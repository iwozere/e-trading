"""
Dramatiq Configuration

Configuration for Dramatiq message broker and workers.
"""

import os
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Create Redis broker
redis_broker = RedisBroker(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD
)

# Create results backend
results_backend = RedisBackend(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD
)

# Configure broker with results
redis_broker.add_middleware(Results(backend=results_backend))

# Set as default broker
dramatiq.set_broker(redis_broker)

# Export broker for use in workers
broker = redis_broker


def setup_dramatiq():
    """
    Setup Dramatiq configuration.

    This function should be called during application startup.
    """
    try:
        # Test Redis connection
        redis_broker.client.ping()
        _logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

        # Configure worker settings
        dramatiq.set_broker(redis_broker)

        _logger.info("Dramatiq configuration completed successfully")

    except Exception as e:
        _logger.error(f"Failed to setup Dramatiq: {e}")
        raise


def get_broker():
    """
    Get the configured Dramatiq broker.

    Returns:
        RedisBroker instance
    """
    return broker


def get_results_backend():
    """
    Get the configured results backend.

    Returns:
        RedisBackend instance
    """
    return results_backend

