# Scheduler Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash scheduler
RUN chown -R scheduler:scheduler /app
USER scheduler

# Set environment variables
ENV PYTHONPATH=/app
ENV TRADING_ENV=production
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import asyncio; from src.scheduler.main import SchedulerApplication; from src.scheduler.config import SchedulerServiceConfig; asyncio.run(SchedulerApplication(SchedulerServiceConfig()).get_status())" || exit 1

# Expose port (if needed for health checks)
EXPOSE 8001

# Run the scheduler service
CMD ["python", "-m", "src.scheduler.main"]