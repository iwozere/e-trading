TODO: Update this file when scheduler implementation is completed.

APScheduler based logic for alerts/schedules execution.

At the startup of the src/scheduler/background_bot.py it takes all the scheduled jobs from job_schedules table and calculates next_run_at for all of them with status PENDING or COMPLETED.
It also should analyze the cron configuration for each of them.

Alerts/schedules are created via API (web ui/telegram) and stored in table job_schedules.
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    job_type = Column(String(50), nullable=False)
    target = Column(String(255), nullable=False)
    task_params = Column(JSON().with_variant(JSONB(), 'postgresql'), nullable=False, default={})
    cron = Column(String(100), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    next_run_at = Column(DateTime(timezone=True), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

Executions of those alerts/schedules are stored in the table job_runs.
    run_id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4, index=True)
    job_type = Column(Text, nullable=False)
    job_id = Column(BigInteger, nullable=True)
    user_id = Column(BigInteger, nullable=True, index=True)
    status = Column(Text, nullable=True, index=True)
    scheduled_for = Column(DateTime(timezone=True), nullable=True, index=True)
    enqueued_at = Column(DateTime(timezone=True), nullable=True, default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    job_snapshot = Column(JSON().with_variant(JSONB(), 'postgresql'), nullable=True)
    result = Column(JSON().with_variant(JSONB(), 'postgresql'), nullable=True)
    error = Column(Text, nullable=True)
    worker_id = Column(String(255), nullable=True)
