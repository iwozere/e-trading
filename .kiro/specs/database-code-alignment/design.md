# Design Document

## Overview

This design outlines the alignment of SQLAlchemy models, repositories, services, and tests with the PostgreSQL database schema. The migration has been completed at the database level, but the application code contains several misalignments that need to be corrected. The approach focuses on systematically updating each component to match the actual database structure while maintaining existing functionality.

## Architecture

### Current State Analysis

Based on the schema comparison between `src/data/db/docs/sql.md` and the existing models, several critical misalignments have been identified:

#### Model Misalignments

1. **Job Models (`model_jobs.py`)**
   - `job_id` should be `int8` (bigint), not `String(255)`
   - Missing `worker_id` column in Run model
   - Incorrect constraint names and index definitions

2. **Telegram Models (`model_telegram.py`)**
   - `TelegramFeedback` missing columns: `type`, `message`, `created`, `status`
   - Missing models for tables that exist in database
   - Incorrect field mappings

3. **User Models (`model_users.py`)**
   - `AuthIdentity.identity_metadata` should map to database column `metadata`
   - `VerificationCode` missing `provider` and `created_at` columns
   - Incorrect constraint definitions

4. **Missing Trading Models**
   - Database has trading tables but models may be incomplete
   - Need to verify all trading relationships are properly defined

### Database Schema Structure

The PostgreSQL schema contains these main table groups:

#### User Management Tables
- `usr_users` - Core user data
- `usr_auth_identities` - Authentication providers
- `usr_verification_codes` - Verification codes with provider support

#### Job System Tables  
- `job_schedules` - Job scheduling definitions
- `job_runs` - Job execution history

#### Telegram Integration Tables
- `telegram_broadcast_logs` - Broadcast message history
- `telegram_command_audits` - Command execution auditing
- `telegram_feedbacks` - User feedback system
- `telegram_settings` - Configuration settings

#### Trading System Tables
- `trading_bot_instances` - Bot instance management
- `trading_trades` - Trade execution records
- `trading_positions` - Position management
- `trading_performance_metrics` - Performance tracking

#### Web UI Tables
- `webui_audit_logs` - Web UI audit trail
- `webui_performance_snapshots` - Performance snapshots
- `webui_strategy_templates` - Strategy templates
- `webui_system_config` - System configuration

## Components and Interfaces

### 1. Model Correction Strategy

**Approach:** Update existing models to match database schema exactly

**Key Changes:**
- Fix data types to match PostgreSQL types
- Correct column names and mappings
- Add missing columns and constraints
- Update foreign key relationships

### 2. Missing Model Creation

**Approach:** Create models for any database tables without corresponding Python models

**Implementation:**
- Follow existing code patterns and conventions
- Use proper PostgreSQL data types
- Include all constraints and indexes
- Define proper relationships

### 3. Repository Updates

**Approach:** Update repository code to work with corrected models

**Changes:**
- Fix attribute access for renamed fields
- Update queries to use correct column names
- Ensure compatibility with new model definitions
- Test all CRUD operations

### 4. Test Infrastructure Updates

**Approach:** Update test factories and test cases for new models

**Changes:**
- Fix factory methods for corrected models
- Update test imports and references
- Ensure test data is PostgreSQL-compatible
- Validate all test scenarios

## Data Models

### Corrected Job Models

```python
class Run(Base):
    """Run model for job execution history with snapshots."""
    __tablename__ = "job_runs"

    run_id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    job_type = Column(Text, nullable=False)  # Changed from String(50)
    job_id = Column(BigInteger, nullable=True)  # Changed from String(255) to int8
    user_id = Column(BigInteger, nullable=True)  # Changed to int8
    status = Column(Text, nullable=True)  # Changed from String(20)
    scheduled_for = Column(DateTime(timezone=True), nullable=True)
    enqueued_at = Column(DateTime(timezone=True), nullable=True, default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    job_snapshot = Column(JSONB, nullable=True)
    result = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
    # Missing field that exists in database:
    worker_id = Column(String(255), nullable=True)  # Add missing field

    __table_args__ = (
        # Correct constraint name from database
        UniqueConstraint("job_type", "job_id", "scheduled_for", name="ux_runs_job_scheduled_for"),
    )
```

### Corrected Telegram Models

```python
class TelegramFeedback(Base):
    __tablename__ = "telegram_feedbacks"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("usr_users.id", ondelete="CASCADE"))
    # Missing fields that exist in database:
    type = Column(String(50), nullable=True)
    message = Column(Text, nullable=True) 
    created = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=True)
```

### Corrected User Models

```python
class AuthIdentity(Base):
    __tablename__ = "usr_auth_identities"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("usr_users.id", ondelete="CASCADE"))
    provider = Column(String(32), nullable=False)
    external_id = Column(String(255), nullable=False)
    # Correct column mapping - attribute name vs database column name
    identity_metadata = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True, default=func.now())

    __table_args__ = (
        # Correct index names from database
        Index("ix_auth_identities_provider", "provider"),
        Index("ix_auth_identities_user_id", "user_id"),
    )

class VerificationCode(Base):
    __tablename__ = "usr_verification_codes"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("usr_users.id", ondelete="CASCADE"))
    code = Column(String(32), nullable=False)
    sent_time = Column(Integer, nullable=False)
    # Missing fields that exist in database:
    provider = Column(String(20), nullable=True, default='telegram')
    created_at = Column(DateTime(timezone=True), nullable=True, default=func.now())

    __table_args__ = (
        Index("ix_verification_codes_user_id", "user_id"),
    )
```

## Error Handling

### Model Validation Errors
- Implement comprehensive model validation
- Check for missing required fields
- Validate foreign key relationships
- Ensure data type compatibility

### Migration Compatibility
- Ensure models work with existing data
- Handle any data format differences
- Validate constraint compatibility
- Test with actual PostgreSQL database

## Testing Strategy

### 1. Model Definition Tests
- Validate all models can be imported
- Check table names match database schema
- Verify column definitions are correct
- Test constraint and index definitions

### 2. Database Integration Tests
- Test model creation and table generation
- Validate foreign key relationships
- Check constraint enforcement
- Test CRUD operations

### 3. Repository Function Tests
- Test all repository methods with corrected models
- Validate query correctness
- Check data access patterns
- Test error handling

### 4. Factory and Test Data Tests
- Update test factories for new model definitions
- Ensure test data is PostgreSQL-compatible
- Validate factory-generated data
- Test edge cases and constraints

## Implementation Phases

### Phase 1: Model Corrections
1. Fix job models (Run and Schedule)
2. Correct telegram models (add missing fields)
3. Update user models (fix column mappings)
4. Add any missing model definitions

### Phase 2: Repository Updates
1. Update repository code for corrected models
2. Fix attribute access patterns
3. Update query logic where needed
4. Test all repository operations

### Phase 3: Test Infrastructure
1. Update test factories for new models
2. Fix test imports and references
3. Update test data generation
4. Ensure PostgreSQL compatibility

### Phase 4: Validation and Testing
1. Run comprehensive test suite
2. Validate all CRUD operations
3. Test foreign key relationships
4. Verify constraint enforcement

## Data Type Mappings

### PostgreSQL Type Corrections

```python
# Correct mappings for PostgreSQL
SQLAlchemy Type          → PostgreSQL Type
Text                     → text (not varchar with length)
BigInteger              → int8/bigint
DateTime(timezone=True) → timestamptz
JSONB                   → jsonb
UUID                    → uuid
```

### Column Name Mappings

```python
# Handle SQL keyword conflicts
class AuthIdentity(Base):
    # Python attribute name → Database column name
    identity_metadata = Column("metadata", JSONB)  # metadata is SQL keyword
```

## Validation Approach

### Schema Validation
1. Compare model definitions with actual database schema
2. Validate all table names, column names, and types
3. Check constraint definitions and names
4. Verify index definitions

### Functional Validation
1. Test all model operations (create, read, update, delete)
2. Validate foreign key relationships work correctly
3. Test constraint enforcement
4. Verify query performance and correctness

### Integration Validation
1. Test with actual PostgreSQL database
2. Validate with existing data (if any)
3. Test repository and service layer integration
4. Run full application test suite

## Risk Mitigation

### Data Compatibility
- Test models with existing PostgreSQL data
- Validate data type conversions
- Check for any data format issues
- Ensure backward compatibility where possible

### Performance Considerations
- Verify index usage is optimal
- Check query performance with corrected models
- Validate relationship loading efficiency
- Monitor database connection handling

### Rollback Strategy
- Maintain backup of current model definitions
- Document all changes made
- Provide rollback procedures if needed
- Test rollback scenarios

## Success Criteria

### Model Alignment
- All models accurately reflect PostgreSQL schema
- No missing tables or columns
- All constraints and indexes properly defined
- Foreign key relationships work correctly

### Code Functionality
- All repository operations work correctly
- Service layer functions properly
- Test suite passes completely
- No database-related errors in application

### Performance and Reliability
- Query performance is acceptable
- Database connections are stable
- Error handling works properly
- Application startup is successful