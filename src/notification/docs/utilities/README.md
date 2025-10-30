# Notification Service Utilities

This directory contains utility scripts and maintenance tools for the notification service. These are not part of the core system runtime but are useful for administration, optimization, and debugging.

## Scripts

### Database Management
- **`database_migrations.py`** - Database migration scripts for adding optimized indexes and constraints
- **`apply_optimizations.py`** - One-time script to apply all database optimizations
- **`verify_optimizations.py`** - Verification script to test optimization components

### Monitoring and Analysis
- **`query_analyzer.py`** - Database query performance analysis tools
- **`performance_dashboard.py`** - Real-time performance monitoring dashboard

## Usage

### Applying Database Optimizations
```bash
# Apply all optimizations to the database
python src/notification/docs/utilities/apply_optimizations.py
```

### Verifying Optimizations
```bash
# Test that optimization components work correctly
python src/notification/docs/utilities/verify_optimizations.py
```

### Performance Monitoring
```bash
# Start the performance dashboard
python src/notification/docs/utilities/performance_dashboard.py
```

## Note

These utilities are designed for:
- Database administrators
- System maintenance
- Performance tuning
- Development and testing

They are not required for normal notification service operation and should be used with caution in production environments.