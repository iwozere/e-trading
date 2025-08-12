# Pipeline Error Handling

This document describes the improved error handling system for the HMM-LSTM trading pipeline.

## Overview

The pipeline now implements a **fail-fast** approach with configurable error handling that distinguishes between critical and optional stages. This prevents wasted resources and ensures data consistency.

## Key Features

### 1. Fail-Fast Mode (Default)
- **Enabled by default**: Pipeline stops immediately when a critical stage fails
- **Prevents resource waste**: No time spent on downstream stages that depend on failed upstream stages
- **Clear error reporting**: Detailed error messages with recovery suggestions

### 2. Stage Criticality Classification

#### Critical Stages (Fail-Fast Enabled)
These stages are essential for the pipeline to produce valid results:

1. **Data Loading** (Stage 1) - Downloads OHLCV data
2. **Data Preprocessing** (Stage 2) - Adds features and indicators
3. **HMM Training** (Stage 3) - Trains regime detection models
4. **HMM Application** (Stage 4) - Applies HMM models to label data
5. **LSTM Training** (Stage 7) - Trains the main prediction model

#### Optional Stages (Can Fail Without Stopping Pipeline)
These stages enhance the pipeline but are not essential:

6. **Indicator Optimization** (Stage 5) - Optimizes technical indicator parameters
7. **LSTM Optimization** (Stage 6) - Optimizes LSTM hyperparameters
8. **Model Validation** (Stage 8) - Validates models and generates reports

## Command Line Options

### Basic Usage
```bash
# Default behavior (fail-fast enabled)
python run_pipeline.py

# Disable fail-fast mode
python run_pipeline.py --no-fail-fast

# Continue even if optional stages fail
python run_pipeline.py --continue-on-optional-failures

# Skip optional stages to focus on core pipeline
python run_pipeline.py --skip-stages 5,6,8
```

### Advanced Usage
```bash
# Skip specific stages
python run_pipeline.py --skip-stages 1,2,3

# Process specific symbols
python run_pipeline.py --symbols BTCUSDT,ETHUSDT

# Override timeframes
python run_pipeline.py --timeframes 1h,4h,1d

# Combine options
python run_pipeline.py --skip-stages 5,6,8 --symbols BTCUSDT --no-fail-fast
```

## Error Handling Behavior

### Critical Stage Failure
When a critical stage fails:

1. **Error logged**: Detailed error message with context
2. **Pipeline stops**: Execution halts immediately (fail-fast mode)
3. **Clear guidance**: Recovery instructions provided
4. **Exit code**: Non-zero exit code for automation

Example output:
```
CRITICAL STAGE FAILED: Stage 3 (HMM Training)
Error: No processed data found for BTCUSDT 1h
Pipeline stopped due to critical stage failure (fail-fast mode enabled)
Fix the issue and restart the pipeline from stage 3
```

### Optional Stage Failure
When an optional stage fails:

1. **Warning logged**: Stage failure noted but not fatal
2. **Pipeline continues**: Execution proceeds to next stage
3. **User choice**: Option to continue or stop
4. **Partial success**: Pipeline can complete successfully

Example output:
```
OPTIONAL STAGE FAILED: Stage 5 (Indicator Optimization)
Error: Optimization timeout after 30 minutes
Optional stage failed but continuing (continue_on_optional_failures=True)
```

## Error Recovery

### When Critical Stages Fail

1. **Check the error message**: Understand what went wrong
2. **Review logs**: Look for detailed error information
3. **Fix the issue**: Address the root cause (data, config, etc.)
4. **Restart from failed stage**: Use `--skip-stages` to resume

Example recovery:
```bash
# If stage 3 failed, restart from stage 3
python run_pipeline.py --skip-stages 1,2

# If stage 4 failed, restart from stage 4
python run_pipeline.py --skip-stages 1,2,3
```

### When Optional Stages Fail

1. **Assess impact**: Determine if the failure affects your use case
2. **Use continue option**: `--continue-on-optional-failures`
3. **Skip problematic stage**: `--skip-stages 5` (for stage 5)
4. **Proceed with defaults**: Pipeline can work with default parameters

## Best Practices

### For Production Use
```bash
# Use fail-fast mode (default)
python run_pipeline.py

# Skip optional stages for faster execution
python run_pipeline.py --skip-stages 5,6,8
```

### For Development/Debugging
```bash
# Disable fail-fast to see all errors
python run_pipeline.py --no-fail-fast

# Continue on optional failures
python run_pipeline.py --continue-on-optional-failures
```

### For Testing
```bash
# Test with minimal data
python run_pipeline.py --symbols BTCUSDT --timeframes 1h

# Validate requirements only
python run_pipeline.py --validate-only
```

## Exit Codes

- **0**: Pipeline completed successfully
- **1**: Pipeline failed (critical stage failure or user interruption)

## Logging

The pipeline provides comprehensive logging:

- **Info level**: Normal operation progress
- **Warning level**: Optional stage failures, non-critical issues
- **Error level**: Critical stage failures, fatal errors
- **Debug level**: Detailed execution information

## Configuration

Error handling behavior can be configured in the pipeline configuration file:

```yaml
# config/pipeline/x01.yaml
pipeline:
  fail_fast: true  # Default: true
  continue_on_optional_failures: false  # Default: false
  max_retries: 3  # Future: retry failed stages
```

## Troubleshooting

### Common Issues

1. **Data not found**: Ensure data loading stage completed successfully
2. **Configuration errors**: Check YAML syntax and required fields
3. **Memory issues**: Reduce batch sizes or use fewer symbols/timeframes
4. **Network timeouts**: Increase timeout values for data downloads

### Getting Help

1. **Check logs**: Review detailed error messages
2. **Validate requirements**: Run `python run_pipeline.py --validate-only`
3. **List stages**: Run `python run_pipeline.py --list-stages`
4. **Test with minimal setup**: Use single symbol/timeframe for testing

## Migration from Old Behavior

If you were using the old pipeline behavior:

- **Old**: Pipeline continued on all failures, asked user for input
- **New**: Pipeline stops on critical failures by default
- **Migration**: Use `--no-fail-fast` to restore old behavior if needed

The new behavior is more robust and prevents wasted computational resources while providing clearer error guidance.
