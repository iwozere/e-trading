# Requirements

## Python Dependencies
- `numpy` >= 1.20
- `pandas` >= 1.3
- `reportlab` >= 3.6 (optional, required for PDF report generation)
- `openpyxl` >= 3.0 (optional, required for Excel report generation)

## External Dependencies
- `src.model` - Shared data classes (`Trade`, `PerformanceMetrics`) for domain objects

## External Services
- None

## System Requirements
- Minimal CPU and memory overhead; performance reports process list items efficiently in memory.

## Security Requirements
- Safe path handling for reports output directories using clean OS system calls.

## Performance Requirements
- Sub-second execution times for metric sets under 100,000 trades.
