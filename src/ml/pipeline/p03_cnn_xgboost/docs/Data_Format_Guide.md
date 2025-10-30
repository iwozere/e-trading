# Data Format Guide

## Overview
The CNN + XGBoost pipeline uses **CSV format consistently** throughout all stages for data storage and exchange. This ensures compatibility, ease of processing, and human readability.

## Data Format Consistency

### Why CSV Format?
- **Compatibility**: Works with most data analysis tools and libraries
- **Human Readable**: Easy to inspect and debug data issues
- **Wide Support**: Supported by pandas, numpy, and other data science libraries
- **No Dependencies**: No special libraries required for reading/writing
- **Version Control Friendly**: Text-based format works well with git

### Pipeline Data Flow

#### Stage 1: Data Loading (`x_01_data_loader.py`)
- **Input**: Downloads from data providers (yfinance, Binance, etc.)
- **Output**: `data/raw/*.csv`
- **Format**: Standard OHLCV data with columns: `open`, `high`, `low`, `close`, `volume`, `date`, `timestamp`

#### Stage 2: CNN Training (`x_02_train_cnn.py`)
- **Input**: `data/raw/*.csv`
- **Output**: Model files (`.pth`, `.json`, `.pkl`) - not CSV
- **Purpose**: Trains individual CNN models for each data file

#### Stage 3: Embedding Generation (`x_03_generate_embeddings.py`)
- **Input**: `data/raw/*.csv`
- **Output**: `data/labeled/*_labeled.csv`
- **Format**: Original OHLCV data + CNN embeddings + metadata
- **Columns**: `open`, `high`, `low`, `close`, `volume`, `date`, `timestamp`, `embedding_0`, `embedding_1`, ..., `sequence_start_idx`, `sequence_end_idx`

#### Stage 4: Technical Analysis (`x_04_ta_features.py`)
- **Input**: `data/labeled/*_labeled.csv`
- **Output**: `data/features/*_features.csv`
- **Format**: Labeled data + technical indicators + target variables
- **Columns**: Original columns + embeddings + technical indicators + `target_direction`, `target_volatility`, `target_trend`, `target_magnitude`

#### Stage 5: XGBoost Training (`x_05_optuna_xgboost.py`, `x_06_train_xgboost.py`)
- **Input**: `data/features/*_features.csv`
- **Output**: Model files (`.pkl`, `.json`) - not CSV
- **Purpose**: Trains XGBoost models using the feature-rich data

#### Stage 6: Validation (`x_07_validate_model.py`)
- **Input**: `data/features/*_features.csv`
- **Output**: Validation reports (`.json`) - not CSV
- **Purpose**: Evaluates model performance and generates reports

## File Naming Conventions

### Raw Data Files
```
{provider}_{symbol}_{timeframe}_{start_date}_{end_date}.csv
```
Example: `yfinance_VT_1d_20210829_20250828.csv`

### Labeled Data Files
```
{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_labeled.csv
```
Example: `yfinance_VT_1d_20210829_20250828_labeled.csv`

### Feature Data Files
```
{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_features.csv
```
Example: `yfinance_VT_1d_20210829_20250828_features.csv`

## Data Structure Examples

### Raw Data (CSV)
```csv
open,high,low,close,volume,date,timestamp
100.0,101.0,99.0,100.5,1000000,2021-08-29,1630195200
100.5,102.0,100.0,101.5,1200000,2021-08-30,1630281600
...
```

### Labeled Data (CSV)
```csv
open,high,low,close,volume,date,timestamp,embedding_0,embedding_1,...,embedding_127,sequence_start_idx,sequence_end_idx
100.5,102.0,100.0,101.5,1200000,2021-08-30,1630281600,0.123,0.456,...,0.789,119,120
101.5,103.0,101.0,102.5,1100000,2021-08-31,1630368000,0.234,0.567,...,0.890,120,121
...
```

### Feature Data (CSV)
```csv
open,high,low,close,volume,date,timestamp,embedding_0,...,embedding_127,rsi,macd,bb_position,...,target_direction,target_volatility,target_trend,target_magnitude
100.5,102.0,100.0,101.5,1200000,2021-08-30,1630281600,0.123,...,0.789,65.2,0.5,0.7,...,1,0,1,0
101.5,103.0,101.0,102.5,1100000,2021-08-31,1630368000,0.234,...,0.890,68.1,0.6,0.8,...,0,1,0,1
...
```

## Benefits of CSV Format

### Development Benefits
- **Easy Debugging**: Can open files in Excel, text editors, or pandas
- **Quick Inspection**: Use `head`, `tail`, `wc -l` commands
- **Version Control**: Track changes in git
- **Cross-Platform**: Works on Windows, Mac, Linux

### Processing Benefits
- **Fast I/O**: pandas CSV reading is highly optimized
- **Memory Efficient**: Can read large files in chunks
- **Flexible**: Easy to add/remove columns
- **Compatible**: Works with most data science tools

### Maintenance Benefits
- **No Dependencies**: Standard library support
- **Backward Compatible**: Easy to migrate from other formats
- **Human Friendly**: Non-technical users can understand
- **Tool Agnostic**: Works with any CSV-capable tool

## Migration from Parquet

### Why the Change?
The pipeline previously used Parquet format for some stages, but was migrated to CSV for:
- **Consistency**: All stages now use the same format
- **Simplicity**: No need to manage multiple file formats
- **Compatibility**: Better integration with existing tools
- **Debugging**: Easier to inspect and troubleshoot data issues

### Migration Details
- **Embedding Generation**: Now saves `*_labeled.csv` instead of `*_labeled.parquet`
- **Technical Analysis**: Now saves `*_features.csv` instead of `*_features.parquet`
- **XGBoost Stages**: Now read CSV files instead of Parquet files
- **Validation**: Now reads CSV files for validation

## Best Practices

### File Handling
- Always use pandas for CSV operations: `pd.read_csv()`, `df.to_csv()`
- Specify `index=False` when saving to avoid extra index columns
- Use appropriate encoding: `encoding='utf-8'` for international characters
- Handle missing values consistently across stages

### Performance
- Use `chunksize` parameter for large files
- Consider compression for storage: `compression='gzip'`
- Use appropriate data types to reduce memory usage
- Clean data before saving to avoid issues downstream

### Naming
- Follow the established naming conventions
- Use descriptive suffixes: `_labeled`, `_features`
- Include timestamps for versioning when needed
- Use consistent date formats: `YYYYMMDD`

## Troubleshooting

### Common Issues
1. **File Not Found**: Check if CSV files exist in expected directories
2. **Column Mismatch**: Verify column names match between stages
3. **Encoding Issues**: Use UTF-8 encoding for international characters
4. **Memory Issues**: Use chunked reading for large files

### Debugging Tips
- Use `df.head()` to inspect data structure
- Check `df.columns` for column names
- Use `df.info()` to see data types and memory usage
- Use `df.isnull().sum()` to check for missing values

## Related Documentation
- [README](README.md) - Pipeline overview and quick start
- [Design](Design.md) - Architecture and design decisions
- [Requirements](Requirements.md) - Technical requirements
- [Tasks](Tasks.md) - Implementation roadmap
