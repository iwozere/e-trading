# Design

## Purpose
The CNN + XGBoost pipeline is designed to provide a hybrid machine learning approach for financial time series analysis. It combines the feature extraction capabilities of Convolutional Neural Networks with the robust classification and regression capabilities of XGBoost to create a comprehensive trading strategy development framework.

**Key Design Principles:**
- **Individual Model Training**: Each data file gets its own specialized CNN model
- **Intelligent Data Handling**: Automatic batching and period management for data providers
- **Comprehensive Artifacts**: Rich output structure with models, reports, and visualizations
- **Scalable Architecture**: Support for multiple data providers and timeframes

## Architecture

### High-Level Architecture
The pipeline consists of several interconnected stages:

1. **Data Loading Stage** (`x_01_data_loader.py`)
   - Multi-provider data discovery and loading
   - **Intelligent Batching**: Automatic handling of provider-specific limits (e.g., yfinance 2y limit)
   - Data quality validation and cleaning
   - Provider-specific configuration management

2. **CNN Training Stage** (`x_02_train_cnn.py`) - **ENHANCED**
   - **Individual Model Training**: Each data file gets its own CNN model
   - **Proper Naming Convention**: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.pth`
   - 1D CNN model architecture for time series
   - Hyperparameter optimization with Optuna
   - **Comprehensive Output**: Individual model files, configs, reports, and visualizations

3. **Embedding Generation Stage** (`x_03_generate_embeddings.py`)
   - Sliding window processing of time series data
   - CNN embedding extraction for all data points
   - Target variable generation

4. **Technical Analysis Stage** (`x_04_ta_features.py`)
   - Technical indicator calculation
   - Feature engineering and combination
   - Feature selection and optimization

5. **XGBoost Optimization Stage** (`x_05_optuna_xgboost.py`)
   - Hyperparameter optimization for XGBoost
   - Feature importance analysis
   - Model selection and validation

6. **XGBoost Training Stage** (`x_06_train_xgboost.py`)
   - Final model training with optimized parameters
   - Cross-validation and performance evaluation
   - Model persistence and configuration saving

7. **Validation Stage** (`x_07_validate_model.py`)
   - Comprehensive model evaluation
   - Backtesting and performance metrics
   - Results visualization and reporting

### Component Design

#### DataLoader Component - **ENHANCED**
- **Responsibilities**: Data discovery, loading, validation, and preprocessing
- **Intelligent Batching**: Automatic period splitting for providers with limits
- **Interfaces**: Configuration-driven data source management
- **Error Handling**: Robust validation with detailed error reporting and fallback mechanisms
- **Output**: Clean, processed data in CSV format with proper naming

#### CNNTrainer Component - **MAJOR UPDATE**
- **Responsibilities**: Individual CNN model training and optimization per data file
- **Architecture**: 1D convolutional layers with configurable parameters
- **Individual Models**: Each data file gets its own trained model
- **Output Structure**: 
  - Model files: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.pth`
  - Config files: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}_config.json`
  - Report files: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}_report.json`
  - Visualization files: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.png`
- **Hyperparameter Optimization**: Optuna-based optimization for each model
- **Visualization**: Training loss plots and model performance charts

#### XGBoostTrainer Component
- **Responsibilities**: XGBoost model training and optimization
- **Architecture**: Gradient boosting with hyperparameter tuning
- **Interfaces**: Optuna-based optimization framework
- **Output**: Optimized XGBoost models and feature importance

#### TechnicalAnalysis Component
- **Responsibilities**: Technical indicator calculation and feature engineering
- **Architecture**: Modular indicator system with configurable parameters
- **Interfaces**: Pandas-based calculations with numpy optimization
- **Output**: Enhanced feature set with technical indicators

## Data Flow

### Input Data Processing - **ENHANCED**
1. **Data Discovery**: Scan `data/raw/` directory for CSV files matching naming patterns
2. **Provider Separation**: Process each data provider independently
3. **Intelligent Batching**: 
   - **yfinance**: Automatically split periods > 2y into smaller chunks
   - **Binance**: Handle 1000 candle limits with automatic batching
   - **Other Providers**: Provider-specific limit handling
4. **Quality Validation**: Check data quality requirements (minimum records, missing data)
5. **Data Cleaning**: Remove outliers and handle missing values
6. **Feature Engineering**: Add derived features (log returns, ratios, moving averages)
7. **CSV Storage**: Save processed data in CSV format for compatibility

### CNN Processing Pipeline - **MAJOR UPDATE**
1. **Individual File Processing**: Process each data file separately
2. **Data Preparation**: Create sliding windows of specified sequence length per file
3. **Model Training**: Train individual CNN with hyperparameter optimization per file
4. **Artifact Generation**: Create individual model files, configs, reports, and visualizations
5. **Embedding Generation**: Extract embeddings for all data points using trained models
6. **Target Creation**: Generate target variables for supervised learning
7. **CSV Output**: Save labeled data with embeddings in CSV format

### XGBoost Processing Pipeline
1. **Feature Combination**: Combine CNN embeddings with technical indicators
2. **Feature Selection**: Remove highly correlated features
3. **Hyperparameter Optimization**: Optimize XGBoost parameters with Optuna
4. **Model Training**: Train final model with optimized parameters
5. **Validation**: Perform cross-validation and backtesting
6. **CSV Processing**: Read and write feature data in CSV format throughout

### Output Generation - **ENHANCED**
1. **Individual Model Artifacts**: 
   - Model files: `.pth` (PyTorch weights)
   - Config files: `_config.json` (architecture and parameters)
   - Scaler files: `_scaler.pkl` (data normalization)
   - Report files: `_report.json` (training results)
   - Visualization files: `.png` (training plots)
2. **Processing Summaries**: Generate provider-specific processing reports
3. **Optimization Reports**: `full_cnn_optimization_report_*.csv` (overall optimization summary)
4. **Performance Metrics**: Calculate comprehensive evaluation metrics
5. **Visualizations**: Create performance charts and analysis plots
6. **CSV Data Files**: All intermediate and final data files in CSV format for compatibility

## Enhanced Output Structure

### Directory Organization
```
src/ml/pipeline/p03_cnn_xgboost/models/cnn/
├── cnn_yfinance_VT_1d_20210829_20250828_20250812_221148.pth
├── cnn_yfinance_VT_1d_20210829_20250828_20250812_221148_config.json
├── cnn_yfinance_VT_1d_20210829_20250828_20250812_221148_scaler.pkl
├── cnn_yfinance_PSNY_1d_20210829_20250828_20250812_221149.pth
├── cnn_yfinance_PSNY_1d_20210829_20250828_20250812_221149_config.json
├── cnn_yfinance_PSNY_1d_20210829_20250828_20250812_221149_scaler.pkl
├── reports/
│   ├── cnn_yfinance_VT_1d_20210829_20250828_20250812_221148_report.json
│   ├── cnn_yfinance_PSNY_1d_20210829_20250828_20250812_221149_report.json
│   └── full_cnn_optimization_report_20250812_221148.csv
└── visualizations/
    ├── cnn_yfinance_VT_1d_20210829_20250828_20250812_221148.png
    └── cnn_yfinance_PSNY_1d_20210829_20250828_20250812_221149.png
```

### File Naming Convention
- **Model Files**: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.pth`
- **Config Files**: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}_config.json`
- **Scaler Files**: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}_scaler.pkl`
- **Report Files**: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}_report.json`
- **Visualization Files**: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.png`

## Design Decisions

### Technology Choices - **UPDATED**
- **PyTorch**: Chosen for CNN implementation due to flexibility and GPU support
- **XGBoost**: Selected for gradient boosting due to superior performance on structured data
- **Optuna**: Used for hyperparameter optimization due to efficient search algorithms
- **CSV Format**: Adopted for data storage due to compatibility, ease of processing, and human readability
- **PyYAML**: Used for configuration management due to readability and flexibility
- **Matplotlib/Seaborn**: Used for visualization due to comprehensive plotting capabilities

### Architecture Patterns - **ENHANCED**
- **Pipeline Pattern**: Sequential processing stages for clear data flow
- **Strategy Pattern**: Provider-specific data handling strategies with intelligent batching
- **Factory Pattern**: Dynamic model creation based on configuration
- **Observer Pattern**: Progress tracking and logging throughout pipeline
- **Individual Model Pattern**: Each data file gets its own specialized model
- **CSV Consistency Pattern**: All data exchange between stages uses CSV format for compatibility

### Performance Considerations - **ENHANCED**
- **Memory Management**: Efficient data loading with chunking for large datasets
- **Parallel Processing**: Multi-provider processing and GPU acceleration
- **Intelligent Batching**: Automatic handling of provider-specific limits
- **Caching**: Intermediate results caching to avoid recomputation
- **Optimization**: Hyperparameter optimization to maximize model performance
- **Individual Models**: Specialized models for each data source for better performance
- **CSV Processing**: Fast CSV I/O operations with pandas for data handling

### Security Decisions
- **Input Validation**: Comprehensive validation of all input data
- **Path Sanitization**: Secure handling of file paths and names
- **Error Handling**: Graceful error handling with detailed logging
- **Data Isolation**: Provider-specific data processing to prevent cross-contamination
- **Timestamp-based Naming**: Unique identifiers to prevent file conflicts

## Integration Patterns

### Module Interactions - **UPDATED**
- **Data Module**: Provides data loading and preprocessing capabilities with intelligent batching
- **Configuration Module**: Manages pipeline configuration and parameters
- **Notification Module**: Handles logging and user notifications
- **Common Module**: Provides shared utilities and constants
- **Visualization Module**: Handles training progress and performance plotting

### API Design Principles - **ENHANCED**
- **Configuration-Driven**: All behavior controlled through YAML configuration
- **Provider-Agnostic**: Support for multiple data providers with consistent interface
- **Intelligent Batching**: Automatic handling of provider-specific limits
- **Individual Models**: Each data file gets its own specialized model
- **Extensible**: Easy addition of new providers, indicators, and models
- **Testable**: Clear separation of concerns for comprehensive testing

### Error Handling and Recovery - **ENHANCED**
- **Graceful Degradation**: Continue processing when individual files fail
- **Intelligent Batching**: Automatic retry with smaller chunks when limits are exceeded
- **Detailed Logging**: Comprehensive error reporting with context
- **Checkpointing**: Save intermediate results to enable recovery
- **Validation**: Multiple validation layers to catch issues early
- **Fallback Mechanisms**: Provider-specific fallback strategies

### Data Persistence - **MAJOR UPDATE**
- **Individual Model Storage**: Hierarchical storage of models by provider, symbol, and timestamp
- **Configuration Persistence**: Save optimized configurations for each model
- **Processing Summaries**: Individual model training reports
- **Performance Metrics**: Comprehensive evaluation results storage
- **Visualization Storage**: Training progress and performance charts
- **Optimization Reports**: Overall optimization summaries across all models
- **CSV Data Storage**: All intermediate and final data files stored in CSV format for compatibility and ease of access
- **Data Flow Persistence**: Consistent CSV format throughout pipeline stages for seamless data exchange
