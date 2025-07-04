# Machine Learning Module (`src/ml`)

This directory contains advanced machine learning pipelines, model management, and regime detection tools for trading and financial analytics.

## File Descriptions

### feature_engineering_pipeline.py
Comprehensive feature engineering pipeline:
- Automated feature extraction
- Technical indicator features
- Market microstructure features
- Feature selection and validation

### automated_training_pipeline.py
Automated training pipeline:
- Scheduled model retraining
- Performance monitoring
- A/B testing framework
- Model drift detection

### mlflow_integration.py
MLflow integration for advanced ML model management:
- Model versioning and tracking
- Experiment management
- Model registry
- Automated model deployment

### nn_regime_detector.py
PyTorch-based neural network for market regime detection:
- LSTM-based sequence classifier for bull/bear/sideways regimes
- Fit, predict, save, and load methods
- GPU support if available

### hmm_regime_detector.py
HMM-based regime detector for financial time series:
- Fits a GaussianHMM to returns or features
- Predicts regimes for new data
- Plots detected regimes

### helformer_optuna_train.py
Hybrid deep learning model (Helformer) with Optuna hyperparameter optimization:
- Data processing and sequence creation
- Transformer + LSTM architecture
- Optuna-based hyperparameter search
- Inference and trading signal generation

---

For more details, see the docstrings and code in each file. 