"""
Feature Extraction Module for CNN-LSTM-XGBoost Pipeline

This module extracts features from the trained CNN-LSTM model for use with XGBoost.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

# Import the HybridCNNLSTM model from the optimization module
from x_03_optuna_cnn_lstm import HybridCNNLSTM

class FeatureExtractor:
    def __init__(self, config_path: str = "config/pipeline/x02.yaml"):
        """
        Initialize FeatureExtractor with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Directory setup
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.models_dir = Path(self.config['paths']['models_cnn_lstm'])
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.results_dir = Path(self.config['paths']['results'])
        self.predictions_dir = self.results_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.cnn_lstm_config = self.config.get('cnn_lstm', {})
        self.hardware_config = self.config.get('hardware', {})

        # Device setup
        self.device = self._setup_device()

        # Load data and model
        self.data = self._load_data()
        self.model = self._load_trained_model()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def _setup_device(self) -> torch.device:
        """Setup device for inference."""
        device_setting = self.hardware_config.get('device', 'auto')

        if device_setting == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_setting == 'cuda':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')

        _logger.info("Using device: %s", device)
        return device

    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load processed data from labeled data directory."""
        _logger.info("Loading processed data...")

        # Find processed data files
        npz_files = list(self.labeled_data_dir.glob("processed_*.npz"))

        if not npz_files:
            raise FileNotFoundError("No processed data files found")

        # Load the first file (you might want to modify this to handle multiple files)
        data_file = npz_files[0]
        _logger.info("Loading data from: %s", data_file.name)

        data = np.load(data_file)

        return {
            'X_train': data['X_train'],
            'X_val': data['X_val'],
            'X_test': data['X_test'],
            'y_train': data['y_train'],
            'y_val': data['y_val'],
            'y_test': data['y_test'],
            'feature_names': data['feature_names'].tolist() if 'feature_names' in data else []
        }

    def _load_trained_model(self) -> HybridCNNLSTM:
        """Load the trained CNN-LSTM model."""
        _logger.info("Loading trained CNN-LSTM model...")

        # Find the most recent model file
        model_files = list(self.checkpoints_dir.glob("cnn_lstm_model_*.pth"))

        if not model_files:
            raise FileNotFoundError("No trained model files found")

        # Load the most recent model
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        _logger.info("Loading model from: %s", latest_model_file.name)

        checkpoint = torch.load(latest_model_file, map_location=self.device)

        # Create model with saved configuration
        model_config = checkpoint['model_config']
        model = HybridCNNLSTM(
            time_steps=model_config['time_steps'],
            features=model_config['features'],
            conv_filters=model_config['conv_filters'],
            lstm_units=model_config['lstm_units'],
            dense_units=model_config['dense_units'],
            attention_heads=model_config['attention_heads'],
            dropout=model_config['dropout'],
            kernel_size=model_config['kernel_size']
        )

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        _logger.info("Model loaded successfully")
        return model

    def extract_cnn_lstm_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from the CNN-LSTM model.

        Args:
            X: Input data of shape (samples, time_steps, features)

        Returns:
            Extracted features
        """
        _logger.info("Extracting CNN-LSTM features...")

        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Extract features from different layers
        features = []

        with torch.no_grad():
            batch_size = X_tensor.size(0)

            # Transpose for convolutional layer: (batch, features, time_steps)
            x = X_tensor.permute(0, 2, 1)

            # Convolutional features
            conv_out = self.model.conv1(x)
            conv_out = self.model.relu(conv_out)
            conv_out = self.model.dropout1(conv_out)

            # Global average pooling over time dimension
            conv_features = torch.mean(conv_out, dim=2)  # (batch, conv_filters)
            features.append(conv_features.cpu().numpy())

            # Transpose back for LSTM: (batch, time_steps, conv_filters)
            x = conv_out.permute(0, 2, 1)

            # First LSTM features
            lstm_out, _ = self.model.lstm1(x)

            # Attention features
            attn_output, attn_weights = self.model.attention(lstm_out, lstm_out, lstm_out)

            # Global average pooling over time dimension
            lstm_features = torch.mean(lstm_out, dim=1)  # (batch, lstm_units)
            attn_features = torch.mean(attn_output, dim=1)  # (batch, lstm_units)
            features.append(lstm_features.cpu().numpy())
            features.append(attn_features.cpu().numpy())

            # Second LSTM features
            lstm_out2, _ = self.model.lstm2(attn_output)
            lstm2_features = torch.mean(lstm_out2, dim=1)  # (batch, dense_units)
            features.append(lstm2_features.cpu().numpy())

            # Final output features (before the last linear layer)
            final_features = lstm_out2[:, -1, :]  # (batch, dense_units)
            features.append(final_features.cpu().numpy())

        # Concatenate all features
        cnn_lstm_features = np.concatenate(features, axis=1)

        _logger.info("Extracted CNN-LSTM features shape: %s", cnn_lstm_features.shape)
        return cnn_lstm_features

    def get_technical_indicators(self, X: np.ndarray) -> np.ndarray:
        """
        Extract technical indicators from the input data.

        Args:
            X: Input data of shape (samples, time_steps, features)

        Returns:
            Technical indicators for each sample
        """
        _logger.info("Extracting technical indicators...")

        # Get the last timestep of each sequence (most recent values)
        last_timestep = X[:, -1, :]

        # Create feature names mapping
        feature_names = self.data['feature_names']

        # Select technical indicator features (exclude OHLCV)
        technical_features = []
        for i, name in enumerate(feature_names):
            if name not in ['open', 'high', 'low', 'close', 'volume']:
                technical_features.append(last_timestep[:, i])

        if technical_features:
            technical_indicators = np.column_stack(technical_features)
        else:
            # If no technical indicators, use OHLCV features
            technical_indicators = last_timestep

        _logger.info("Technical indicators shape: %s", technical_indicators.shape)
        return technical_indicators

    def combine_features(self, cnn_lstm_features: np.ndarray, technical_indicators: np.ndarray) -> np.ndarray:
        """
        Combine CNN-LSTM features with technical indicators.

        Args:
            cnn_lstm_features: Features from CNN-LSTM model
            technical_indicators: Technical indicator features

        Returns:
            Combined features
        """
        _logger.info("Combining features...")

        # Combine features
        combined_features = np.concatenate([cnn_lstm_features, technical_indicators], axis=1)

        _logger.info("Combined features shape: %s", combined_features.shape)
        return combined_features

    def extract_all_features(self) -> Dict[str, np.ndarray]:
        """
        Extract features for all data splits.

        Returns:
            Dictionary with features for each split
        """
        _logger.info("Extracting features for all data splits...")

        features = {}

        for split_name in ['train', 'val', 'test']:
            _logger.info("Processing %s split...", split_name)

            X = self.data[f'X_{split_name}']

            # Extract CNN-LSTM features
            cnn_lstm_features = self.extract_cnn_lstm_features(X)

            # Extract technical indicators
            technical_indicators = self.get_technical_indicators(X)

            # Combine features
            combined_features = self.combine_features(cnn_lstm_features, technical_indicators)

            features[split_name] = combined_features

            _logger.info("%s features shape: %d", split_name, combined_features.shape)

        return features

    def save_features(self, features: Dict[str, np.ndarray]) -> Dict[str, str]:
        """
        Save extracted features to files.

        Args:
            features: Dictionary with features for each split

        Returns:
            Dictionary with file paths
        """
        _logger.info("Saving extracted features...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}

        for split_name, split_features in features.items():
            # Save as numpy array
            npz_path = self.predictions_dir / f"xgboost_features_{split_name}_{timestamp}.npz"
            np.savez_compressed(
                npz_path,
                features=split_features,
                targets=self.data[f'y_{split_name}'],
                feature_names=self.data['feature_names']
            )
            saved_files[f'{split_name}_npz'] = str(npz_path)

            # Save as CSV for easier inspection
            csv_path = self.predictions_dir / f"xgboost_features_{split_name}_{timestamp}.csv"
            feature_df = pd.DataFrame(
                split_features,
                columns=[f'feature_{i}' for i in range(split_features.shape[1])]
            )
            feature_df['target'] = self.data[f'y_{split_name}']
            feature_df.to_csv(csv_path, index=False)
            saved_files[f'{split_name}_csv'] = str(csv_path)

        # Save feature info
        info_path = self.predictions_dir / f"xgboost_features_info_{timestamp}.json"
        feature_info = {
            'timestamp': timestamp,
            'feature_shapes': {split: features[split].shape for split in features.keys()},
            'feature_names': self.data['feature_names'],
            'cnn_lstm_features_count': features['train'].shape[1] - len(self.data['feature_names']),
            'technical_indicators_count': len(self.data['feature_names']),
            'total_features': features['train'].shape[1],
            'saved_files': saved_files
        }

        with open(info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)

        saved_files['info'] = str(info_path)

        _logger.info("Features saved successfully")
        return saved_files

    def run(self) -> Dict[str, Any]:
        """
        Run the complete feature extraction process.

        Returns:
            Dictionary with extraction results
        """
        _logger.info("Starting feature extraction process...")

        try:
            # Extract features for all splits
            features = self.extract_all_features()

            # Save features
            saved_files = self.save_features(features)

            # Create summary
            summary = {
                'train_samples': features['train'].shape[0],
                'val_samples': features['val'].shape[0],
                'test_samples': features['test'].shape[0],
                'feature_count': features['train'].shape[1],
                'cnn_lstm_features': features['train'].shape[1] - len(self.data['feature_names']),
                'technical_indicators': len(self.data['feature_names'])
            }

            _logger.info("Feature extraction completed successfully!")
            _logger.info("Summary: %s", summary)

            return {
                'success': True,
                'features': features,
                'saved_files': saved_files,
                'summary': summary
            }

        except Exception as e:
            _logger.exception("Feature extraction failed:")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Main entry point for feature extraction."""
    import argparse

    parser = argparse.ArgumentParser(description='Feature extraction for XGBoost')
    parser.add_argument('--config', default='config/pipeline/x02.yaml', help='Configuration file path')
    parser.add_argument('--output-dir', help='Output directory for features')

    args = parser.parse_args()

    try:
        extractor = FeatureExtractor(args.config)

        # Override output directory if provided
        if args.output_dir:
            extractor.predictions_dir = Path(args.output_dir)
            extractor.predictions_dir.mkdir(parents=True, exist_ok=True)

        results = extractor.run()

        if results['success']:
            print("Feature extraction completed successfully!")
            print(f"Summary: {results['summary']}")
            print(f"Features saved to: {results['saved_files']}")
        else:
            print(f"Feature extraction failed: {results['error']}")
            sys.exit(1)

    except Exception as e:
        _logger.error("Feature extraction failed: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
