"""
Embedding Generation Stage for CNN + XGBoost Pipeline.

This module generates embeddings from OHLCV time series data using the trained CNN model.
The embeddings serve as learned features that will be combined with technical indicators
for the XGBoost classification stage.
"""

import sys
from pathlib import Path

# Add project root to path to import common utilities
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

import json
import pickle
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.notification.logger import setup_logger
from src.util.config import load_config
_logger = setup_logger(__name__)


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for time series feature extraction.

    This is the same architecture as used in the training stage.
    """

    def __init__(self,
                 input_channels: int = 5,
                 sequence_length: int = 120,
                 num_filters: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout_rate: float = 0.3) -> None:
        """
        Initialize the 1D CNN architecture.

        Args:
            input_channels: Number of input features (OHLCV = 5)
            sequence_length: Length of time series sequence
            num_filters: List of filter counts for each convolutional layer
            kernel_sizes: List of kernel sizes for each convolutional layer
            dropout_rate: Dropout rate for regularization
        """
        super(CNN1D, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length

        # Build convolutional layers
        layers = []
        in_channels = input_channels

        for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = filters

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final classification layer (same as training)
        self.classification_layer = nn.Linear(num_filters[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)

        Returns:
            Features tensor of shape (batch_size, num_filters[-1]) before classification
        """
        # Apply convolutional layers
        x = self.conv_layers(x)

        # Global average pooling
        x = self.global_pool(x)

        # Flatten and return features before classification
        x = x.view(x.size(0), -1)

        return x


class EmbeddingGenerator:
    """
    Embedding Generator for the CNN + XGBoost pipeline.

    Loads the trained CNN model and generates embeddings from all raw data files.
    The embeddings are saved alongside the original data for use in the XGBoost stage.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the embedding generator.

        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.cnn_config = config.get("cnn", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _logger.info("Initializing embedding generator on device: %s", self.device)

        # Create output directories
        self.labeled_dir = Path("data/labeled")
        self.labeled_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = None
        self.scaler = None
        self.model_config = None

    def run(self) -> Dict[str, Any]:
        """
        Execute the embedding generation stage.

        Returns:
            Dictionary containing generation results and metadata
        """
        _logger.info("Starting embedding generation stage")

        try:
            # Load trained model and artifacts
            self._load_model_and_artifacts()

            # Discover raw data files
            data_files = self._discover_raw_data()
            if not data_files:
                raise ValueError("No raw data files found")

            _logger.info("Found %d raw data files", len(data_files))

            # Generate embeddings for each file
            generation_results = self._generate_embeddings_for_all_files(data_files)

            # Save generation summary
            self._save_generation_summary(generation_results)

            _logger.info("Embedding generation stage completed successfully")
            return generation_results

        except Exception as e:
            _logger.exception("Error in embedding generation stage: %s", e)
            raise

    def _load_model_and_artifacts(self) -> None:
        """Load the trained CNN model and associated artifacts."""
        models_dir = Path("src/ml/pipeline/p03_cnn_xgboost/models/cnn")

        # Find the most recent model configuration file
        config_files = list(models_dir.glob("*_config.json"))
        if not config_files:
            raise FileNotFoundError(f"No model configuration files found in {models_dir}")

        # Sort by modification time and get the most recent
        config_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        config_path = config_files[0]

        _logger.info("Loading model configuration from: %s", config_path)

        with open(config_path, "r") as f:
            self.model_config = json.load(f)

        # Extract model ID from config filename to find corresponding scaler and model files
        model_id = config_path.stem.replace("_config", "")
        scaler_path = models_dir / f"{model_id}_scaler.pkl"
        model_path = models_dir / f"{model_id}.pth"

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Create and load model
        self.model = CNN1D(
            input_channels=self.model_config["input_channels"],
            sequence_length=self.model_config["sequence_length"],
            num_filters=self.model_config["num_filters"],
            kernel_sizes=self.model_config["kernel_sizes"],
            dropout_rate=self.model_config["dropout_rate"]
        ).to(self.device)

        # Load trained weights
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        _logger.info("Loaded trained CNN model and artifacts from: %s", model_id)

    def _discover_raw_data(self) -> List[Path]:
        """
        Discover raw data files from the data loader stage.

        Returns:
            List of paths to raw data files
        """
        data_dir = Path("data/raw")
        if not data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {data_dir}")

        # Look for CSV files (raw data is saved as CSV)
        data_files = list(data_dir.glob("*.csv"))

        if not data_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        return data_files

    def _generate_embeddings_for_all_files(self, data_files: List[Path]) -> Dict[str, Any]:
        """
        Generate embeddings for all raw data files.

        Args:
            data_files: List of paths to raw data files

        Returns:
            Dictionary containing generation results
        """
        _logger.info("Generating embeddings for %d files", len(data_files))

        results = {
            "files_processed": 0,
            "total_embeddings": 0,
            "failed_files": [],
            "file_results": []
        }

        for file_path in data_files:
            try:
                file_result = self._generate_embeddings_for_file(file_path)
                results["files_processed"] += 1
                results["total_embeddings"] += file_result["embeddings_count"]
                results["file_results"].append(file_result)

                _logger.info("Processed %s: %d embeddings", file_path.name, file_result["embeddings_count"])

            except Exception as e:
                _logger.warning("Failed to process %s: %s", file_path, e)
                results["failed_files"].append({
                    "file": str(file_path),
                    "error": str(e)
                })

        _logger.info("Embedding generation completed: %d files processed, %d total embeddings",
                    results["files_processed"], results["total_embeddings"])

        return results

    def _generate_embeddings_for_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Generate embeddings for a single data file.

        Args:
            file_path: Path to the data file

        Returns:
            Dictionary containing file processing results
        """
        # Load data
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        # Extract OHLCV features
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in ohlcv_cols):
            raise ValueError(f"Missing OHLCV columns in {file_path}")

        ohlcv_data = df[ohlcv_cols].values

        # Create sequences
        sequence_length = self.model_config["sequence_length"]
        sequences = self._create_sequences(ohlcv_data, sequence_length)

        if not sequences:
            raise ValueError(f"No valid sequences found in {file_path}")

        # Normalize sequences
        sequences_normalized = self._normalize_sequences(sequences)

        # Generate embeddings
        embeddings = self._generate_embeddings(sequences_normalized)

        # Create labeled data DataFrame
        labeled_df = self._create_labeled_dataframe(df, embeddings, sequence_length)

        # Save labeled data
        output_path = self._save_labeled_data(labeled_df, file_path)

        return {
            "input_file": str(file_path),
            "output_file": str(output_path),
            "embeddings_count": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "original_rows": len(df),
            "labeled_rows": len(labeled_df)
        }

    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> List[np.ndarray]:
        """
        Create sequences from time series data.

        Args:
            data: OHLCV data array
            sequence_length: Length of each sequence

        Returns:
            List of sequences
        """
        sequences = []

        for i in range(len(data) - sequence_length + 1):
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)

        return sequences

    def _normalize_sequences(self, sequences: List[np.ndarray]) -> np.ndarray:
        """
        Normalize sequences using the fitted scaler.

        Args:
            sequences: List of sequences

        Returns:
            Normalized sequences array
        """
        sequences_array = np.array(sequences)
        sequences_reshaped = sequences_array.reshape(-1, sequences_array.shape[-1])
        sequences_normalized = self.scaler.transform(sequences_reshaped)
        return sequences_normalized.reshape(sequences_array.shape)

    def _generate_embeddings(self, sequences: np.ndarray) -> np.ndarray:
        """
        Generate embeddings using the trained CNN model.

        Args:
            sequences: Normalized sequences array

        Returns:
            Embeddings array
        """
        batch_size = self.config.get("embedding_generation", {}).get("batch_size", 64)

        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)

        # Create data loader
        dataset = TensorDataset(sequences_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embeddings_list = []

        with torch.no_grad():
            for (batch_sequences,) in dataloader:
                # Forward pass
                batch_embeddings = self.model(batch_sequences.transpose(1, 2))
                embeddings_list.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings_list)

    def _create_labeled_dataframe(self,
                                 original_df: pd.DataFrame,
                                 embeddings: np.ndarray,
                                 sequence_length: int) -> pd.DataFrame:
        """
        Create labeled DataFrame with embeddings and original data.

        Args:
            original_df: Original data DataFrame
            embeddings: Generated embeddings array
            sequence_length: Length of sequences used for embedding generation

        Returns:
            Labeled DataFrame with embeddings
        """
        # Create embedding column names
        embedding_cols = [f"embedding_{i}" for i in range(embeddings.shape[1])]

        # Create embeddings DataFrame
        embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)

        # Align with original data (skip first sequence_length-1 rows)
        aligned_df = original_df.iloc[sequence_length-1:].reset_index(drop=True)

        # Combine original data with embeddings
        labeled_df = pd.concat([aligned_df, embeddings_df], axis=1)

        # Add metadata columns
        labeled_df["sequence_start_idx"] = range(sequence_length-1, len(original_df))
        labeled_df["sequence_end_idx"] = range(sequence_length, len(original_df) + 1)

        return labeled_df

    def _save_labeled_data(self, labeled_df: pd.DataFrame, original_file_path: Path) -> Path:
        """
        Save labeled data to the labeled directory.

        Args:
            labeled_df: Labeled DataFrame with embeddings
            original_file_path: Path to original data file

        Returns:
            Path to saved labeled data file
        """
        # Create output filename
        output_filename = f"{original_file_path.stem}_labeled.csv"
        output_path = self.labeled_dir / output_filename

        # Save as CSV
        labeled_df.to_csv(output_path, index=False)

        _logger.debug("Saved labeled data to %s", output_path)

        return output_path

    def _save_generation_summary(self, generation_results: Dict[str, Any]) -> None:
        """
        Save generation summary to file.

        Args:
            generation_results: Results from embedding generation
        """
        summary_path = self.labeled_dir / "embedding_generation_summary.json"

        summary = {
            "stage": "embedding_generation",
            "status": "completed",
            "timestamp": pd.Timestamp.now().isoformat(),
            "device_used": str(self.device),
            "model_config": self.model_config,
            "generation_results": generation_results
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        _logger.info("Saved generation summary to %s", summary_path)


def generate_embeddings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to generate embeddings.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Dictionary containing generation results
    """
    generator = EmbeddingGenerator(config)
    return generator.run()


if __name__ == "__main__":
    # Load configuration
    config_path = Path("config/pipeline/p03.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(str(config_path))

    # Run embedding generation
    results = generate_embeddings(config)
    _logger.info("Embedding Generation Results: %s", results)
