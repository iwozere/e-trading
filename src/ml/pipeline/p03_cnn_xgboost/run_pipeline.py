#!/usr/bin/env python3
"""
CNN + XGBoost Pipeline Runner

This script orchestrates the complete CNN + XGBoost pipeline for financial time series analysis.
The pipeline consists of 7 stages:
1. Data Loading (x_01_data_loader.py)
2. CNN Training (x_02_train_cnn.py)
3. Embedding Generation (x_03_generate_embeddings.py)
4. TA Feature Engineering (x_04_ta_features.py)
5. XGBoost Optimization (x_05_optuna_xgboost.py)
6. XGBoost Training (x_06_train_xgboost.py)
7. Validation (x_07_validate_model.py)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.util.config import load_config, validate_config
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class CNNXGBoostPipeline:
    """Main pipeline orchestrator for CNN + XGBoost trading system."""

    def __init__(self, config_path: str, skip_stages: Optional[List[int]] = None):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to configuration file
            skip_stages: List of stage numbers to skip
        """
        self.config_path = Path(config_path)
        self.skip_stages = skip_stages or []

        # Load and validate configuration
        self.config = self._load_configuration()

        # Setup logging
        # Pipeline stages
        self.stages = [
            ("Data Loading", self._run_data_loader),
            ("CNN Training", self._run_cnn_training),
            ("Embedding Generation", self._run_embedding_generation),
            ("TA Feature Engineering", self._run_ta_features),
            ("XGBoost Optimization", self._run_xgboost_optimization),
            ("XGBoost Training", self._run_xgboost_training),
            ("Validation", self._run_validation)
        ]

        # Pipeline state
        self.pipeline_state = {
            "start_time": None,
            "end_time": None,
            "completed_stages": [],
            "failed_stages": [],
            "stage_results": {},
            "overall_status": "not_started"
        }

    def _load_configuration(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        try:
            config = load_config(self.config_path)
            validate_config(config, self._get_config_schema())
            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)

    def _get_config_schema(self) -> Dict[str, Any]:
        """Define configuration schema for validation."""
        return {
            "data_sources": dict,  # Multi-provider data sources
            "data": dict,  # Data configuration
            "cnn": dict,  # CNN configuration
            "xgboost": dict,  # XGBoost configuration
            "technical_indicators": dict,  # Technical indicators configuration
            "feature_engineering": dict,  # Feature engineering configuration
            "targets": dict,  # Target configuration
            "training": dict,  # Training configuration
            "validation": dict,  # Validation configuration
            "performance": dict,  # Performance configuration
            "logging": dict,  # Logging configuration
            "output": dict,  # Output configuration
            "advanced": dict  # Advanced configuration
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})

        logger = setup_logger(
            name="cnn_xgboost_pipeline",
            level=log_config.get("level", "INFO"),
            log_file=log_config.get("file", "logs/p03_cnn_xgboost.log"),
            format_type=log_config.get("format", "json")
        )

        return logger

    def run(self) -> bool:
        """
        Run the complete pipeline.

        Returns:
            True if pipeline completed successfully, False otherwise
        """
        _logger.info("Starting CNN + XGBoost Pipeline")
        self.pipeline_state["start_time"] = time.time()
        self.pipeline_state["overall_status"] = "running"

        try:
            # Create necessary directories
            self._create_directories()

            # Run pipeline stages
            for stage_num, (stage_name, stage_func) in enumerate(self.stages, 1):
                if stage_num in self.skip_stages:
                    _logger.info("Skipping stage %d: %s", stage_num, stage_name)
                    continue

                _logger.info("Starting stage %d: %s", stage_num, stage_name)
                stage_start_time = time.time()

                try:
                    result = stage_func()
                    stage_end_time = time.time()

                    self.pipeline_state["stage_results"][stage_num] = {
                        "name": stage_name,
                        "status": "completed",
                        "start_time": stage_start_time,
                        "end_time": stage_end_time,
                        "duration": stage_end_time - stage_start_time,
                        "result": result
                    }

                    self.pipeline_state["completed_stages"].append(stage_num)
                    _logger.info("Completed stage %d: %s in %.2fs", stage_num, stage_name, stage_end_time - stage_start_time)

                except Exception as e:
                    stage_end_time = time.time()
                    _logger.error("Failed stage %d: %s - %s", stage_num, stage_name, e)

                    self.pipeline_state["stage_results"][stage_num] = {
                        "name": stage_name,
                        "status": "failed",
                        "start_time": stage_start_time,
                        "end_time": stage_end_time,
                        "duration": stage_end_time - stage_start_time,
                        "error": str(e)
                    }

                    self.pipeline_state["failed_stages"].append(stage_num)
                    self.pipeline_state["overall_status"] = "failed"

                    # Save pipeline state before exiting
                    self._save_pipeline_state()
                    return False

            # Pipeline completed successfully
            self.pipeline_state["end_time"] = time.time()
            self.pipeline_state["overall_status"] = "completed"

            _logger.info("Pipeline completed successfully")
            self._save_pipeline_state()
            self._generate_summary_report()

            return True

        except Exception:
            _logger.exception("Pipeline failed with error:")
            self.pipeline_state["overall_status"] = "failed"
            self._save_pipeline_state()
            return False

    def _create_directories(self):
        """Create necessary directories for the pipeline."""
        directories = [
            self.config["paths"]["data_labeled"],
            self.config["paths"]["models_cnn"],
            self.config["paths"]["models_xgboost"],
            self.config["paths"]["results"],
            Path(self.config["logging"]["file"]).parent,
            self.config["performance"]["cache_dir"]
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            _logger.debug("Created directory: %s", directory)

    def _run_data_loader(self) -> Dict[str, Any]:
        """Run stage 1: Data loading (download only)."""
        from x_01_data_loader import DataLoader

        data_loader = DataLoader()
        result = data_loader.run()

        _logger.info("Data loader completed. Downloaded %d files", result.get('total_downloads', 0))
        return result

    def _run_cnn_training(self) -> Dict[str, Any]:
        """Run stage 2: CNN training with hyperparameter optimization."""
        from x_02_train_cnn import CNNTrainer

        cnn_trainer = CNNTrainer(self.config)
        result = cnn_trainer.run()

        _logger.info("CNN training completed. Trained %d models", result.get('trained_models', 0))
        return result

    def _run_embedding_generation(self) -> Dict[str, Any]:
        """Run stage 3: Generate embeddings for all data."""
        from x_03_generate_embeddings import EmbeddingGenerator

        embedding_generator = EmbeddingGenerator(self.config)
        result = embedding_generator.run()

        _logger.info("Embedding generation completed. Generated %d embeddings", result.get('embeddings_count', 0))
        return result

    def _run_ta_features(self) -> Dict[str, Any]:
        """Run stage 4: Technical analysis feature engineering."""
        from x_04_ta_features import TAFeatureEngineer

        ta_engineer = TAFeatureEngineer(self.config)
        result = ta_engineer.run()

        _logger.info("TA feature engineering completed. Generated %d features", result.get('features_count', 0))
        return result

    def _run_xgboost_optimization(self) -> Dict[str, Any]:
        """Run stage 5: XGBoost hyperparameter optimization."""
        from x_05_optuna_xgboost import XGBoostOptimizer

        xgb_optimizer = XGBoostOptimizer(self.config)
        result = xgb_optimizer.run()

        _logger.info("XGBoost optimization completed. Best score: %.4f", result.get('best_score', 0))
        return result

    def _run_xgboost_training(self) -> Dict[str, Any]:
        """Run stage 6: Final XGBoost model training."""
        from x_06_train_xgboost import XGBoostTrainer

        xgb_trainer = XGBoostTrainer(self.config)
        result = xgb_trainer.run()

        _logger.info("XGBoost training completed. Trained %d models", result.get('trained_models', 0))
        return result

    def _run_validation(self) -> Dict[str, Any]:
        """Run stage 7: Model validation and backtesting."""
        from x_07_validate_model import ModelValidator

        validator = ModelValidator(self.config)
        result = validator.run()

        _logger.info("Validation completed. Overall accuracy: %.4f", result.get('overall_accuracy', 0))
        return result

    def _save_pipeline_state(self):
        """Save pipeline state to file."""
        state_file = Path(self.config["paths"]["results"]) / "pipeline_state.json"

        # Convert timestamps to strings for JSON serialization
        state_copy = self.pipeline_state.copy()
        if state_copy["start_time"]:
            state_copy["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(state_copy["start_time"]))
        if state_copy["end_time"]:
            state_copy["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(state_copy["end_time"]))

        # Convert stage timestamps
        for stage_result in state_copy["stage_results"].values():
            if stage_result["start_time"]:
                stage_result["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stage_result["start_time"]))
            if stage_result["end_time"]:
                stage_result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stage_result["end_time"]))

        with open(state_file, 'w') as f:
            json.dump(state_copy, f, indent=2)

        _logger.info("Pipeline state saved to %s", state_file)

    def _generate_summary_report(self):
        """Generate a summary report of the pipeline execution."""
        if self.pipeline_state["overall_status"] != "completed":
            return

        total_duration = self.pipeline_state["end_time"] - self.pipeline_state["start_time"]
        completed_stages = len(self.pipeline_state["completed_stages"])

        report = {
            "pipeline_summary": {
                "status": self.pipeline_state["overall_status"],
                "total_duration": f"{total_duration:.2f}s",
                "completed_stages": completed_stages,
                "total_stages": len(self.stages),
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.pipeline_state["start_time"])),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.pipeline_state["end_time"]))
            },
            "stage_summary": {}
        }

        for stage_num, stage_result in self.pipeline_state["stage_results"].items():
            report["stage_summary"][f"stage_{stage_num}"] = {
                "name": stage_result["name"],
                "status": stage_result["status"],
                "duration": f"{stage_result['duration']:.2f}s"
            }

        report_file = Path(self.config["paths"]["results"]) / "pipeline_summary.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        _logger.info("Pipeline summary saved to %s", report_file)

        # Print summary to console
        print("\n" + "="*60)
        print("CNN + XGBoost Pipeline Summary")
        print("="*60)
        print(f"Status: {report['pipeline_summary']['status']}")
        print(f"Total Duration: {report['pipeline_summary']['total_duration']}")
        print(f"Completed Stages: {report['pipeline_summary']['completed_stages']}/{report['pipeline_summary']['total_stages']}")
        print(f"Start Time: {report['pipeline_summary']['start_time']}")
        print(f"End Time: {report['pipeline_summary']['end_time']}")
        print("\nStage Details:")
        for stage_key, stage_info in report["stage_summary"].items():
            print(f"  {stage_info['name']}: {stage_info['status']} ({stage_info['duration']})")
        print("="*60)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="CNN + XGBoost Pipeline Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline/p03.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-stages",
        type=str,
        help="Comma-separated list of stage numbers to skip (e.g., '4,5')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Parse skip stages
    skip_stages = []
    if args.skip_stages:
        try:
            skip_stages = [int(x.strip()) for x in args.skip_stages.split(",")]
        except ValueError:
            _logger.exception("Error: skip-stages must be comma-separated integers")
            sys.exit(1)

    # Create and run pipeline
    pipeline = CNNXGBoostPipeline(args.config, skip_stages)

    success = pipeline.run()

    if success:
        _logger.info("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        _logger.error("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
