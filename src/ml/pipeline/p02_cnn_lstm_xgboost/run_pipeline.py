"""
CNN-LSTM-XGBoost Pipeline Runner

This module orchestrates the complete CNN-LSTM-XGBoost pipeline, running all stages
in sequence or selectively based on configuration and command-line arguments.

Pipeline Stages:
1. Data Loading (x_01_data_loader.py)
2. Feature Engineering (x_02_feature_engineering.py)
3. CNN-LSTM Optimization (x_03_optuna_cnn_lstm.py)
4. CNN-LSTM Training (x_04_train_cnn_lstm.py)
5. Feature Extraction (x_05_extract_features.py)
6. XGBoost Optimization (x_06_optuna_xgboost.py)
7. XGBoost Training (x_07_train_xgboost.py)
8. Model Validation (x_08_validate_models.py)

Features:
- Stage-by-stage execution with dependency management
- Parallel processing where applicable
- Comprehensive logging and progress tracking
- Error handling and recovery mechanisms
- Configuration validation
- Pipeline state management
"""

import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import time
import traceback

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class PipelineRunner:
    def __init__(self, config_path: str = "config/pipeline/x02.yaml"):
        """
        Initialize PipelineRunner with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Pipeline stages
        self.stages = {
            1: ("Data Loading", "x_01_data_loader.py"),
            2: ("Feature Engineering", "x_02_feature_engineering.py"),
            3: ("CNN-LSTM Optimization", "x_03_optuna_cnn_lstm.py"),
            4: ("CNN-LSTM Training", "x_04_train_cnn_lstm.py"),
            5: ("Feature Extraction", "x_05_extract_features.py"),
            6: ("XGBoost Optimization", "x_06_optuna_xgboost.py"),
            7: ("XGBoost Training", "x_07_train_xgboost.py"),
            8: ("Model Validation", "x_08_validate_models.py")
        }

        # Pipeline state
        self.pipeline_state = {
            'start_time': None,
            'end_time': None,
            'completed_stages': [],
            'failed_stages': [],
            'stage_results': {},
            'overall_success': False
        }

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def _validate_config(self) -> bool:
        """Validate pipeline configuration."""
        _logger.info("Validating pipeline configuration...")

        required_sections = [
            'data_sources', 'cnn_lstm', 'xgboost', 'technical_indicators',
            'optuna', 'evaluation', 'paths', 'logging', 'hardware'
        ]

        missing_sections = []
        for section in required_sections:
            if section not in self.config:
                missing_sections.append(section)

        if missing_sections:
            _logger.error("Missing configuration sections: %s", missing_sections)
            return False

        # Validate paths
        required_paths = [
            'data_raw', 'data_labeled', 'models_cnn_lstm', 'models_xgboost',
            'results', 'reports', 'visualizations', 'predictions'
        ]

        missing_paths = []
        for path_key in required_paths:
            if path_key not in self.config['paths']:
                missing_paths.append(path_key)

        if missing_paths:
            _logger.error("Missing path configurations: %s", missing_paths)
            return False

        _logger.info("Configuration validation passed")
        return True

    def _create_directories(self):
        """Create necessary directories."""
        _logger.info("Creating pipeline directories...")

        paths_config = self.config['paths']

        for path_key, path_value in paths_config.items():
            path = Path(path_value)
            path.mkdir(parents=True, exist_ok=True)
            _logger.info("Created directory: %s", path)

    def _run_stage_1_data_loading(self) -> Dict[str, Any]:
        """Run Stage 1: Data Loading."""
        _logger.info("=" * 50)
        _logger.info("STAGE 1: Data Loading")
        _logger.info("=" * 50)

        try:
            from x_01_data_loader import DataLoader

            data_loader = DataLoader(str(self.config_path))
            results = data_loader.run()

            # Validate results
            validation_results = data_loader.validate_downloaded_data()
            results['validation'] = validation_results

            _logger.info("Data loading completed: %d successful, %d failed", len(results['success']), len(results['failed']))

            return {
                'success': len(results['failed']) == 0,
                'results': results,
                'message': f"Data loading completed with {len(results['success'])} successful downloads"
            }

        except Exception as e:
            _logger.exception("Data loading failed:")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _run_stage_2_feature_engineering(self) -> Dict[str, Any]:
        """Run Stage 2: Feature Engineering."""
        _logger.info("=" * 50)
        _logger.info("STAGE 2: Feature Engineering")
        _logger.info("=" * 50)

        try:
            from x_02_feature_engineering import FeatureEngineer

            feature_engineer = FeatureEngineer(str(self.config_path))
            results = feature_engineer.run()

            _logger.info("Feature engineering completed: %d successful, %d failed", len(results['success']), len(results['failed']))

            return {
                'success': len(results['failed']) == 0,
                'results': results,
                'message': f"Feature engineering completed with {len(results['success'])} successful processing"
            }

        except Exception as e:
            _logger.exception("Feature engineering failed:")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _run_stage_3_cnn_lstm_optimization(self) -> Dict[str, Any]:
        """Run Stage 3: CNN-LSTM Optimization."""
        _logger.info("=" * 50)
        _logger.info("STAGE 3: CNN-LSTM Optimization")
        _logger.info("=" * 50)

        try:
            from x_03_optuna_cnn_lstm import CNNLSTMOptimizer

            optimizer = CNNLSTMOptimizer(str(self.config_path))
            study = optimizer.run_optimization()

            _logger.info("CNN-LSTM optimization completed: Best trial %s", study.best_trial.number)

            return {
                'success': True,
                'results': {
                    'best_trial': study.best_trial.number,
                    'best_value': study.best_trial.value,
                    'best_params': study.best_trial.params
                },
                'message': f"CNN-LSTM optimization completed with best trial {study.best_trial.number}"
            }

        except Exception as e:
            _logger.exception("CNN-LSTM optimization failed:")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _run_stage_4_cnn_lstm_training(self) -> Dict[str, Any]:
        """Run Stage 4: CNN-LSTM Training."""
        _logger.info("=" * 50)
        _logger.info("STAGE 4: CNN-LSTM Training")
        _logger.info("=" * 50)

        try:
            from x_04_train_cnn_lstm import CNNLSTMTrainer

            trainer = CNNLSTMTrainer(str(self.config_path))
            results = trainer.run()

            if results['success']:
                _logger.info("CNN-LSTM training completed successfully")
                _logger.info("Model saved to: %s", results['model_path'])
                _logger.info("Test metrics: %s", results['metrics'])
            else:
                _logger.error("CNN-LSTM training failed: %s", results['error'])

            return results

        except Exception as e:
            _logger.exception("CNN-LSTM training failed:")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _run_stage_5_feature_extraction(self) -> Dict[str, Any]:
        """Run Stage 5: Feature Extraction."""
        _logger.info("=" * 50)
        _logger.info("STAGE 5: Feature Extraction")
        _logger.info("=" * 50)

        try:
            from x_05_extract_features import FeatureExtractor

            extractor = FeatureExtractor(str(self.config_path))
            results = extractor.run()

            if results['success']:
                _logger.info("Feature extraction completed successfully")
                _logger.info("Summary: %s", results['summary'])
            else:
                _logger.error("Feature extraction failed: %s", results['error'])

            return results

        except Exception as e:
            _logger.exception("Feature extraction failed:")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _run_stage_6_xgboost_optimization(self) -> Dict[str, Any]:
        """Run Stage 6: XGBoost Optimization."""
        _logger.info("=" * 50)
        _logger.info("STAGE 6: XGBoost Optimization")
        _logger.info("=" * 50)

        try:
            from x_06_optuna_xgboost import XGBoostOptimizer

            optimizer = XGBoostOptimizer(str(self.config_path))
            results = optimizer.optimize()

            _logger.info("XGBoost optimization completed successfully")
            _logger.info("Best validation MSE: %.6f", results['best_validation_mse'])

            return {
                'success': True,
                'results': results,
                'message': f"XGBoost optimization completed with best validation MSE: {results['best_validation_mse']:.6f}"
            }

        except Exception as e:
            _logger.exception("XGBoost optimization failed:")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _run_stage_7_xgboost_training(self) -> Dict[str, Any]:
        """Run Stage 7: XGBoost Training."""
        _logger.info("=" * 50)
        _logger.info("STAGE 7: XGBoost Training")
        _logger.info("=" * 50)

        try:
            from x_07_train_xgboost import XGBoostTrainer

            trainer = XGBoostTrainer(str(self.config_path))
            results = trainer.train()

            _logger.info("XGBoost training completed successfully")
            _logger.info("Test MSE: %.6f", results['test']['mse'])
            _logger.info("Test Directional Accuracy: %.4f", results['test']['directional_accuracy'])

            return {
                'success': True,
                'results': results,
                'message': f"XGBoost training completed with test MSE: {results['test']['mse']:.6f}"
            }

        except Exception as e:
            _logger.exception("XGBoost training failed:")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _run_stage_8_model_validation(self) -> Dict[str, Any]:
        """Run Stage 8: Model Validation."""
        _logger.info("=" * 50)
        _logger.info("STAGE 8: Model Validation")
        _logger.info("=" * 50)

        try:
            from x_08_validate_models import ModelValidator

            validator = ModelValidator(str(self.config_path))
            results = validator.validate()

            _logger.info("Model validation completed successfully")
            _logger.info("Validation report generated")

            return {
                'success': True,
                'results': results,
                'message': f"Model validation completed successfully"
            }

        except Exception as e:
            _logger.exception("Model validation failed:")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def run_stage(self, stage_number: int) -> Dict[str, Any]:
        """Run a specific pipeline stage."""
        if stage_number not in self.stages:
            return {
                'success': False,
                'error': f"Invalid stage number: {stage_number}"
            }

        stage_name, stage_file = self.stages[stage_number]

        _logger.info("Running stage %s: %s",stage_number, stage_name)

        # Stage execution mapping
        stage_functions = {
            1: self._run_stage_1_data_loading,
            2: self._run_stage_2_feature_engineering,
            3: self._run_stage_3_cnn_lstm_optimization,
            4: self._run_stage_4_cnn_lstm_training,
            5: self._run_stage_5_feature_extraction,
            6: self._run_stage_6_xgboost_optimization,
            7: self._run_stage_7_xgboost_training,
            8: self._run_stage_8_model_validation
        }

        start_time = time.time()

        try:
            result = stage_functions[stage_number]()
            result['execution_time'] = time.time() - start_time

            if result['success']:
                self.pipeline_state['completed_stages'].append(stage_number)
                _logger.info("Stage {} completed successfully in %.2f seconds", stage_number, result['execution_time'])
            else:
                self.pipeline_state['failed_stages'].append(stage_number)
                _logger.error("Stage %s failed after %.2f seconds", stage_number, result['execution_time'])

            self.pipeline_state['stage_results'][stage_number] = result

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            _logger.exception("Stage %d failed with exception after %.2f seconds:", stage_number, execution_time)

            result = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'execution_time': execution_time
            }

            self.pipeline_state['failed_stages'].append(stage_number)
            self.pipeline_state['stage_results'][stage_number] = result

            return result

    def run_stages(self, stage_numbers: List[int]) -> Dict[str, Any]:
        """Run multiple pipeline stages."""
        _logger.info("Running stages: %s", stage_numbers)

        results = {}

        for stage_number in stage_numbers:
            if stage_number in self.pipeline_state['completed_stages']:
                _logger.info("Stage %s already completed, skipping", stage_number)
                continue

            result = self.run_stage(stage_number)
            results[stage_number] = result

            # Stop if a stage fails (unless configured to continue)
            if not result['success'] and not self.config.get('pipeline', {}).get('continue_on_failure', False):
                _logger.error("Stage %s failed, stopping pipeline", stage_number)
                break

        return results

    def run(self, start_stage: int = 1, end_stage: int = 8, skip_stages: List[int] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline or a subset of stages.

        Args:
            start_stage: First stage to run (inclusive)
            end_stage: Last stage to run (inclusive)
            skip_stages: List of stages to skip

        Returns:
            Pipeline execution results
        """
        _logger.info("=" * 80)
        _logger.info("STARTING CNN-LSTM-XGBOOST PIPELINE")
        _logger.info("=" * 80)

        # Initialize pipeline state
        self.pipeline_state['start_time'] = datetime.now()

        # Validate configuration
        if not self._validate_config():
            return {
                'success': False,
                'error': 'Configuration validation failed'
            }

        # Create directories
        self._create_directories()

        # Determine stages to run
        stages_to_run = list(range(start_stage, end_stage + 1))

        if skip_stages:
            stages_to_run = [s for s in stages_to_run if s not in skip_stages]

        _logger.info("Pipeline stages to run: %s", stages_to_run)

        # Run stages
        results = self.run_stages(stages_to_run)

        # Finalize pipeline state
        self.pipeline_state['end_time'] = datetime.now()
        self.pipeline_state['overall_success'] = len(self.pipeline_state['failed_stages']) == 0

        # Create pipeline summary
        total_time = (self.pipeline_state['end_time'] - self.pipeline_state['start_time']).total_seconds()

        summary = {
            'success': self.pipeline_state['overall_success'],
            'total_time': total_time,
            'completed_stages': self.pipeline_state['completed_stages'],
            'failed_stages': self.pipeline_state['failed_stages'],
            'stage_results': results,
            'pipeline_state': self.pipeline_state
        }

        # Log summary
        _logger.info("=" * 80)
        _logger.info("PIPELINE EXECUTION SUMMARY")
        _logger.info("=" * 80)
        _logger.info("Overall Success: %s", summary['success'])
        _logger.info("Total Execution Time: %.2f seconds", total_time)
        _logger.info("Completed Stages: %s", summary['completed_stages'])
        _logger.info("Failed Stages: %s", summary['failed_stages'])

        if summary['success']:
            _logger.info("Pipeline completed successfully!")
        else:
            _logger.error("Pipeline completed with failures!")

        return summary

def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(description='CNN-LSTM-XGBoost Pipeline Runner')
    parser.add_argument('--config', default='config/pipeline/x02.yaml', help='Configuration file path')
    parser.add_argument('--start-stage', type=int, default=1, help='First stage to run (inclusive)')
    parser.add_argument('--end-stage', type=int, default=8, help='Last stage to run (inclusive)')
    parser.add_argument('--skip-stages', type=str, help='Comma-separated list of stages to skip')
    parser.add_argument('--stages', type=str, help='Comma-separated list of specific stages to run')

    args = parser.parse_args()

    try:
        # Parse stage arguments
        skip_stages = None
        if args.skip_stages:
            skip_stages = [int(s.strip()) for s in args.skip_stages.split(',')]

        start_stage = args.start_stage
        end_stage = args.end_stage

        if args.stages:
            stage_list = [int(s.strip()) for s in args.stages.split(',')]
            start_stage = min(stage_list)
            end_stage = max(stage_list)
            skip_stages = [s for s in range(start_stage, end_stage + 1) if s not in stage_list]

        # Run pipeline
        runner = PipelineRunner(args.config)
        results = runner.run(
            start_stage=start_stage,
            end_stage=end_stage,
            skip_stages=skip_stages
        )

        if results['success']:
            print("Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("Pipeline completed with failures!")
            sys.exit(1)

    except Exception as e:
        _logger.exception("Pipeline execution failed:")
        print(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
