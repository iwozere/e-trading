"""
Complete HMM-LSTM Trading Pipeline Runner

This script orchestrates the entire HMM-LSTM trading pipeline from data loading
to model validation. It runs all pipeline stages in sequence and provides
comprehensive logging and error handling.

Pipeline Stages:
1. Data Loading (x_01_data_loader.py)
2. Data Preprocessing (x_02_preprocess.py)
3. HMM Training (x_03_train_hmm.py)
4. HMM Application (x_04_apply_hmm.py)
5. Indicator Optimization (x_05_optuna_indicators.py)
6. LSTM Optimization (x_06_optuna_lstm.py)
7. LSTM Training (x_07_train_lstm.py)
8. Model Validation (x_08_validate_lstm.py)

Usage:
    python run_pipeline.py [options]

Options:
    --config: Path to configuration file (default: config/pipeline/x01.yaml)
    --skip-stages: Comma-separated list of stages to skip (e.g., "1,2,3")
    --symbols: Comma-separated list of symbols to process (overrides config)
    --timeframes: Comma-separated list of timeframes to process (overrides config)
    --parallel: Run stages in parallel where possible
"""

import argparse
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
import yaml
import importlib.util
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'pipeline_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self, config_path: str = "config/pipeline/x01.yaml"):
        """
        Initialize pipeline runner.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.pipeline_dir = Path(__file__).parent

        # Pipeline stages with their respective modules
        self.stages = {
            1: {
                'name': 'Data Loading',
                'module': 'x_01_data_loader',
                'description': 'Download OHLCV data for symbols and timeframes'
            },
            2: {
                'name': 'Data Preprocessing',
                'module': 'x_02_preprocess',
                'description': 'Add features, indicators, and normalize data'
            },
            3: {
                'name': 'HMM Training',
                'module': 'x_03_train_hmm',
                'description': 'Train Hidden Markov Models for regime detection'
            },
            4: {
                'name': 'HMM Application',
                'module': 'x_04_apply_hmm',
                'description': 'Apply HMM models to label data with regimes'
            },
            5: {
                'name': 'Indicator Optimization',
                'module': 'x_05_optuna_indicators',
                'description': 'Optimize technical indicator parameters with Optuna'
            },
            6: {
                'name': 'LSTM Optimization',
                'module': 'x_06_optuna_lstm',
                'description': 'Optimize LSTM hyperparameters with Optuna'
            },
            7: {
                'name': 'LSTM Training',
                'module': 'x_07_train_lstm',
                'description': 'Train LSTM models with optimized parameters'
            },
            8: {
                'name': 'Model Validation',
                'module': 'x_08_validate_lstm',
                'description': 'Validate models and generate reports'
            }
        }

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    def _load_module(self, module_name: str):
        """Dynamically load a pipeline module."""
        module_path = self.pipeline_dir / f"{module_name}.py"

        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def run_stage(self, stage_num: int) -> Dict:
        """
        Run a specific pipeline stage.

        Args:
            stage_num: Stage number to run

        Returns:
            Dict with stage execution results
        """
        if stage_num not in self.stages:
            raise ValueError(f"Invalid stage number: {stage_num}")

        stage = self.stages[stage_num]
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE {stage_num}: {stage['name']}")
        logger.info(f"Description: {stage['description']}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        try:
            # Load and execute module
            module = self._load_module(stage['module'])

            # Run the main function of the module
            if hasattr(module, 'main'):
                result = module.main()
            else:
                logger.warning(f"No main() function found in {stage['module']}")
                result = None

            execution_time = time.time() - start_time

            logger.info(f"✓ Stage {stage_num} completed successfully in {execution_time:.2f} seconds")

            return {
                'stage': stage_num,
                'name': stage['name'],
                'success': True,
                'execution_time': execution_time,
                'result': result
            }

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Stage {stage_num} failed: {str(e)}"
            logger.error(error_msg)

            return {
                'stage': stage_num,
                'name': stage['name'],
                'success': False,
                'execution_time': execution_time,
                'error': error_msg
            }

    def run_pipeline(self, skip_stages: Optional[List[int]] = None,
                    symbols: Optional[List[str]] = None,
                    timeframes: Optional[List[str]] = None) -> Dict:
        """
        Run the complete pipeline.

        Args:
            skip_stages: List of stage numbers to skip
            symbols: Override symbols from config
            timeframes: Override timeframes from config

        Returns:
            Dict with complete pipeline results
        """
        logger.info("Starting HMM-LSTM Trading Pipeline")
        logger.info(f"Configuration: {self.config_path}")

        if symbols:
            logger.info(f"Override symbols: {symbols}")
            # Temporarily update config
            original_symbols = self.config['symbols']
            self.config['symbols'] = symbols

        if timeframes:
            logger.info(f"Override timeframes: {timeframes}")
            # Temporarily update config
            original_timeframes = self.config['timeframes']
            self.config['timeframes'] = timeframes

        skip_stages = skip_stages or []
        pipeline_start_time = time.time()

        results = {
            'pipeline_start_time': datetime.now().isoformat(),
            'config': self.config,
            'stages_to_run': [s for s in self.stages.keys() if s not in skip_stages],
            'stages_skipped': skip_stages,
            'stage_results': [],
            'overall_success': True
        }

        # Run each stage
        for stage_num in sorted(self.stages.keys()):
            if stage_num in skip_stages:
                logger.info(f"Skipping Stage {stage_num}: {self.stages[stage_num]['name']}")
                continue

            stage_result = self.run_stage(stage_num)
            results['stage_results'].append(stage_result)

            if not stage_result['success']:
                results['overall_success'] = False
                logger.error(f"Pipeline failed at Stage {stage_num}")

                # Ask user if they want to continue
                response = input(f"\nStage {stage_num} failed. Continue with remaining stages? (y/n): ")
                if response.lower() != 'y':
                    logger.info("Pipeline execution stopped by user")
                    break

        # Calculate total execution time
        total_time = time.time() - pipeline_start_time
        results['total_execution_time'] = total_time
        results['pipeline_end_time'] = datetime.now().isoformat()

        # Restore original config if overridden
        if symbols:
            self.config['symbols'] = original_symbols
        if timeframes:
            self.config['timeframes'] = original_timeframes

        # Log summary
        self._log_pipeline_summary(results)

        return results

    def _log_pipeline_summary(self, results: Dict) -> None:
        """Log pipeline execution summary."""
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info(f"{'='*60}")

        successful_stages = [r for r in results['stage_results'] if r['success']]
        failed_stages = [r for r in results['stage_results'] if not r['success']]

        logger.info(f"Total execution time: {results['total_execution_time']:.2f} seconds")
        logger.info(f"Stages completed: {len(successful_stages)}")
        logger.info(f"Stages failed: {len(failed_stages)}")
        logger.info(f"Overall success: {results['overall_success']}")

        if successful_stages:
            logger.info("\nSuccessful stages:")
            for stage in successful_stages:
                logger.info(f"  ✓ Stage {stage['stage']}: {stage['name']} ({stage['execution_time']:.2f}s)")

        if failed_stages:
            logger.info("\nFailed stages:")
            for stage in failed_stages:
                logger.info(f"  ✗ Stage {stage['stage']}: {stage['name']} - {stage['error']}")

        if results['overall_success']:
            logger.info("\n🎉 Pipeline completed successfully!")
            logger.info("Check the results/ and reports/ directories for outputs.")
        else:
            logger.warning("\n⚠️  Pipeline completed with errors.")

        logger.info(f"{'='*60}")

    def list_stages(self) -> None:
        """List all available pipeline stages."""
        print("\nHMM-LSTM Trading Pipeline Stages:")
        print("=" * 50)

        for stage_num, stage in self.stages.items():
            print(f"{stage_num}. {stage['name']}")
            print(f"   Module: {stage['module']}")
            print(f"   Description: {stage['description']}")
            print()

    def validate_requirements(self) -> bool:
        """Validate that all required components are available."""
        logger.info("Validating pipeline requirements...")

        # Check if all stage modules exist
        missing_modules = []
        for stage_num, stage in self.stages.items():
            module_path = self.pipeline_dir / f"{stage['module']}.py"
            if not module_path.exists():
                missing_modules.append(f"Stage {stage_num}: {stage['module']}.py")

        if missing_modules:
            logger.error("Missing pipeline modules:")
            for module in missing_modules:
                logger.error(f"  - {module}")
            return False

        # Check required directories
        required_dirs = [
            self.config['paths']['data_raw'],
            self.config['paths']['data_processed'],
            self.config['paths']['data_labeled'],
            self.config['paths']['results'],
            self.config['paths']['reports']
        ]

        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        logger.info("✓ All requirements validated")
        return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="HMM-LSTM Trading Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                           # Run complete pipeline
  python run_pipeline.py --skip-stages 1,2,3      # Skip first 3 stages
  python run_pipeline.py --symbols BTCUSDT,ETHUSDT # Process specific symbols
  python run_pipeline.py --list-stages             # List all stages
        """
    )

    parser.add_argument('--config', type=str, default='config/pipeline/x01.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-stages', type=str,
                       help='Comma-separated list of stages to skip (e.g., "1,2,3")')
    parser.add_argument('--symbols', type=str,
                       help='Comma-separated list of symbols (overrides config)')
    parser.add_argument('--timeframes', type=str,
                       help='Comma-separated list of timeframes (overrides config)')
    parser.add_argument('--list-stages', action='store_true',
                       help='List all pipeline stages and exit')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate requirements without running pipeline')

    args = parser.parse_args()

    try:
        # Initialize pipeline runner
        runner = PipelineRunner(args.config)

        # Handle list stages
        if args.list_stages:
            runner.list_stages()
            return

        # Validate requirements
        if not runner.validate_requirements():
            logger.error("Requirements validation failed")
            sys.exit(1)

        if args.validate_only:
            logger.info("Requirements validation completed successfully")
            return

        # Parse arguments
        skip_stages = []
        if args.skip_stages:
            skip_stages = [int(s.strip()) for s in args.skip_stages.split(',')]

        symbols = None
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',')]

        timeframes = None
        if args.timeframes:
            timeframes = [s.strip() for s in args.timeframes.split(',')]

        # Run pipeline
        results = runner.run_pipeline(
            skip_stages=skip_stages,
            symbols=symbols,
            timeframes=timeframes
        )

        # Exit with appropriate code
        sys.exit(0 if results['overall_success'] else 1)

    except KeyboardInterrupt:
        logger.info("\nPipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
