"""
Complete HMM-LSTM Trading Pipeline Runner

This script orchestrates the entire HMM-LSTM trading pipeline from data loading
to model validation. It runs all pipeline stages in sequence and provides
comprehensive logging and error handling.

Pipeline Stages:
1. Data Loading (x_01_data_loader.py)
2. HMM Training (x_02_train_hmm.py) - with parameter optimization
3. HMM Application (x_03_apply_hmm.py)
4. Indicator Optimization (x_04_optuna_indicators.py)
5. LSTM Optimization (x_05_optuna_lstm.py)
6. LSTM Training (x_06_train_lstm.py)
7. Model Validation (x_07_validate_lstm.py)

Usage:
    python run_pipeline.py [options]

Options:
    --config: Path to configuration file (default: config/pipeline/p01.yaml)
    --skip-stages: Comma-separated list of stages to skip (e.g., "1,2,3")
    --symbols: Comma-separated list of symbols to process (overrides config)
    --timeframes: Comma-separated list of timeframes to process (overrides config)
    --parallel: Run stages in parallel where possible
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import yaml
import importlib.util
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class PipelineRunner:
    def __init__(self, config_path: str = "config/pipeline/p01.yaml"):
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
                'description': 'Download OHLCV data for symbols and timeframes',
                'critical': True  # Critical - needed for all subsequent stages
            },
            2: {
                'name': 'HMM Training',
                'module': 'x_02_train_hmm',
                'description': 'Train Hidden Markov Models for regime detection with parameter optimization',
                'critical': True  # Critical - needed for HMM application
            },
            3: {
                'name': 'HMM Application',
                'module': 'x_03_apply_hmm',
                'description': 'Apply HMM models to label data with regimes',
                'critical': True  # Critical - needed for LSTM training
            },
            4: {
                'name': 'Indicator Optimization',
                'module': 'x_04_optuna_indicators',
                'description': 'Optimize technical indicator parameters with Optuna',
                'critical': False  # Optional - can proceed without optimization
            },
            5: {
                'name': 'LSTM Optimization',
                'module': 'x_05_optuna_lstm',
                'description': 'Optimize LSTM hyperparameters with Optuna',
                'critical': False  # Optional - can proceed with default parameters
            },
            6: {
                'name': 'LSTM Training',
                'module': 'x_06_train_lstm',
                'description': 'Train LSTM models with optimized parameters',
                'critical': True  # Critical - main model training
            },
            7: {
                'name': 'Model Validation',
                'module': 'x_07_validate_lstm',
                'description': 'Validate models and generate reports',
                'critical': False  # Optional - validation can be skipped
            }
        }

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
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
        _logger.info("%s", "=" * 60)
        _logger.info("STAGE %d: %s", stage_num, stage['name'])
        _logger.info("Description: %s", stage['description'])
        _logger.info("%s", "=" * 60)
        _logger.info("Starting stage %d: %s", stage_num, stage['name'])

        start_time = time.time()

        try:
            # Load and execute module
            module = self._load_module(stage['module'])

            # Run the main function of the module
            if hasattr(module, 'main'):
                result = module.main()
            else:
                _logger.warning("No main() function found in %s", stage['module'])
                result = None

            execution_time = time.time() - start_time

            _logger.info("Stage %d completed successfully in %.2f seconds", stage_num, execution_time)

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
            _logger.exception(error_msg)

            return {
                'stage': stage_num,
                'name': stage['name'],
                'success': False,
                'execution_time': execution_time,
                'error': error_msg
            }

    def run_pipeline(self, skip_stages: Optional[List[int]] = None,
                    symbols: Optional[List[str]] = None,
                    timeframes: Optional[List[str]] = None,
                    fail_fast: bool = True,
                    continue_on_optional_failures: bool = False) -> Dict:
        """
        Run the complete pipeline.

        Args:
            skip_stages: List of stage numbers to skip
            symbols: Override symbols from config (legacy support)
            timeframes: Override timeframes from config (legacy support)
            fail_fast: If True, stop immediately on critical stage failures
            continue_on_optional_failures: If True, continue even if optional stages fail

        Returns:
            Dict with complete pipeline results
        """
        _logger.info("Starting HMM-LSTM Trading Pipeline")
        _logger.info("Configuration: %s", self.config_path)
        _logger.info("Fail-fast mode: %s", "ENABLED" if fail_fast else "DISABLED")
        _logger.info("Continue on optional failures: %s", "ENABLED" if continue_on_optional_failures else "DISABLED")

        # Check for multi-provider configuration
        if 'data_sources' in self.config:
            _logger.info("Using multi-provider configuration")
            self._log_multi_provider_config()
        else:
            _logger.info("Using legacy configuration")
            if symbols:
                _logger.info("Override symbols: %s", symbols)
                # Temporarily update config
                original_symbols = self.config['symbols']
                self.config['symbols'] = symbols

            if timeframes:
                _logger.info("Override timeframes: %s", timeframes)
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
            'overall_success': True,
            'failed_critical_stages': [],
            'failed_optional_stages': [],
            'multi_provider': 'data_sources' in self.config
        }

        # Run each stage
        for stage_num in sorted(self.stages.keys()):
            if stage_num in skip_stages:
                _logger.info("Skipping Stage %d: %s", stage_num, self.stages[stage_num]['name'])
                continue

            stage = self.stages[stage_num]
            stage_result = self.run_stage(stage_num)
            results['stage_results'].append(stage_result)

            if not stage_result['success']:
                if stage['critical']:
                    # Critical stage failed
                    results['failed_critical_stages'].append(stage_result)
                    results['overall_success'] = False

                    _logger.error("CRITICAL STAGE FAILED: Stage %d (%s)", stage_num, stage['name'])
                    _logger.error("Error: %s", stage_result['error'])

                    if fail_fast:
                        _logger.error("Pipeline stopped due to critical stage failure (fail-fast mode enabled)")
                        _logger.error("Fix the issue and restart the pipeline from stage %d", stage_num)
                        break
                    else:
                        _logger.warning("Critical stage failed but continuing (fail-fast mode disabled)")
                        # Ask user if they want to continue
                        response = input(f"\nCritical stage {stage_num} failed. Continue with remaining stages? (y/n): ")
                        if response.lower() != 'y':
                            _logger.info("Pipeline execution stopped by user")
                            break
                else:
                    # Optional stage failed
                    results['failed_optional_stages'].append(stage_result)

                    _logger.warning("OPTIONAL STAGE FAILED: Stage %d (%s)", stage_num, stage['name'])
                    _logger.warning("Error: %s", stage_result['error'])

                    if not continue_on_optional_failures:
                        _logger.warning("Optional stage failed and continue_on_optional_failures=False")
                        # Ask user if they want to continue
                        response = input(f"\nOptional stage {stage_num} failed. Continue with remaining stages? (y/n): ")
                        if response.lower() != 'y':
                            _logger.info("Pipeline execution stopped by user")
                            break
                    else:
                        _logger.info("Optional stage failed but continuing (continue_on_optional_failures=True)")

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
        _logger.info("\n%s", "="*60)
        _logger.info("PIPELINE EXECUTION SUMMARY")
        _logger.info("%s", "="*60)

        successful_stages = [r for r in results['stage_results'] if r['success']]
        failed_critical = results.get('failed_critical_stages', [])
        failed_optional = results.get('failed_optional_stages', [])

        _logger.info("Total execution time: %.2f seconds", results['total_execution_time'])
        _logger.info("Stages completed: %d", len(successful_stages))
        _logger.info("Critical stages failed: %d", len(failed_critical))
        _logger.info("Optional stages failed: %d", len(failed_optional))
        _logger.info("Overall success: %s", results['overall_success'])

        if successful_stages:
            _logger.info("\nSuccessful stages:")
            for stage in successful_stages:
                stage_info = self.stages[stage['stage']]
                criticality = "CRITICAL" if stage_info['critical'] else "OPTIONAL"
                _logger.info("  [OK] Stage %d (%s): %s (%.2fs)",
                           stage['stage'], criticality, stage['name'], stage['execution_time'])

        if failed_critical:
            _logger.error("\nFailed CRITICAL stages:")
            for stage in failed_critical:
                _logger.error("  [CRITICAL FAILED] Stage %d: %s - %s",
                            stage['stage'], stage['name'], stage['error'])

        if failed_optional:
            _logger.warning("\nFailed OPTIONAL stages:")
            for stage in failed_optional:
                _logger.warning("  [OPTIONAL FAILED] Stage %d: %s - %s",
                              stage['stage'], stage['name'], stage['error'])

        if results['overall_success']:
            _logger.info("\n[SUCCESS] Pipeline completed successfully!")
            _logger.info("Check the results/ and reports/ directories for outputs.")
        else:
            if failed_critical:
                _logger.error("\n[FAILURE] Pipeline failed due to critical stage failures.")
                _logger.error("Fix the issues and restart the pipeline from the failed stage.")
            else:
                _logger.warning("\n[PARTIAL SUCCESS] Pipeline completed with optional stage failures.")

        _logger.info("%s", "="*60)

    def list_stages(self) -> None:
        """List all available pipeline stages."""
        print("\nHMM-LSTM Trading Pipeline Stages:")
        print("=" * 60)

        for stage_num, stage in self.stages.items():
            criticality = "[CRITICAL]" if stage['critical'] else "[OPTIONAL]"
            print(f"{stage_num}. {stage['name']} {criticality}")
            print(f"   Module: {stage['module']}")
            print(f"   Description: {stage['description']}")
            print()

    def _log_multi_provider_config(self) -> None:
        """
        Log the multi-provider configuration details.
        """
        data_sources = self.config['data_sources']
        _logger.info("Multi-provider configuration:")

        total_symbols = 0
        total_timeframes = 0

        for provider, config in data_sources.items():
            symbols = config.get('symbols', [])
            timeframes = config.get('timeframes', [])
            total_symbols += len(symbols)
            total_timeframes += len(timeframes)

            _logger.info("  %s:", provider)
            _logger.info("    Symbols: %s", symbols)
            _logger.info("    Timeframes: %s", timeframes)
            _logger.info("    Total combinations: %d", len(symbols) * len(timeframes))

        _logger.info("Total symbols across all providers: %d", total_symbols)
        _logger.info("Total timeframes across all providers: %d", total_timeframes)

    def validate_requirements(self) -> bool:
        """Validate that all required components are available."""
        _logger.info("Validating pipeline requirements...")

        # Check if all stage modules exist
        missing_modules = []
        for stage_num, stage in self.stages.items():
            module_path = self.pipeline_dir / f"{stage['module']}.py"
            if not module_path.exists():
                missing_modules.append(f"Stage {stage_num}: {stage['module']}.py")

        if missing_modules:
            _logger.error("Missing pipeline modules:")
            for module in missing_modules:
                _logger.error("  - %s", module)
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

        _logger.info("[OK] All requirements validated")
        return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="HMM-LSTM Trading Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                           # Run complete pipeline with fail-fast
  python run_pipeline.py --skip-stages 1,2,3      # Skip first 3 stages
  python run_pipeline.py --symbols BTCUSDT,ETHUSDT # Process specific symbols
  python run_pipeline.py --list-stages             # List all stages
  python run_pipeline.py --no-fail-fast            # Disable fail-fast mode
  python run_pipeline.py --continue-on-optional-failures  # Continue on optional failures
        """
    )

    parser.add_argument('--config', type=str, default='config/pipeline/p01.yaml',
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
    parser.add_argument('--no-fail-fast', action='store_true',
                       help='Disable fail-fast mode (continue on critical failures)')
    parser.add_argument('--continue-on-optional-failures', action='store_true',
                       help='Continue pipeline even if optional stages fail')

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
            _logger.error("Requirements validation failed")
            sys.exit(1)

        if args.validate_only:
            _logger.info("Requirements validation completed successfully")
            return

        # Parse arguments
        skip_stages = []
        if args.skip_stages:
            skip_stages = [int(s.strip()) for s in args.skip_stages.split(',')]
        else:
            skip_stages = [1,2]

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
            timeframes=timeframes,
            fail_fast=not args.no_fail_fast,
            continue_on_optional_failures=args.continue_on_optional_failures
        )

        # Exit with appropriate code
        sys.exit(0 if results['overall_success'] else 1)

    except KeyboardInterrupt:
        _logger.info("Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        _logger.exception("Pipeline execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
