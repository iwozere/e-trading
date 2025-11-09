"""
HMM Training for Market Regime Detection with Parameter Optimization

This module trains Hidden Markov Models to detect market regimes using
raw OHLCV data and technical indicators. The HMM identifies different
market states (e.g., trending, ranging, volatile) that will be used
as features for the LSTM model.

Features:
- Reads raw CSV files from data/raw directory
- Computes technical indicators using TA-Lib
- Optimizes HMM parameters using Optuna or Grid Search
- Maps HMM states to market regimes (bull, bear, sideways)
- Saves trained models and labeled data
- Supports multiple symbols and timeframes

Based on hmm_optimization.py with pipeline integration.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from hmmlearn.hmm import GaussianHMM
from itertools import product
import optuna
import yaml
from pathlib import Path
import pickle
import json
from datetime import datetime
import sys
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class HMMTrainer:
    def __init__(self, config_path: str = "config/pipeline/p01.yaml"):
        """
        Initialize HMM trainer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Generate timestamp for this run
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                # Directory setup
        self.raw_data_dir = Path("data/raw")
        self.labeled_data_dir = Path("data/labeled")
        self.models_dir = Path("src/ml/pipeline/p01_hmm_lstm/models/hmm")
        self.hmm_dir = Path("src/ml/pipeline/p01_hmm_lstm/models/hmm")  # New directory for HMM-specific files

        # Create directories if they don't exist
        self.labeled_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hmm_dir.mkdir(parents=True, exist_ok=True)

        # HMM configuration
        self.n_components = self.config['hmm']['n_components']
        self.covariance_type = self.config['hmm'].get('covariance_type', 'full')
        self.n_iter = self.config['hmm'].get('n_iter', 100)

        # Optimization configuration
        self.search_mode = self.config.get('hmm', {}).get('search_mode', 'optuna')
        self.n_optuna_trials = self.config.get('hmm', {}).get('n_optuna_trials', 50)

        # Get symbols and timeframes from config
        self.symbols = self.config.get('symbols', [])
        self.timeframes = self.config.get('timeframes', [])

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def compute_indicators(self, df: pd.DataFrame, rsi_period: int, atr_period: int,
                          bb_period: int, vol_period: int) -> pd.DataFrame:
        """
        Compute TA-Lib indicators for RSI, ATR, Bollinger Bands, and SMA Volume.

        Args:
            df: DataFrame with OHLCV data
            rsi_period: RSI calculation period
            atr_period: ATR calculation period
            bb_period: Bollinger Bands calculation period
            vol_period: Volume SMA calculation period

        Returns:
            DataFrame with added indicator columns
        """
        # Add log_return if not present
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Compute indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=bb_period,
                                           nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['vol_sma'] = talib.SMA(df['volume'], timeperiod=vol_period)

        return df

    def evaluate_hmm_info_criteria(self, X: np.ndarray, n_states: int) -> Tuple[float, float, GaussianHMM]:
        """
        Fit GaussianHMM and return AIC, BIC, and fitted model.

        Args:
            X: Feature matrix
            n_states: Number of HMM states

        Returns:
            Tuple of (aic, bic, fitted_model)
        """
        model = GaussianHMM(n_components=n_states, covariance_type=self.covariance_type,
                           n_iter=self.n_iter, random_state=42)
        model.fit(X)
        logL = model.score(X)
        k = n_states * (n_states - 1) + n_states * X.shape[1] * 2
        aic = -2 * logL + 2 * k
        bic = -2 * logL + k * np.log(len(X))
        return aic, bic, model

    def map_states_to_regimes(self, df: pd.DataFrame, state_col: str = 'state',
                              ret_col: str = 'log_return', atr_col: str = 'atr') -> Tuple[Dict, pd.DataFrame]:
        """
        Map HMM states to individual regime labels (up to 8 regimes).

        Args:
            df: DataFrame with HMM states
            state_col: Column name for HMM states
            ret_col: Column name for returns
            atr_col: Column name for ATR

        Returns:
            Tuple of (state_to_regime_mapping, updated_dataframe)
        """
        stats = []
        states = sorted(df[state_col].unique())

        for s in states:
            sub = df[df[state_col] == s]
            mean_r = sub[ret_col].mean()
            std_r = sub[ret_col].std()
            mean_atr = sub[atr_col].mean() if atr_col in sub.columns else np.nan
            stats.append((s, mean_r, std_r, mean_atr, len(sub)))

        stats_df = pd.DataFrame(stats, columns=['state', 'mean_r', 'std_r', 'mean_atr', 'count'])

        # Sort by mean return to assign regime labels
        stats_df = stats_df.sort_values('mean_r').reset_index(drop=True)

        # Define regime labels for up to 8 states (numeric values)
        regime_labels = list(range(8))  # [0, 1, 2, 3, 4, 5, 6, 7]

        # Create mapping: state -> regime_number
        mapping = {int(row['state']): regime_labels[i] if i < len(regime_labels) else regime_labels[-1]
                   for i, row in stats_df.iterrows()}

        # Log regime mapping results
        _logger.info("Mapped %d HMM states to individual regimes:", len(states))
        for i, row in stats_df.iterrows():
            state_id = int(row['state'])
            regime_number = mapping[state_id]
            mean_return = row['mean_r']
            _logger.info("  State %d -> Regime %d: Mean Return: %.6f", state_id, regime_number, mean_return)

        df['regime'] = df[state_col].map(mapping)
        return mapping, df

    def plot_results(self, df: pd.DataFrame, file_name: str):
        """
        Plot price with Bollinger Bands, RSI, and log return distribution.

        Args:
            df: DataFrame with regime labels
            file_name: Output file name for the plot
        """
        # Create figure with 4 subplots - large size like in backup
        fig, axes = plt.subplots(4, 1, figsize=(80, 20), sharex=True)
        fig.suptitle(f'HMM Regime Detection - {file_name} (Dataset: {len(df)} points)', fontsize=18)

        # Color mapping for individual regimes (up to 8)
        regime_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink']
        unique_regimes = sorted(df['regime'].unique())
        color_map = {regime: regime_colors[int(regime)] if int(regime) < len(regime_colors) else 'gray'
                    for regime in unique_regimes}

        # Plot 1: Price with Bollinger Bands and regime overlay
        axes[0].set_title('Price with Bollinger Bands and Regimes')

        # Plot price line first (background)
        axes[0].plot(df.index, df['close'], 'k-', alpha=0.3, linewidth=0.5, label='Close Price')

        # Plot Bollinger Bands with more visible lines
        axes[0].plot(df.index, df['bb_upper'], color='orange', alpha=0.8, linewidth=0.2, label='BB Upper')
        #axes[0].plot(df.index, df['bb_middle'], color='grey', alpha=0.8, linewidth=0.2, label='BB Middle')
        axes[0].plot(df.index, df['bb_lower'], color='orange', alpha=0.8, linewidth=0.2, label='BB Lower')

        # Scatter regime points with small dots like in backup
        for regime in unique_regimes:
            mask = df['regime'] == regime
            if mask.any():
                axes[0].scatter(df.index[mask], df['close'][mask],
                              c=color_map[regime], alpha=0.6, s=0.5,
                              label=f'Regime {int(regime)}')

        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: RSI with overbought/oversold lines
        axes[1].set_title('RSI with Overbought/Oversold Levels')
        axes[1].plot(df.index, df['rsi'], color='purple', linewidth=1, label='RSI')
        axes[1].axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[1].axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[1].set_ylabel('RSI')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Log returns with regime overlay
        axes[2].set_title('Log Returns with Regimes')

        # Plot log returns line (removed to avoid confusion with regime dots)
        # axes[2].plot(df.index, df['log_return'], 'k-', alpha=0.3, linewidth=0.5, label='Log Returns')

        # Scatter regime points with small dots
        for regime in unique_regimes:
            mask = df['regime'] == regime
            if mask.any():
                axes[2].scatter(df.index[mask], df['log_return'][mask],
                              c=color_map[regime], alpha=0.6, s=0.5,
                              label=f'{regime.replace("_", " ").title()}')

        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_ylabel('Log Return')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Regime timeline
        axes[3].set_title('Regime Timeline')

        # Create regime mapping for timeline
        regime_mapping = {regime: i for i, regime in enumerate(unique_regimes)}
        regime_values = df['regime'].map(regime_mapping)

        axes[3].plot(df.index, regime_values, 'o', markersize=1, alpha=0.7)
        axes[3].set_ylabel('Regime')
        axes[3].set_xlabel('Time')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_yticks(range(len(unique_regimes)))
        axes[3].set_yticklabels([regime.replace('_', ' ').title() for regime in unique_regimes])

        plt.tight_layout()
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()

    def run_grid_search(self, df: pd.DataFrame, file_base: str) -> Tuple[pd.DataFrame, GaussianHMM, Tuple, List]:
        """
        Run exhaustive grid search over all parameter combinations.

        Args:
            df: Input DataFrame
            file_base: Base name for output files

        Returns:
            Tuple of (best_dataframe, best_model, best_params, report_rows)
        """
        rsi_grid = [7, 14, 21]
        atr_grid = [7, 14, 21]
        bb_grid = [14, 20, 30]
        vol_grid = [14, 20, 30]
        n_states_grid = [2, 3, 4, 5]

        best_bic = np.inf
        best_aic = None
        best_params = None
        best_model = None
        best_df = None
        report_rows = []

        total_combinations = len(rsi_grid) * len(atr_grid) * len(bb_grid) * len(vol_grid) * len(n_states_grid)
        _logger.info("Starting grid search with %d parameter combinations", total_combinations)

        for i, (rsi_p, atr_p, bb_p, vol_p, n_states) in enumerate(product(rsi_grid, atr_grid, bb_grid, vol_grid, n_states_grid)):
            try:
                temp_df = self.compute_indicators(df.copy(), rsi_p, atr_p, bb_p, vol_p).dropna()
                features = temp_df[['rsi', 'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'vol_sma', 'log_return']].values
                aic, bic, model = self.evaluate_hmm_info_criteria(features, n_states)

                report_rows.append({
                    'file': file_base, 'rsi': rsi_p, 'atr': atr_p, 'bb': bb_p,
                    'vol_sma': vol_p, 'n_states': n_states, 'aic': aic, 'bic': bic
                })

                if bic < best_bic:
                    best_bic = bic
                    best_aic = aic
                    best_params = (rsi_p, atr_p, bb_p, vol_p, n_states)
                    best_model = model
                    best_df = temp_df.copy()

                if (i + 1) % 100 == 0:
                    _logger.info("Grid search progress: %d/%d combinations", i + 1, total_combinations)

            except Exception as e:
                _logger.warning("Failed to evaluate parameters (rsi=%d, atr=%d, bb=%d, vol=%d, states=%d): %s",
                               rsi_p, atr_p, bb_p, vol_p, n_states, str(e))
                continue

        # Check if we have a valid result
        if best_bic == np.inf:
            _logger.warning("No valid grid search results found for %s", file_base)
            return None, None, None, report_rows

        return best_df, best_model, best_params, report_rows

    def run_optuna_search(self, df: pd.DataFrame, file_base: str) -> Tuple[pd.DataFrame, GaussianHMM, Tuple, List]:
        """
        Run Optuna optimization to minimize BIC over parameter space.

        Args:
            df: Input DataFrame
            file_base: Base name for output files

        Returns:
            Tuple of (best_dataframe, best_model, best_params, report_rows)
        """
        best_result = {'bic': np.inf}
        report_rows = []

        def objective(trial):
            rsi_p = trial.suggest_int('rsi', 5, 25)
            atr_p = trial.suggest_int('atr', 5, 25)
            bb_p = trial.suggest_int('bb', 10, 30)
            vol_p = trial.suggest_int('vol_sma', 10, 30)
            n_states = trial.suggest_int('n_states', 2, 6)

            try:
                temp_df = self.compute_indicators(df.copy(), rsi_p, atr_p, bb_p, vol_p).dropna()
                features = temp_df[['rsi', 'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'vol_sma', 'log_return']].values

                # Initialize model without warm_start (not supported in hmmlearn)
                model = GaussianHMM(n_components=n_states, covariance_type=self.covariance_type,
                                   n_iter=self.n_iter, random_state=42)

                # Fit the model
                model.fit(features)
                logL = model.score(features)

                # Report progress for pruning
                trial.report(-logL, 1)  # Report negative log likelihood as "loss"
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                k = n_states * (n_states - 1) + n_states * features.shape[1] * 2
                bic = -2 * logL + k * np.log(len(features))
                aic = -2 * logL + 2 * k

                report_rows.append({
                    'file': file_base, 'rsi': rsi_p, 'atr': atr_p, 'bb': bb_p,
                    'vol_sma': vol_p, 'n_states': n_states, 'aic': aic, 'bic': bic
                })

                if bic < best_result['bic']:
                    best_result.update({
                        'bic': bic, 'aic': aic, 'params': (rsi_p, atr_p, bb_p, vol_p, n_states),
                        'model': model, 'df': temp_df.copy()
                    })

                return bic

            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                _logger.warning("Optuna trial failed: %s", str(e))
                return np.inf

        _logger.info("Starting Optuna optimization with %d trials", self.n_optuna_trials)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_optuna_trials)

        # Check if we have a valid result
        if best_result['bic'] == np.inf:
            _logger.warning("No valid Optuna trials completed for %s", file_base)
            return None, None, None, report_rows

        return best_result['df'], best_result['model'], best_result['params'], report_rows

    def process_file(self, file_path: str) -> Dict:
        """
        Process a single CSV file for HMM training.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary with processing results
        """
        try:
            df = pd.read_csv(file_path).dropna().reset_index(drop=True)
            file_base = os.path.splitext(os.path.basename(file_path))[0]

            _logger.info("Processing file: %s", file_base)

            # Validate input data
            required_input_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_input_columns = [col for col in required_input_columns if col not in df.columns]

            if missing_input_columns:
                _logger.error("Missing required input columns for %s: %s", file_base, missing_input_columns)
                return {'success': False, 'file': file_base, 'error': f'Missing input columns: {missing_input_columns}'}

            if len(df) < 100:  # Need minimum data for HMM training
                _logger.warning("Insufficient data for %s: only %d rows", file_base, len(df))
                return {'success': False, 'file': file_base, 'error': f'Insufficient data: {len(df)} rows'}

            # Run optimization
            best_aic = None
            best_bic = None
            if self.search_mode == "grid":
                best_df, best_model, best_params, report_rows = self.run_grid_search(df, file_base)
                # For grid search, we need to get the best AIC/BIC from the report
                if report_rows:
                    best_row = min(report_rows, key=lambda x: x['bic'])
                    best_aic = best_row['aic']
                    best_bic = best_row['bic']
            else:
                best_df, best_model, best_params, report_rows = self.run_optuna_search(df, file_base)
                # For Optuna, we need to get the best AIC/BIC from the report
                if report_rows:
                    best_row = min(report_rows, key=lambda x: x['bic'])
                    best_aic = best_row['aic']
                    best_bic = best_row['bic']

            if best_df is None or best_model is None or best_params is None:
                _logger.warning("No valid model found for %s", file_base)
                return {'success': False, 'file': file_base, 'error': 'No valid model found - optimization failed'}

            # Apply best model
            required_columns = ['rsi', 'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'vol_sma', 'log_return']
            missing_columns = [col for col in required_columns if col not in best_df.columns]

            if missing_columns:
                _logger.error("Missing required columns for %s: %s", file_base, missing_columns)
                return {'success': False, 'file': file_base, 'error': f'Missing columns: {missing_columns}'}

            features_best = best_df[required_columns].values
            best_df['state'] = best_model.predict(features_best)

            # Map states to regimes
            mapping, best_df = self.map_states_to_regimes(best_df)

            # Save labeled data
            output_file = self.labeled_data_dir / f"{file_base}_labeled.csv"
            best_df.to_csv(output_file, index=False)

            # Save model
            model_file = self.models_dir / f"hmm_{file_base}_{self.run_timestamp}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': best_model,
                    'params': best_params,
                    'mapping': mapping,
                    'file_base': file_base
                }, f)

                        # Save plot
            plot_file = self.hmm_dir / f"hmm_{file_base}_{self.run_timestamp}.png"
            self.plot_results(best_df, plot_file)

            # Save optimization report
            report_file = self.hmm_dir / f"hmm_optimization_{file_base}_{self.run_timestamp}.csv"
            pd.DataFrame(report_rows).to_csv(report_file, index=False)

            # Get regime mapping details for JSON output
            regime_details = None
            if best_params[4] > 1:  # n_states > 1
                # Recreate the regime mapping analysis for JSON output
                stats = []
                states = sorted(best_df['state'].unique())
                for s in states:
                    sub = best_df[best_df['state'] == s]
                    mean_r = sub['log_return'].mean()
                    std_r = sub['log_return'].std()
                    mean_atr = sub['atr'].mean() if 'atr' in sub.columns else np.nan
                    stats.append((s, mean_r, std_r, mean_atr, len(sub)))

                stats_df = pd.DataFrame(stats, columns=['state', 'mean_r', 'std_r', 'mean_atr', 'count'])
                regime_details = {
                    'n_states': len(states),
                    'state_statistics': stats_df.to_dict('records'),
                    'regime_mapping_method': 'Sort by mean return',
                    'regime_labels': list(range(8))  # [0, 1, 2, 3, 4, 5, 6, 7]
                }

            # Save best parameters in JSON format
            best_params_dict = {
                'file_base': file_base,
                'optimization_method': self.search_mode,
                'best_parameters': {
                    'rsi_period': best_params[0],
                    'atr_period': best_params[1],
                    'bb_period': best_params[2],
                    'vol_sma_period': best_params[3],
                    'n_states': best_params[4]
                },
                'model_performance': {
                    'aic': best_aic,
                    'bic': best_bic
                },
                'regime_counts': best_df['regime'].value_counts().to_dict(),
                'regime_details': regime_details,
                'total_trials': len(report_rows),
                'timestamp': datetime.now().isoformat(),
                'run_timestamp': self.run_timestamp
            }

            json_file = self.hmm_dir / f"best_params_{file_base}_{self.run_timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(best_params_dict, f, indent=2)

            _logger.info("Successfully processed %s", file_base)

            return {
                'success': True,
                'file': file_base,
                'model_file': str(model_file),
                'labeled_file': str(output_file),
                'plot_file': str(plot_file),
                'report_file': str(report_file),
                'json_file': str(json_file),
                'best_params': best_params,
                'regime_counts': best_df['regime'].value_counts().to_dict()
            }

        except Exception as e:
            _logger.exception("Error processing file %s: %s", file_path, str(e))
            return {'success': False, 'file': os.path.basename(file_path), 'error': str(e)}

    def train_all(self) -> Dict:
        """
        Train HMM models for all files in the raw data directory.

        Returns:
            Dictionary with training results
        """
        _logger.info("Starting HMM training for all files")

        # Find all CSV files in raw data directory
        csv_files = list(self.raw_data_dir.glob("*.csv"))

        if not csv_files:
            _logger.warning("No CSV files found in %s", self.raw_data_dir)
            return {'success': False, 'error': 'No CSV files found'}

        _logger.info("Found %d CSV files to process", len(csv_files))

        results = []
        full_report_rows = []

        for file_path in csv_files:
            result = self.process_file(str(file_path))
            results.append(result)

            if result['success']:
                # Add to full report
                report_file = result['report_file']
                if os.path.exists(report_file):
                    report_df = pd.read_csv(report_file)
                    full_report_rows.extend(report_df.to_dict('records'))

        # Save comprehensive report
        if full_report_rows:
            full_report_file = self.hmm_dir / f"full_hmm_optimization_report_{self.run_timestamp}.csv"
            pd.DataFrame(full_report_rows).to_csv(full_report_file, index=False)
            _logger.info("Saved comprehensive optimization report to %s", full_report_file)

        # Save summary of best parameters for all files
        successful_results = [r for r in results if r['success']]
        if successful_results:
            summary_data = {
                'summary': {
                    'total_files_processed': len(csv_files),
                    'successful_files': len(successful_results),
                    'failed_files': len(results) - len(successful_results),
                    'timestamp': datetime.now().isoformat(),
                    'run_timestamp': self.run_timestamp
                },
                'best_parameters_by_file': {}
            }

            for result in successful_results:
                file_base = result['file']
                summary_data['best_parameters_by_file'][file_base] = {
                    'best_params': result['best_params'],
                    'regime_counts': result['regime_counts'],
                    'model_file': result['model_file'],
                    'plot_file': result['plot_file'],
                    'json_file': result['json_file']
                }

            summary_json_file = self.hmm_dir / f"best_params_summary_{self.run_timestamp}.json"
            with open(summary_json_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            _logger.info("Saved best parameters summary to %s", summary_json_file)

        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        _logger.info("HMM training completed: %d successful, %d failed", successful, failed)

        return {
            'success': True,
            'total_files': len(csv_files),
            'successful': successful,
            'failed': failed,
            'results': results,
            'full_report_file': str(full_report_file) if full_report_rows else None
        }

def main():
    """Main function to run HMM training."""
    trainer = HMMTrainer()
    result = trainer.train_all()

    if result['success']:
        _logger.info("HMM training completed successfully")
        return result
    else:
        _logger.error("HMM training failed: %s", result.get('error', 'Unknown error'))
        return result

if __name__ == "__main__":
    main()
