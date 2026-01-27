# Workflows

This document outlines the core workflows for optimizing strategies, extracting configurations, and running comprehensive simulations.

## 1. Optimization Workflow

Use this workflow to find the best parameters for your strategies using historical data.

### Prerequisites
- **Data**: Ensure CSV data files are present in the `data/` directory (e.g., `data/BTCUSDT_1h_....csv`).
- **Configuration**: Configure range of parameters and mixins in `config/optimizer/optimizer.json`.

### Steps

#### 1. Run Optimization
Execute the optimization script to run backtests across parameter ranges.
```powershell
python src/backtester/optimizer/run_optimizer.py
```
- **Output**: JSON result files in `results/`.
- **Note**: This process uses multiprocessing and can take time depending on the configuration.

#### 2. Analyze Results
Aggregate all optimization JSON results into a single CSV for easy comparison.
```powershell
python src/backtester/optimizer/run_json2csv.py
```
- **Output**: `results/optimization_results_summary.csv`.
- **Usage**: Open the CSV to sort by `Total Profit`, `Sharpe Ratio`, etc., and identify the best filenames.

#### 3. Extract Strategy Configuration
Convert a specific optimization result file into a ready-to-use strategy configuration JSON.
```powershell
python src/trading/tools/convert_optimization_result.py "results/optimization_filename.json"
```
- **Arguments**:
  - `input_file`: Path to the optimization result JSON (required).
  - `--output-dir`: Directory to save the config (detault: `config/contracts/instances/strategies`).
  - `--name`: Manual filename for the output (optional).
- **Output**: A new JSON file in `config/contracts/instances/strategies/` containing the best parameters from that run.

---

## 2. Simulation Batch Workflow

Use this workflow to validate selected strategies across multiple datasets (symbols/timeframes) to ensure robustness.

### Prerequisites
- **Strategies**: JSON configuration files must exist in `config/contracts/instances/strategies/`.
  - *Tip*: Use the "Extract Strategy Configuration" step above to populate this.
- **Data**: CSV data files in the `data/` directory.

### Steps

#### 1. Run Batch Simulation
Run every strategy against every data file in the `data/` directory.
```powershell
python src/trading/run_batch_simulation.py
```
- **Process**:
  - Automatically detects all `.csv` files in `data/`.
  - Automatically detects all `.json` strategies in `config/contracts/instances/strategies/`.
  - Runs a simulation for every combination (Cartesian product).
- **Output**: Simulation result JSONs in `results/simulation/`.

#### 2. Analyze Simulation Results
Aggregate simulation results to calculate advanced metrics (Sortino Ratio, Max Consecutive Losses, etc.).
```powershell
python src/trading/tools/analyze_simulation_results.py
```
- **Output**: `results/simulation_analysis.csv`.
- **Metrics Includes**:
  - Total PnL, Win Rate, Profit Factor
  - **Sortino Ratio**: Downside risk-adjusted return.
  - **Max Consecutive Losses**: Streak analysis.
  - **Average Trade Duration**.
