# Vectorbt Optimization Pipeline Usage Guide

The Vectorbt pipeline is a high-performance, CLI-driven tool for trading strategy optimization, reporting, and promotion.

## 🚀 Getting Started

### Prerequisites
Ensure you are using the project's virtual environment:
```powershell
.venv\Scripts\activate
```

### Basic Command Structure
The pipeline is orchestrated via `src/vectorbt/main.py`:
```powershell
python src/vectorbt/main.py [command] [options]
```

---

## 🛠️ Commands

### 1. `optimize`
Runs the Optuna hyperparameter optimization study.

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--interval` | Timeframe (e.g., `1h`, `4h`, `1d`) | `1h` |
| `--symbols` | Comma-separated tickers | `BTC,ETH,XRP,LTC` |
| `--trials` | Number of optimization trials | `100` |
| `--jobs` | Parallel optimization workers | `6` (auto-capped) |
| `--batch` | **Isolate** each symbol into its own study | `False` |
| `--auto-promote`| Automatically promote best trials to DB | `False` |

#### Example: Portfolio Optimization
Optimizes one unified strategy across multiple coins simultaneously:
```powershell
python src/vectorbt/main.py optimize --symbols BTC,ETH --interval 1h --trials 50
```

#### Example: Batch Optimization
Sequentially optimizes each coin in isolation:
```powershell
python src/vectorbt/main.py optimize --symbols BTC,ETH --interval 1h --batch
```

---

### 2. `promote`
Promotes specific trials from a research study (SQLite) to the production database (PostgreSQL).

| Argument | Description | Default |
| :--- | :--- | :--- |
| `study_name` | Name of the study folder | (Required) |
| `--top-n` | Number of top trials to promote | `5` |
| `--min-calmar` | Minimum Calmar ratio threshold | `1.5` |
| `--max-drawdown`| Maximum drawdown limit | `0.4` |

#### Example:
```powershell
python src/vectorbt/main.py promote optimization_BTC_1h --top-n 3 --min-calmar 2.0
```

---

### 3. `list-studies`
Lists all available optimization studies found in the database.
```powershell
python src/vectorbt/main.py list-studies
```

---

## 📂 Result Organization

All outputs are saved to the `results/vectorbt/` directory using a hierarchical structure:

```text
results/vectorbt/
├── <symbol>/                   # e.g., BTC or BTC-ETH (for portfolio)
│   └── <interval>/             # e.g., 1h or 4h
│       ├── optimization.db     # SQLite study file
│       ├── study_summary.md    # Markdown results overview
│       ├── opt_history.html    # Optimization convergence chart
│       └── reports/            # Detailed trial reports
│           ├── trial_42.json   # Full metrics and trade list
│           └── trial_42_dashboard.html
```

## 📊 Viewing Reports
1.  **JSON**: Contains `deposit_amount`, `total_profit_after_commission`, and a full list of `trades`.
2.  **HTML Dashboard**: Open in any browser to see interactive equity curves and drawdown charts.
3.  **Study Summary**: A quick leaderboard of the best parameters found.
