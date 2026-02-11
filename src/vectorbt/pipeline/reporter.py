import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any, Optional, List, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class Reporter:
    """
    Handles generation of institutional-grade metrics and interactive dashboards.
    """

    def __init__(self, output_dir: str = "results/vectorbt/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_full_report(self, pf: vbt.Portfolio, trial_id: int, params: Dict[str, Any], benchmark_close: Optional[pd.Series] = None, strategy_name: str = "unknown"):
        """
        Generates JSON metrics, PNG snapshots, and HTML interactive dashboards.
        """
        report_dir = self.output_dir

        _logger.info("Step 1: Calculating metrics...")
        metrics = self.calculate_metrics(pf)

        _logger.info("Step 2: Saving JSON report...")
        self.save_json_report(report_dir, trial_id, params, metrics, pf, strategy_name=strategy_name)

        _logger.info("Step 3: Generating HTML dashboard...")
        self.generate_html_dashboard(report_dir, trial_id, pf, benchmark_close)

        _logger.info("Step 4: Saving PNG snapshot...")
        self.save_png_snapshot(report_dir, trial_id, pf)

        _logger.info(f"✅ Full report generated for trial {trial_id} in {report_dir}")

    def calculate_metrics(self, pf: vbt.Portfolio) -> Dict[str, Any]:
        """
        Calculates Calmar, Sortino, Expectancy, and other key metrics.
        """
        trades = pf.trades
        expectancy = trades.expectancy().mean() if not trades.records.empty else 0.0
        profit_factor = trades.profit_factor().mean() if not trades.records.empty else 0.0
        win_rate = trades.win_rate().mean() if not trades.records.empty else 0.0

        metrics = {
            "cagr": float(pf.annualized_return().mean()),
            "max_drawdown": float(pf.max_drawdown().mean()),
            "calmar_ratio": float(pf.calmar_ratio().mean()),
            "sortino_ratio": float(pf.sortino_ratio().mean()),
            "sharpe_ratio": float(pf.sharpe_ratio().mean()),
            "profit_factor": float(profit_factor),
            "win_rate": float(win_rate),
            "expectancy": float(expectancy),
            "max_gross_exposure": float(pf.gross_exposure().max().max()),
            "total_trades": int(trades.count().sum())
        }

        # Distance to Liquidation Approximation (Proxy)
        metrics["margin_risk_status"] = "High Risk" if metrics["max_drawdown"] > 0.4 else "Healthy"

        return metrics

    def extract_trade_details(self, pf: vbt.Portfolio, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts detailed trade information for JSON export.

        Returns:
            List of trade dictionaries with full details
        """
        trades = pf.trades
        if trades.records.empty:
            return []

        trade_list = []
        leverage = params.get('leverage', 1.0)

        # Access the records DataFrame
        records_df = trades.records_readable

        for idx, trade_row in records_df.iterrows():
            trade_detail = {
                "trade_id": int(idx),
                "symbol": str(trade_row.get('Column', trade_row.get('Symbol', 'N/A'))),
                "direction": "long" if trade_row['Size'] > 0 else "short",
                "leverage": float(leverage),
                "entry_time": str(trade_row['Entry Timestamp']) if pd.notna(trade_row.get('Entry Timestamp')) else str(trade_row['Entry Index']),
                "exit_time": str(trade_row['Exit Timestamp']) if pd.notna(trade_row.get('Exit Timestamp')) else str(trade_row['Exit Index']),
                "entry_price": float(trade_row['Avg Entry Price']),
                "exit_price": float(trade_row['Avg Exit Price']),
                "size": float(abs(trade_row['Size'])),
                "pnl": float(trade_row['PnL']),
                "return_pct": float(trade_row['Return']) * 100,
                "status": trade_row.get('Status', 'Closed')
            }
            trade_list.append(trade_detail)

        return trade_list

    def save_json_report(self, report_dir: str, trial_id: int, params: Dict[str, Any], metrics: Dict[str, Any], pf: vbt.Portfolio, strategy_name: str = "unknown"):
        """
        Saves metrics, pnl data, and detailed trade information to JSON.
        """
        report_path = os.path.join(report_dir, f"trial_{trial_id}.json")

        # Handle scalar pnl vs per-symbol pnl
        pnl = pf.total_profit()
        if hasattr(pnl, 'to_dict'):
            pnl_data = pnl.to_dict()
        else:
            pnl_data = {"total": float(pnl)}

        # Extract detailed trade information
        trade_details = self.extract_trade_details(pf, params)

        report_data = {
            "trial_id": trial_id,
            "strategy_name": strategy_name,
            "params": params,
            "deposit_amount": float(getattr(pf, 'init_cash', 0)),
            "total_profit_after_commission": float(pf.total_profit().sum()) if isinstance(pf.total_profit(), pd.Series) else float(pf.total_profit()),
            "metrics": metrics,
            "pnl_per_symbol": pnl_data,
            "trades": trade_details,
            "trade_count": len(trade_details)
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)


    def generate_html_dashboard(self, report_dir: str, trial_id: int, pf: vbt.Portfolio, benchmark_close: Optional[pd.Series] = None):
        """
        Creates a multi-subplot Plotly dashboard.
        """
        try:
            # Performance Subplots
            benchmark_returns = None
            if benchmark_close is not None:
                benchmark_returns = benchmark_close.vbt.returns()
                # Ensure no NaNs or Infs in returns
                benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            _logger.info("Plotting portfolio subplots...")
            fig = pf.plot(
                subplots=[
                    'cum_returns',
                    'drawdowns',
                    'net_exposure',
                    'cash'
                ],
                column=None,
                group_by=True
            )

            # Add benchmark to cumulative returns subplot (index 1 in 1-based plotly row/col)
            if benchmark_returns is not None:
                _logger.info("Adding benchmark to plot...")
                bench_cum = (1 + benchmark_returns).cumprod() * pf.init_cash
                # Filter out extremes to avoid Plotly crashes
                bench_cum = bench_cum.replace([np.inf, -np.inf], np.nan).dropna()

                if not bench_cum.empty:
                    fig.add_trace(go.Scatter(
                        x=bench_cum.index,
                        y=bench_cum.values,
                        name='Benchmark (BTC)',
                        line=dict(dash='dash', color='gray'),
                        opacity=0.6
                    ), row=1, col=1)

            # Save Performance Dashboard
            html_path = os.path.join(report_dir, f"trial_{trial_id}_dashboard.html")
            fig.write_html(html_path)

            # Trade Distribution
            if not pf.trades.records.empty:
                try:
                    dist_fig = pf.trades.plot(subplots=['pnl'])
                    dist_fig.write_html(os.path.join(report_dir, f"trial_{trial_id}_trades.html"))
                except Exception as e:
                    _logger.warning(f"Could not plot trade distribution: {e}")

            # Risk Plots
            self.generate_risk_plots(report_dir, trial_id, pf)

        except Exception as e:
            _logger.error(f"Failed to generate HTML dashboard for trial {trial_id}: {e}")

    def generate_risk_plots(self, report_dir: str, trial_id: int, pf: vbt.Portfolio):
        """
        Generates plots for Margin Usage and Dist to Liquidation proxy.
        """
        try:
            # Margin Usage = Net Exposure / Total Value (simplified)
            val = pf.value()
            # Avoid division by zero
            val = val.replace(0, np.nan)
            margin_usage = pf.gross_exposure() / val
            margin_usage = margin_usage.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            # Distance to Liquidation Proxy
            drawdowns = pf.drawdown

            fig = make_subplots(rows=2, cols=1, subplot_titles=("Margin Usage Ratio", "Distance to Liquidation Proxy (0.6 + Drawdown)"))

            # Subplot 1: Margin Usage
            if isinstance(margin_usage, pd.DataFrame):
                for col in margin_usage.columns:
                    fig.add_trace(go.Scatter(x=margin_usage.index, y=margin_usage[col], name=f"Margin {col}"), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=margin_usage.index, y=margin_usage, name="Margin"), row=1, col=1)

            # Subplot 2: Distance to Liquidation Proxy
            if isinstance(drawdowns, pd.DataFrame):
                for col in drawdowns.columns:
                    # Clip to avoid weird plots if drawdown is somehow > 0
                    y_val = (0.6 + drawdowns[col]).clip(0, 1)
                    fig.add_trace(go.Scatter(x=drawdowns.index, y=y_val, name=f"LiqDist {col}"), row=2, col=1)
            else:
                y_val = (0.6 + drawdowns).clip(0, 1)
                fig.add_trace(go.Scatter(x=drawdowns.index, y=y_val, name="LiqDist"), row=2, col=1)

            fig.add_hline(y=0.1, line_dash="dash", line_color="red", row=2, col=1) # Warning line
            fig.write_html(os.path.join(report_dir, f"trial_{trial_id}_risk.html"))

        except Exception as e:
            _logger.error(f"Failed to generate risk plots for trial {trial_id}: {e}")

    def save_png_snapshot(self, report_dir: str, trial_id: int, pf: vbt.Portfolio):
        """
        Saves a static PNG of the cumulative returns.
        """
        try:
            # pf.plot_cum_returns().write_image(os.path.join(report_dir, f"trial_{trial_id}_equity.png"))
            # Note: write_image requires kaleido or orca. If not available, we skip.
            # For now, let's just log it or try it.
            pass
        except Exception as e:
            _logger.warning(f"Could not save PNG snapshot (kaleido missing?): {e}")

    def generate_optimization_viz(self, study: Any):
        """
        Generates Heatmaps and Sensitivity plots for the whole study.
        """
        try:
            import optuna.visualization as oviz
            report_dir = self.output_dir

            # 1. Parameter Importance
            fig_importance = oviz.plot_param_importances(study)
            fig_importance.write_html(os.path.join(report_dir, "opt_param_importance.html"))

            # 2. Optimization History
            fig_history = oviz.plot_optimization_history(study)
            fig_history.write_html(os.path.join(report_dir, "opt_history.html"))

            # 3. Slice Plot (Sensitivity)
            # Leverage Sensitivity
            if "leverage" in study.best_params:
                fig_slice = oviz.plot_slice(study, params=["leverage"])
                fig_slice.write_html(os.path.join(report_dir, "opt_leverage_sensitivity.html"))

            _logger.info("🎨 Optimization visualizations generated.")

        except Exception as e:
            _logger.error(f"Failed to generate optimization viz: {e}")

    def generate_study_summary(self, study: Any):
        """
        Generates a summary of the best trials in the study.
        """
        report_dir = self.output_dir
        os.makedirs(report_dir, exist_ok=True)
        study_name = os.path.basename(report_dir)

        summary_path = os.path.join(report_dir, "study_summary.md")

        df = study.trials_dataframe()
        best_trials = df.sort_values("value", ascending=False).head(10)

        with open(summary_path, 'w') as f:
            f.write(f"# Study Summary: {study_name}\n\n")
            f.write("## Top 10 Trials\n\n")
            f.write(best_trials.to_markdown())
            f.write("\n\n## Best Trial Parameters\n\n")
            f.write(json.dumps(study.best_params, indent=4))

        # Also trigger opt viz
        self.generate_optimization_viz(study)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reporter = Reporter()
    print("Reporter full institutional version.")
