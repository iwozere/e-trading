from pathlib import Path

import pandas as pd


def select_top_candidates(results_root: str = "results/p07_combined", top_n: int = 5):
    """
    Analyzes aggregated results and selects top unique ticker/tf pairs for robustness testing.
    Filters:
    - Total Trades >= 30
    - Profit Factor > 1.1
    - Total Return [%] > 0
    Sorted by Sharpe Ratio descending.
    """
    root = Path(results_root)
    agg_file = root / "p07_combined_aggregated_results.csv"

    if not agg_file.exists():
        print(f"Aggregated results file not found at {agg_file}")
        return

    df = pd.read_csv(agg_file)

    # Ensure numeric columns are actually numeric
    numeric_cols = ["Total Trades", "Profit Factor", "Total Return [%]", "Sharpe Ratio"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply Filters
    # Note: We prioritize Validation (VAL) results if available, but for general selection
    # we filter the whole set and then drop duplicates to get unique pairs.
    mask = (df["Total Trades"] >= 30) & (df["Profit Factor"] > 1.1) & (df["Total Return [%]"] > 0)

    candidates = df[mask].copy()

    if candidates.empty:
        print("No candidates met the robustness criteria (Trades >= 30, PF > 1.1, Return > 0).")
        return

    # Sort by Sharpe Ratio
    candidates = candidates.sort_values(by="Sharpe Ratio", ascending=False)

    # Get unique (ticker, timeframe) pairs, keeping the one with highest Sharpe
    top_pairs = candidates.drop_duplicates(subset=["ticker", "timeframe"]).head(top_n)

    output_file = root / "p07_robustness_candidates.csv"
    top_pairs[["ticker", "timeframe", "Sharpe Ratio", "Profit Factor", "Total Trades"]].to_csv(output_file, index=False)

    print(f"Selected {len(top_pairs)} candidates. Saved to {output_file}")
    print(top_pairs[["ticker", "timeframe", "Sharpe Ratio"]])


if __name__ == "__main__":
    select_top_candidates()
