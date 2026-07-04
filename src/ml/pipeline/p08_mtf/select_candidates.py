from pathlib import Path

import pandas as pd


def select_top_candidates(results_root: str = "results/p08_mtf", top_n: int = 5):
    """
    Analyzes p08_mtf aggregated results and selects top unique ticker/tf pairs for robustness testing.
    Filters:
    - Total Trades >= 15 (Reduced from 30 as p08 segments might be smaller/more selective)
    - Profit Factor > 1.2
    - Total Return [%] > 0
    Sorted by Sharpe Ratio descending.
    """
    root = Path(results_root)
    agg_file = root / "p08_mtf_aggregated_results.csv"

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
    # p08 MTF often has fewer but higher quality trades.
    # Let's use slightly more relaxed trade count but stricter PF.
    mask = (df["Total Trades"] >= 15) & (df["Profit Factor"] > 1.2) & (df["Total Return [%]"] > 0)

    candidates = df[mask].copy()

    if candidates.empty:
        print("No candidates met the robustness criteria (Trades >= 15, PF > 1.2, Return > 0).")
        # Try even more relaxed if empty? No, let's stay honest.
        return

    # Sort by Sharpe Ratio
    candidates = candidates.sort_values(by="Sharpe Ratio", ascending=False)

    # Get unique (ticker, timeframe) pairs, keeping the one with highest Sharpe
    top_pairs = candidates.drop_duplicates(subset=["ticker", "timeframe"]).head(top_n)

    output_file = root / "p08_robustness_candidates.csv"
    top_pairs[["ticker", "timeframe", "Sharpe Ratio", "Profit Factor", "Total Trades"]].to_csv(output_file, index=False)

    print(f"Selected {len(top_pairs)} candidates for P08. Saved to {output_file}")
    print(top_pairs[["ticker", "timeframe", "Sharpe Ratio", "Profit Factor", "Total Trades"]])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P08 Top Candidate Selector")
    parser.add_argument("--root", type=str, default="results/p08_mtf", help="Results root directory")
    parser.add_argument("--n", type=int, default=5, help="Number of candidates to select")
    args = parser.parse_args()

    select_top_candidates(results_root=args.root, top_n=args.n)
