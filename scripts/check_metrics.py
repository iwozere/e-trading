import pandas as pd

def check_metrics():
    path = r"C:\dev\cursor\e-trading\results\p07_combined\p07_generalization_LTCUSDT_4h_segments.csv"
    try:
        df = pd.read_csv(path)
        # Filter for cases where sharpe > 0 and total_return_pct < 0
        anomalies = df[(df['sharpe'] > 0) & (df['total_return_pct'] < 0)]
        
        if anomalies.empty:
            print("No cases found with positive Sharpe and negative Total Return.")
            # Let's print the first row just to see what the values are
            print("Sample data:")
            print(df[['ticker', 'num_trades', 'sharpe', 'total_return_pct']].head(1))
        else:
            print(f"Found {len(anomalies)} cases:")
            print(anomalies[['ticker', 'data_start', 'num_trades', 'sharpe', 'total_return_pct']])
    except Exception as e:
        print(f"Error reading file: {e}")
        
if __name__ == "__main__":
    check_metrics()
