import pandas as pd

def inspect_anomaly():
    path = r"C:\dev\cursor\e-trading\results\p07_combined\p07_generalization_LTCUSDT_4h_segments.csv"
    df = pd.read_csv(path)
    # Find one where return is very negative but sharpe is positive
    anomaly = df[(df['sharpe'] > 0) & (df['total_return_pct'] < -10)].iloc[0]
    print("Anomaly Found:")
    print(anomaly)

if __name__ == "__main__":
    inspect_anomaly()
