"""
Neural Network Regime Detection Example
--------------------------------------

This example demonstrates how to use the NNRegimeDetector (LSTM-based PyTorch model)
for market regime detection and how to integrate its output into a trading decision pipeline.

Features:
- Generates synthetic data with regime labels
- Trains the neural network regime detector
- Predicts regimes on new data
- Shows how to use regime predictions in trading logic
- Plots the detected regimes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ml.future.nn_regime_detector import NNRegimeDetector

# 1. Generate synthetic data with features and regime labels
def generate_synthetic_data(n_samples=1000, n_features=4, n_regimes=3, seed=42):
    np.random.seed(seed)
    # Simulate regime switches
    regime_lengths = np.random.randint(100, 300, size=n_regimes)
    regimes = np.concatenate([
        np.full(l, i) for i, l in enumerate(regime_lengths)
    ])
    regimes = np.tile(regimes, int(np.ceil(n_samples / regimes.size)))[:n_samples]
    # Simulate features (e.g., return, volatility, rsi, macd)
    features = np.zeros((n_samples, n_features))
    for i in range(n_regimes):
        idx = regimes == i
        features[idx, 0] = np.random.normal(loc=0.01 * (i - 1), scale=0.02, size=idx.sum())  # return
        features[idx, 1] = np.random.normal(loc=0.05 + 0.02 * i, scale=0.01, size=idx.sum())  # volatility
        features[idx, 2] = np.random.normal(loc=50 + 20 * (i - 1), scale=10, size=idx.sum())  # rsi
        features[idx, 3] = np.random.normal(loc=0, scale=1, size=idx.sum())  # macd
    # Create DataFrame
    df = pd.DataFrame(features, columns=['return', 'volatility', 'rsi', 'macd'])
    df['regime'] = regimes
    df['datetime'] = pd.date_range('2023-01-01', periods=n_samples, freq='h')
    return df

def main():
    # Generate data
    df = generate_synthetic_data()
    features = df[['return', 'volatility', 'rsi', 'macd']].values
    labels = df['regime'].values
    print("Data shape:", features.shape)
    print("Regime label distribution:", np.bincount(labels))

    # 2. Train the neural network regime detector
    print("\nTraining neural network regime detector...")
    model = NNRegimeDetector(input_size=4, n_regimes=3)
    model.fit(features, labels, epochs=8, seq_len=20)

    # 3. Predict regimes
    print("\nPredicting regimes...")
    preds = model.predict(features, seq_len=20)
    # Align predictions with original data (due to sequence window)
    df = df.iloc[20:].copy()
    df['predicted_regime'] = preds
    print("Prediction shape:", preds.shape)

    # 4. Example: Use regime in trading logic
    print("\nExample trading logic based on detected regime:")
    for regime in range(3):
        count = (df['predicted_regime'] == regime).sum()
        print(f"Regime {regime}: {count} bars detected")
    # Simple rule: If regime==0 (bull), 'buy'; if regime==1 (bear), 'sell'; else 'hold'
    df['signal'] = np.where(df['predicted_regime'] == 0, 'buy',
                     np.where(df['predicted_regime'] == 1, 'sell', 'hold'))
    print(df[['datetime', 'predicted_regime', 'signal']].head())

    # 5. Plot the detected regimes
    plt.figure(figsize=(15, 5))
    for regime in range(3):
        mask = df['predicted_regime'] == regime
        plt.plot(df['datetime'][mask], df['return'][mask], '.', label=f'Regime {regime}')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.title('Detected Regimes by Neural Network')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()