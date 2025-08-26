# Knowledge Base: Neural Networks and Regime Detection in Trading

## Neural Networks in Trading

**Neural networks** are machine learning models inspired by the human brain. They consist of layers of interconnected nodes ("neurons") that learn complex patterns from data. Common types include:
- Feedforward Neural Networks (FNN)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN, e.g., LSTM, GRU)
- Transformer Networks

### Why Use Neural Networks in Trading?
- Capture non-linear relationships in price and indicator data
- Model temporal dependencies (e.g., with RNNs/LSTMs)
- Learn from large, multi-dimensional datasets (price, volume, news, etc.)
- Adapt to changing market conditions through retraining

### How Can They Be Integrated?
- **Prediction Models:** Predict future prices or returns
- **Signal Generation:** Output buy/sell/hold signals
- **Feature Extraction:** Use neural nets to extract features for other models
- **Portfolio Optimization:** Predict returns/risk for portfolio allocation
- **Regime Detection:** Classify market regimes (bull, bear, sideways)
- **Reinforcement Learning:** Learn trading policies by interacting with the market

**Integration in this codebase:**
- Add neural network modules in `src/ml/`
- Use in custom entry/exit mixins (e.g., `NeuralNetEntryMixin`)
- Train models using `src/ml/automated_training_pipeline.py` and use them for inference

---

## Regime Detection

**Regime detection** is the process of identifying different "states" or "regimes" of the market, such as bull, bear, or sideways. Different strategies perform better in different regimes, so detecting the current regime allows you to adapt your trading logic.

### Why is Regime Detection Useful?
- Improved returns by using the best strategy for the current regime
- Better risk management
- Adaptive trading, avoiding overfitting to a single market condition

### How is Regime Detection Done?

#### 1. Rule-Based Approaches
- Moving average crossovers (e.g., short MA above long MA = bull)
- Volatility thresholds (e.g., ATR, VIX)
- Price action (support/resistance, breakouts)

#### 2. Statistical & Machine Learning Approaches
- **Hidden Markov Models (HMM):** Infer hidden market states from observed data
- **Clustering:** k-means, Gaussian Mixture Models
- **Neural Networks:** Classify regimes using historical data and indicators

#### 3. Deep Learning for Regime Detection
- Input: Price, volume, indicators, macro data
- Output: Regime label (bull, bear, sideways)
- Can be supervised (labeled data) or unsupervised (clustering)

### Integration in Trading Systems
- **Offline:** Label historical data with regimes, backtest strategies per regime
- **Online:** Detect regime in real-time, switch strategies or parameters accordingly

---

## HMM-Based Regime Detection

A **Hidden Markov Model (HMM)** is a statistical model that infers hidden states (regimes) from observable data (e.g., returns, volatility).

### How It Works
- **States:** Market is in one of several hidden states (e.g., bull, bear)
- **Observations:** You observe something related to the state (e.g., returns)
- **Transitions:** Market can switch between states with certain probabilities
- **Goal:** Given a sequence of observations, infer the most likely sequence of hidden states

### Python Example
```python
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

df = pd.read_csv('your_price_data.csv', parse_dates=['datetime'])
df['returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

X = df['returns'].values.reshape(-1, 1)
model = GaussianHMM(n_components=2, covariance_type='full', n_iter=1000)
model.fit(X)
hidden_states = model.predict(X)
df['regime'] = hidden_states

plt.figure(figsize=(15,5))
for i in range(model.n_components):
    mask = df['regime'] == i
    plt.plot(df['datetime'][mask], df['close'][mask], '.', label=f'Regime {i}')
plt.legend()
plt.show()
```

### How to Use in a Trading Strategy
- Use trend-following strategies in bull regimes
- Use mean-reversion or defensive strategies in bear regimes
- Switch logic based on the detected regime

### Advantages
- Unsupervised: No need to label data
- Probabilistic: Estimates probability of each regime
- Flexible: Can use more than two regimes or add more features

---

## Provided Module: `HMMRegimeDetector`

A ready-to-use module is available at `src/ml/hmm_regime_detector.py`.

### Features
- Fit a Gaussian HMM to price returns or feature matrix
- Predict regimes for new data
- Plot detected regimes

### Usage Example
```python
from src.ml.hmm_regime_detector import HMMRegimeDetector
import pandas as pd

df = pd.read_csv('your_price_data.csv', parse_dates=['datetime'])
detector = HMMRegimeDetector(n_regimes=2)
detector.fit(df['close'])
df['regime'] = detector.predict(df['close'])
detector.plot_regimes(df['datetime'], df['close'], df['regime'])
```

### Requirements
- `hmmlearn`
- `numpy`
- `pandas`
- `matplotlib`

### Integration Tips
- Place regime detection code in `src/ml/`
- Use regime labels in your strategy logic to adapt trading behavior
- For live trading, run the detector in real-time and switch strategies accordingly

---

**For more advanced integration or examples (e.g., with Backtrader), see the codebase or request a template!** 