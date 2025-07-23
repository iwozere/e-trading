"""
hmm_regime_detector.py
----------------------

A ready-to-use HMM-based regime detector for financial time series.
Uses hmmlearn to fit a GaussianHMM to returns or other features.

Features:
- Fit an HMM to returns or other features
- Predict regimes for new data
- Plot detected regimes

Uses Hidden Markov Models with Gaussian emissions
Unsupervised learning approach - discovers regimes automatically from data patterns
Works primarily with price series (computes log returns internally)
Probabilistic model that assumes regimes follow a Markov chain
Simpler architecture with fewer hyperparameters

Dependencies:
- numpy
- pandas
- hmmlearn
- matplotlib (for plotting)

Usage Example:
--------------
from src.ml.hmm_regime_detector import HMMRegimeDetector

df = pd.read_csv('your_price_data.csv', parse_dates=['datetime'])
detector = HMMRegimeDetector(n_regimes=2)
detector.fit(df['close'])
df['regime'] = detector.predict(df['close'])
detector.plot_regimes(df['datetime'], df['close'], df['regime'])
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

class HMMRegimeDetector:
    def __init__(self, n_regimes=2, n_iter=1000, covariance_type='full', random_state=42):
        """
        Initialize the HMM regime detector.
        Args:
            n_regimes (int): Number of regimes (hidden states)
            n_iter (int): Number of EM iterations to perform
            covariance_type (str): Covariance type for GaussianHMM
            random_state (int): Random seed
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = GaussianHMM(n_components=n_regimes, covariance_type=covariance_type, n_iter=n_iter, random_state=random_state)
        self.fitted = False

    def _compute_returns(self, prices):
        """Compute log returns from price series."""
        returns = np.log(np.array(prices) / np.array(prices).shift(1))
        returns = np.nan_to_num(returns)
        return returns.reshape(-1, 1)

    def fit(self, prices_or_features):
        """
        Fit the HMM to a price series or feature matrix.
        Args:
            prices_or_features (array-like or pd.Series or pd.DataFrame):
                If 1D, treated as price series (log returns will be computed).
                If 2D, treated as feature matrix.
        """
        if isinstance(prices_or_features, (pd.Series, np.ndarray, list)) and np.ndim(prices_or_features) == 1:
            X = self._compute_returns(pd.Series(prices_or_features))
        else:
            X = np.array(prices_or_features)
        self.model.fit(X)
        self.fitted = True
        return self

    def predict(self, prices_or_features):
        """
        Predict regimes for a price series or feature matrix.
        Args:
            prices_or_features (array-like or pd.Series or pd.DataFrame):
                If 1D, treated as price series (log returns will be computed).
                If 2D, treated as feature matrix.
        Returns:
            np.ndarray: Array of regime labels (0, 1, ...)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        if isinstance(prices_or_features, (pd.Series, np.ndarray, list)) and np.ndim(prices_or_features) == 1:
            X = self._compute_returns(pd.Series(prices_or_features))
        else:
            X = np.array(prices_or_features)
        return self.model.predict(X)

    def plot_regimes(self, datetimes, prices, regimes, title='HMM Regime Detection'):
        """
        Plot price series colored by detected regimes.
        Args:
            datetimes (array-like): Timestamps
            prices (array-like): Price series
            regimes (array-like): Regime labels
            title (str): Plot title
        """
        plt.figure(figsize=(15, 5))
        regimes = np.array(regimes)
        for regime in np.unique(regimes):
            mask = regimes == regime
            plt.plot(np.array(datetimes)[mask], np.array(prices)[mask], '.', label=f'Regime {regime}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example usage (uncomment to run as script)
# if __name__ == '__main__':
#     import pandas as pd
#     df = pd.read_csv('your_price_data.csv', parse_dates=['datetime'])
#     detector = HMMRegimeDetector(n_regimes=2)
#     detector.fit(df['close'])
#     df['regime'] = detector.predict(df['close'])
#     detector.plot_regimes(df['datetime'], df['close'], df['regime'])