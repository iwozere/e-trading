import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Optional
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class P07RegimeModel:
    """
    GaussianHMM for Market Regime Classification.
    Uses macro features (VIX, BTC Market Cap) to identify Bull, Bear, and Sideways states.
    """

    def __init__(self, n_components: int = 3, model_dir: Path = Path("src/ml/pipeline/p07_combined/models/regime")):
        self.n_components = n_components
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.state_mapping = {} # To be defined after inspection (e.g., 0=Bear, 1=Sideways, 2=Bull)

    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Process macro features into stationarity-adjusted inputs.
        VIX: raw (stationary-ish)
        BTC MC: Log-return (stationary)
        """
        features = pd.DataFrame(index=df.index)

        if "vix" in df.columns:
            features["vix"] = df["vix"]
        else:
            features["vix"] = np.nan

        if "btc_mc" in df.columns:
            # Use log returns for market cap to ensure stationarity
            features["btc_mc_log_ret"] = np.log(df["btc_mc"] / df["btc_mc"].shift(1))
        else:
            features["btc_mc_log_ret"] = np.nan

        features.dropna(inplace=True)

        if features.empty:
            _logger.warning("Macro features became empty after dropna. Returning empty array.")
            return np.array([]).reshape(0, 2)

        if fit:
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)

    def train(self, macro_df: pd.DataFrame, anchor_date: Optional[pd.Timestamp] = None):
        """
        Train the HMM on macro historical data.
        If anchor_date is provided, only data up to that date is used (prevents look-ahead bias).
        """
        if anchor_date:
            macro_df = macro_df[macro_df.index <= anchor_date]

        X = self.prepare_features(macro_df, fit=True)
        if X.size == 0:
            _logger.error("No samples available for HMM training.")
            return

        _logger.info("Training GaussianHMM with %d components on %d samples.", self.n_components, len(X))
        self.model.fit(X)

        # After training, inspect means to perform state mapping
        # This is a heuristic: lower VIX + positive BTC MC might be Bull.
        # We can also just store the raw states and let XGBoost learn their value.
        _logger.info("HMM Training complete.")

    def predict(self, macro_df: pd.DataFrame) -> pd.Series:
        """Predict regime states for given macro data."""
        X = self.prepare_features(macro_df, fit=False)
        res = pd.Series(0, index=macro_df.index, name="global_regime")

        if X.size == 0:
            return res

        states = self.model.predict(X)

        # Align with original index (accounting for dropped NaNs during feature prep)
        features = pd.DataFrame(index=macro_df.index)
        if "vix" in macro_df.columns:
            features["vix"] = macro_df["vix"]
        if "btc_mc" in macro_df.columns:
            features["btc_mc_log_ret"] = np.log(macro_df["btc_mc"] / macro_df["btc_mc"].shift(1))

        valid_idx = features.dropna().index
        res.loc[valid_idx] = states
        return res.ffill().fillna(0).astype(int)

    def save(self, name: str = "macro_hmm.joblib"):
        """Persist model and scaler."""
        path = self.model_dir / name
        joblib.dump({"model": self.model, "scaler": self.scaler, "n_components": self.n_components}, path)
        _logger.info("Regime model saved to %s", path)

    def load(self, name: str = "macro_hmm.joblib"):
        """Load persisted model and scaler."""
        path = self.model_dir / name
        if not path.exists():
            _logger.error("Model file %s not found.", path)
            return False

        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.n_components = data["n_components"]
        return True
