"""
Feature Engineering Pipeline

This module provides comprehensive feature engineering capabilities:
- Automated feature extraction
- Technical indicator features
- Market microstructure features
- Feature selection and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class TechnicalIndicatorFeatures:
    """Generates technical indicator features."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.features = {}

    def generate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all technical indicator features."""
        features_df = data.copy()

        # Trend indicators
        features_df = self._add_trend_indicators(features_df)

        # Momentum indicators
        features_df = self._add_momentum_indicators(features_df)

        # Volatility indicators
        features_df = self._add_volatility_indicators(features_df)

        # Volume indicators
        features_df = self._add_volume_indicators(features_df)

        # Oscillator indicators
        features_df = self._add_oscillator_indicators(features_df)

        # Pattern recognition
        features_df = self._add_pattern_indicators(features_df)

        return features_df

    def _add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            data[f'sma_{period}'] = talib.SMA(data['close'], timeperiod=period)
            data[f'ema_{period}'] = talib.EMA(data['close'], timeperiod=period)
            data[f'wma_{period}'] = talib.WMA(data['close'], timeperiod=period)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(data['close'])
        data['macd'] = macd
        data['macd_signal'] = macd_signal
        data['macd_histogram'] = macd_hist

        # Parabolic SAR
        data['sar'] = talib.SAR(data['high'], data['low'])

        # ADX (Average Directional Index)
        data['adx'] = talib.ADX(data['high'], data['low'], data['close'])
        data['plus_di'] = talib.PLUS_DI(data['high'], data['low'], data['close'])
        data['minus_di'] = talib.MINUS_DI(data['high'], data['low'], data['close'])

        # Ichimoku Cloud
        data['tenkan_sen'] = talib.SMA(data['close'], timeperiod=9)
        data['kijun_sen'] = talib.SMA(data['close'], timeperiod=26)
        data['senkou_span_a'] = (data['tenkan_sen'] + data['kijun_sen']) / 2
        data['senkou_span_b'] = talib.SMA(data['close'], timeperiod=52)

        return data

    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        for period in [7, 14, 21]:
            data[f'rsi_{period}'] = talib.RSI(data['close'], timeperiod=period)

        # Stochastic
        slowk, slowd = talib.STOCH(data['high'], data['low'], data['close'])
        data['stoch_k'] = slowk
        data['stoch_d'] = slowd

        # Williams %R
        data['williams_r'] = talib.WILLR(data['high'], data['low'], data['close'])

        # CCI (Commodity Channel Index)
        data['cci'] = talib.CCI(data['high'], data['low'], data['close'])

        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            data[f'roc_{period}'] = talib.ROC(data['close'], timeperiod=period)

        # Momentum
        for period in [5, 10, 20]:
            data[f'momentum_{period}'] = talib.MOM(data['close'], timeperiod=period)

        return data

    def _add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Bollinger Bands
        for period in [10, 20, 50]:
            upper, middle, lower = talib.BBANDS(data['close'], timeperiod=period)
            data[f'bb_upper_{period}'] = upper
            data[f'bb_middle_{period}'] = middle
            data[f'bb_lower_{period}'] = lower
            data[f'bb_width_{period}'] = (upper - lower) / middle
            data[f'bb_position_{period}'] = (data['close'] - lower) / (upper - lower)

        # ATR (Average True Range)
        for period in [7, 14, 21]:
            data[f'atr_{period}'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)

        # Keltner Channel
        for period in [10, 20]:
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)
            ema = talib.EMA(data['close'], timeperiod=period)
            data[f'keltner_upper_{period}'] = ema + 2 * atr
            data[f'keltner_lower_{period}'] = ema - 2 * atr
            data[f'keltner_width_{period}'] = (data[f'keltner_upper_{period}'] - data[f'keltner_lower_{period}']) / ema

        # Historical Volatility
        for period in [10, 20, 50]:
            returns = data['close'].pct_change()
            data[f'hist_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)

        return data

    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # OBV (On Balance Volume)
        data['obv'] = talib.OBV(data['close'], data['volume'])

        # Volume SMA
        for period in [5, 10, 20]:
            data[f'volume_sma_{period}'] = talib.SMA(data['volume'], timeperiod=period)
            data[f'volume_ratio_{period}'] = data['volume'] / data[f'volume_sma_{period}']

        # Chaikin Money Flow
        data['cmf'] = talib.ADOSC(data['high'], data['low'], data['close'], data['volume'])

        # Money Flow Index
        data['mfi'] = talib.MFI(data['high'], data['low'], data['close'], data['volume'])

        # Accumulation/Distribution Line
        data['ad'] = talib.AD(data['high'], data['low'], data['close'], data['volume'])

        # Volume Price Trend
        data['vpt'] = data['volume'] * ((data['close'] - data['close'].shift(1)) / data['close'].shift(1))
        data['vpt'] = data['vpt'].cumsum()

        return data

    def _add_oscillator_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add oscillator indicators."""
        # Stochastic RSI
        for period in [7, 14]:
            rsi = talib.RSI(data['close'], timeperiod=period)
            data[f'stoch_rsi_k_{period}'], data[f'stoch_rsi_d_{period}'] = talib.STOCH(
                rsi, rsi, rsi, fastk_period=5, slowk_period=3, slowd_period=3
            )

        # Ultimate Oscillator
        data['ult_osc'] = talib.ULTOSC(data['high'], data['low'], data['close'])

        # TRIX
        data['trix'] = talib.TRIX(data['close'])

        # AROON
        data['aroon_up'], data['aroon_down'] = talib.AROON(data['high'], data['low'])
        data['aroon_osc'] = talib.AROONOSC(data['high'], data['low'])

        return data

    def _add_pattern_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern indicators."""
        # Common candlestick patterns
        patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLDARKCLOUDCOVER',
            'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING',
            'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLHAMMER', 'CDLHANGINGMAN',
            'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE',
            'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK',
            'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
            'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW',
            'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK',
            'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES',
            'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN',
            'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING',
            'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
        ]

        for pattern in patterns:
            try:
                pattern_func = getattr(talib, pattern)
                data[f'pattern_{pattern.lower()}'] = pattern_func(
                    data['open'], data['high'], data['low'], data['close']
                )
            except:
                continue

        return data


class MarketMicrostructureFeatures:
    """Generates market microstructure features."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def generate_features(self, data: pd.DataFrame, orderbook_data: pd.DataFrame = None) -> pd.DataFrame:
        """Generate market microstructure features."""
        features_df = data.copy()

        # Basic microstructure features
        features_df = self._add_basic_microstructure_features(features_df)

        # Order book features (if available)
        if orderbook_data is not None:
            features_df = self._add_orderbook_features(features_df, orderbook_data)

        # Volume profile features
        features_df = self._add_volume_profile_features(features_df)

        # Price impact features
        features_df = self._add_price_impact_features(features_df)

        # Liquidity features
        features_df = self._add_liquidity_features(features_df)

        return features_df

    def _add_basic_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic market microstructure features."""
        # Bid-Ask Spread (simulated if not available)
        data['spread'] = (data['high'] - data['low']) / data['close']

        # Price efficiency
        data['price_efficiency'] = abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)

        # Volume-weighted average price (VWAP)
        data['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']

        # Realized volatility
        for period in [5, 10, 20]:
            returns = data['close'].pct_change()
            data[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)

        # Price momentum
        for period in [1, 5, 10]:
            data[f'price_momentum_{period}'] = data['close'] / data['close'].shift(period) - 1

        # Volume momentum
        for period in [1, 5, 10]:
            data[f'volume_momentum_{period}'] = data['volume'] / data['volume'].shift(period) - 1

        # High-Low ratio
        data['hl_ratio'] = data['high'] / data['low']

        # Close-Open ratio
        data['co_ratio'] = data['close'] / data['open']

        return data

    def _add_orderbook_features(self, data: pd.DataFrame, orderbook: pd.DataFrame) -> pd.DataFrame:
        """Add order book features."""
        # Order book imbalance
        if 'bid_volume' in orderbook.columns and 'ask_volume' in orderbook.columns:
            data['order_imbalance'] = (orderbook['bid_volume'] - orderbook['ask_volume']) / (orderbook['bid_volume'] + orderbook['ask_volume'])

        # Order book depth
        if 'bid_depth' in orderbook.columns and 'ask_depth' in orderbook.columns:
            data['order_depth'] = orderbook['bid_depth'] + orderbook['ask_depth']
            data['depth_imbalance'] = (orderbook['bid_depth'] - orderbook['ask_depth']) / data['order_depth']

        # Order book pressure
        if 'bid_pressure' in orderbook.columns and 'ask_pressure' in orderbook.columns:
            data['order_pressure'] = orderbook['bid_pressure'] - orderbook['ask_pressure']

        return data

    def _add_volume_profile_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume profile features."""
        # Volume-weighted price levels
        data['volume_price_level'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()

        # Volume concentration
        data['volume_concentration'] = data['volume'] / data['volume'].rolling(20).mean()

        # Volume trend
        data['volume_trend'] = data['volume'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

        # Volume volatility
        data['volume_volatility'] = data['volume'].rolling(20).std() / data['volume'].rolling(20).mean()

        return data

    def _add_price_impact_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price impact features."""
        # Kyle's lambda (price impact)
        for period in [5, 10, 20]:
            returns = data['close'].pct_change()
            volume = data['volume']

            # Calculate price impact
            data[f'price_impact_{period}'] = (
                returns.rolling(period).cov(volume) / volume.rolling(period).var()
            )

        # Amihud illiquidity
        for period in [5, 10, 20]:
            returns = data['close'].pct_change()
            volume = data['volume']
            data[f'amihud_illiquidity_{period}'] = abs(returns) / volume

        return data

    def _add_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity features."""
        # Roll's implicit spread estimator
        for period in [5, 10, 20]:
            returns = data['close'].pct_change()
            data[f'roll_spread_{period}'] = 2 * np.sqrt(-returns.rolling(period).cov(returns.shift(1)))

        # Effective spread
        data['effective_spread'] = 2 * abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)

        # Quoted spread (simulated)
        data['quoted_spread'] = data['spread'] * 0.5  # Simplified assumption

        return data


class StatisticalFeatures:
    """Generates statistical features."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features."""
        features_df = data.copy()

        # Rolling statistics
        features_df = self._add_rolling_statistics(features_df)

        # Cross-sectional features
        features_df = self._add_cross_sectional_features(features_df)

        # Time series features
        features_df = self._add_time_series_features(features_df)

        return features_df

    def _add_rolling_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features."""
        for period in [5, 10, 20, 50]:
            # Price statistics
            data[f'price_mean_{period}'] = data['close'].rolling(period).mean()
            data[f'price_std_{period}'] = data['close'].rolling(period).std()
            data[f'price_skew_{period}'] = data['close'].rolling(period).skew()
            data[f'price_kurt_{period}'] = data['close'].rolling(period).kurt()

            # Volume statistics
            data[f'volume_mean_{period}'] = data['volume'].rolling(period).mean()
            data[f'volume_std_{period}'] = data['volume'].rolling(period).std()
            data[f'volume_skew_{period}'] = data['volume'].rolling(period).skew()

            # Return statistics
            returns = data['close'].pct_change()
            data[f'return_mean_{period}'] = returns.rolling(period).mean()
            data[f'return_std_{period}'] = returns.rolling(period).std()
            data[f'return_skew_{period}'] = returns.rolling(period).skew()
            data[f'return_kurt_{period}'] = returns.rolling(period).kurt()

        return data

    def _add_cross_sectional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional features."""
        # Z-score of price
        for period in [10, 20, 50]:
            data[f'price_zscore_{period}'] = (
                (data['close'] - data['close'].rolling(period).mean()) /
                data['close'].rolling(period).std()
            )

        # Z-score of volume
        for period in [10, 20, 50]:
            data[f'volume_zscore_{period}'] = (
                (data['volume'] - data['volume'].rolling(period).mean()) /
                data['volume'].rolling(period).std()
            )

        # Percentile ranks
        for period in [10, 20, 50]:
            data[f'price_percentile_{period}'] = data['close'].rolling(period).rank(pct=True)
            data[f'volume_percentile_{period}'] = data['volume'].rolling(period).rank(pct=True)

        return data

    def _add_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time series features."""
        # Autocorrelation
        for period in [1, 5, 10]:
            returns = data['close'].pct_change()
            data[f'return_autocorr_{period}'] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=period)
            )

        # Trend strength
        for period in [10, 20, 50]:
            data[f'trend_strength_{period}'] = data['close'].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.std(x)
            )

        # Mean reversion
        for period in [10, 20, 50]:
            data[f'mean_reversion_{period}'] = (
                data['close'] - data['close'].rolling(period).mean()
            ) / data['close'].rolling(period).std()

        return data


class FeatureSelector:
    """Handles feature selection and validation."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.selected_features = []
        self.feature_importance = {}
        self.correlation_matrix = None

    def select_features(self,
                       X: pd.DataFrame,
                       y: pd.Series,
                       method: str = "mutual_info",
                       n_features: int = 50,
                       threshold: float = 0.01) -> pd.DataFrame:
        """Select features using various methods."""
        try:
            if method == "mutual_info":
                return self._mutual_info_selection(X, y, n_features, threshold)
            elif method == "f_regression":
                return self._f_regression_selection(X, y, n_features, threshold)
            elif method == "correlation":
                return self._correlation_selection(X, y, threshold)
            elif method == "pca":
                return self._pca_selection(X, n_features)
            else:
                raise ValueError(f"Unknown selection method: {method}")

        except Exception:
            logger.exception("Error in feature selection: ")
            return X

    def _mutual_info_selection(self,
                             X: pd.DataFrame,
                             y: pd.Series,
                             n_features: int,
                             threshold: float) -> pd.DataFrame:
        """Select features using mutual information."""
        # Remove non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])

        # Fill NaN values
        X_numeric = X_numeric.fillna(X_numeric.mean())

        # Calculate mutual information
        mi_scores = mutual_info_regression(X_numeric, y, random_state=42)

        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': X_numeric.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)

        # Select features
        selected_features = feature_importance[
            feature_importance['importance'] > threshold
        ]['feature'].head(n_features).tolist()

        self.selected_features = selected_features
        self.feature_importance = dict(zip(feature_importance['feature'], feature_importance['importance']))

        return X[selected_features]

    def _f_regression_selection(self,
                              X: pd.DataFrame,
                              y: pd.Series,
                              n_features: int,
                              threshold: float) -> pd.DataFrame:
        """Select features using F-regression."""
        # Remove non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])

        # Fill NaN values
        X_numeric = X_numeric.fillna(X_numeric.mean())

        # Calculate F-scores
        f_scores, _ = f_regression(X_numeric, y)

        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': X_numeric.columns,
            'importance': f_scores
        }).sort_values('importance', ascending=False)

        # Select features
        selected_features = feature_importance[
            feature_importance['importance'] > threshold
        ]['feature'].head(n_features).tolist()

        self.selected_features = selected_features
        self.feature_importance = dict(zip(feature_importance['feature'], feature_importance['importance']))

        return X[selected_features]

    def _correlation_selection(self,
                             X: pd.DataFrame,
                             y: pd.Series,
                             threshold: float) -> pd.DataFrame:
        """Select features based on correlation with target."""
        # Remove non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])

        # Fill NaN values
        X_numeric = X_numeric.fillna(X_numeric.mean())

        # Calculate correlations with target
        correlations = X_numeric.corrwith(y).abs()

        # Select features with high correlation
        selected_features = correlations[correlations > threshold].index.tolist()

        self.selected_features = selected_features
        self.feature_importance = correlations.to_dict()

        return X[selected_features]

    def _pca_selection(self, X: pd.DataFrame, n_features: int) -> pd.DataFrame:
        """Select features using PCA."""
        # Remove non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])

        # Fill NaN values
        X_numeric = X_numeric.fillna(X_numeric.mean())

        # Apply PCA
        pca = PCA(n_components=min(n_features, X_numeric.shape[1]))
        pca_result = pca.fit_transform(X_numeric)

        # Create feature names
        feature_names = [f'pca_component_{i+1}' for i in range(pca_result.shape[1])]

        # Create DataFrame
        pca_df = pd.DataFrame(pca_result, columns=feature_names, index=X.index)

        self.selected_features = feature_names
        self.feature_importance = dict(zip(feature_names, pca.explained_variance_ratio_))

        return pca_df

    def analyze_correlations(self, X: pd.DataFrame, threshold: float = 0.8) -> Dict[str, List[str]]:
        """Analyze feature correlations and identify multicollinearity."""
        # Remove non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])

        # Calculate correlation matrix
        self.correlation_matrix = X_numeric.corr()

        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = abs(self.correlation_matrix.iloc[i, j])
                if corr_value > threshold:
                    high_corr_pairs.append({
                        'feature1': self.correlation_matrix.columns[i],
                        'feature2': self.correlation_matrix.columns[j],
                        'correlation': corr_value
                    })

        return {
            'high_correlation_pairs': high_corr_pairs,
            'correlation_matrix': self.correlation_matrix
        }

    def get_feature_stability(self,
                            X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            feature_importance: Dict[str, float]) -> Dict[str, float]:
        """Calculate feature stability across train/test sets."""
        stability_scores = {}

        for feature in feature_importance.keys():
            if feature in X_train.columns and feature in X_test.columns:
                # Calculate distribution similarity
                train_dist = X_train[feature].describe()
                test_dist = X_test[feature].describe()

                # Calculate stability score (simplified)
                stability = 1 - abs(train_dist['mean'] - test_dist['mean']) / (train_dist['mean'] + 1e-8)
                stability_scores[feature] = max(0, stability)

        return stability_scores


class FeatureEngineeringPipeline:
    """Main feature engineering pipeline."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.technical_features = TechnicalIndicatorFeatures(config.get("technical", {}))
        self.microstructure_features = MarketMicrostructureFeatures(config.get("microstructure", {}))
        self.statistical_features = StatisticalFeatures(config.get("statistical", {}))
        self.feature_selector = FeatureSelector(config.get("selection", {}))

        # Feature scalers
        self.scalers = {}
        self.feature_names = []

    def generate_features(self,
                         data: pd.DataFrame,
                         orderbook_data: pd.DataFrame = None,
                         target_column: str = None) -> pd.DataFrame:
        """Generate all features."""
        try:
            logger.info("Starting feature generation...")

            # Generate technical indicators
            logger.info("Generating technical indicators...")
            features_df = self.technical_features.generate_all_features(data)

            # Generate microstructure features
            logger.info("Generating microstructure features...")
            features_df = self.microstructure_features.generate_features(features_df, orderbook_data)

            # Generate statistical features
            logger.info("Generating statistical features...")
            features_df = self.statistical_features.generate_features(features_df)

            # Store feature names
            self.feature_names = [col for col in features_df.columns if col not in data.columns]

            logger.info("Generated %d features", len(self.feature_names))
            return features_df

        except Exception:
            logger.exception("Error in feature generation: ")
            return data

    def select_features(self,
                       X: pd.DataFrame,
                       y: pd.Series,
                       method: str = "mutual_info",
                       n_features: int = 50) -> pd.DataFrame:
        """Select the most important features."""
        try:
            logger.info("Selecting features using %s...", method)

            selected_X = self.feature_selector.select_features(
                X, y, method, n_features
            )

            logger.info("Selected %d features", len(selected_X.columns))
            return selected_X

        except Exception:
            logger.exception("Error in feature selection: ")
            return X

    def scale_features(self,
                      X: pd.DataFrame,
                      scaler_type: str = "standard",
                      fit: bool = True) -> pd.DataFrame:
        """Scale features using various methods."""
        try:
            if scaler_type == "standard":
                scaler = StandardScaler()
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
            elif scaler_type == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")

            if fit:
                scaled_X = scaler.fit_transform(X)
                self.scalers[scaler_type] = scaler
            else:
                if scaler_type in self.scalers:
                    scaled_X = self.scalers[scaler_type].transform(X)
                else:
                    raise ValueError(f"Scaler {scaler_type} not fitted")

            return pd.DataFrame(scaled_X, columns=X.columns, index=X.index)

        except Exception:
            logger.exception("Error in feature scaling: ")
            return X

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_selector.feature_importance

    def get_correlation_analysis(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get correlation analysis results."""
        return self.feature_selector.analyze_correlations(X)

    def get_feature_stability(self,
                            X_train: pd.DataFrame,
                            X_test: pd.DataFrame) -> Dict[str, float]:
        """Get feature stability scores."""
        return self.feature_selector.get_feature_stability(
            X_train, X_test, self.feature_selector.feature_importance
        )

    def save_pipeline(self, filepath: str):
        """Save the feature engineering pipeline."""
        import pickle

        pipeline_data = {
            'feature_names': self.feature_names,
            'scalers': self.scalers,
            'feature_selector': self.feature_selector,
            'config': self.config
        }

        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)

        logger.info("Saved feature engineering pipeline to %s", filepath)

    def load_pipeline(self, filepath: str):
        """Load the feature engineering pipeline."""
        import pickle

        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)

        self.feature_names = pipeline_data['feature_names']
        self.scalers = pipeline_data['scalers']
        self.feature_selector = pipeline_data['feature_selector']
        self.config = pipeline_data['config']

        logger.info("Loaded feature engineering pipeline from %s", filepath)
