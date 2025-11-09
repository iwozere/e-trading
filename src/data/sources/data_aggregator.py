"""
Data aggregator for combining data from multiple sources.

This module provides functionality to aggregate, merge, and synchronize
data from multiple data sources and providers.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

from src.data.sources.data_source_factory import get_data_source_factory
from src.data.utils import get_data_handler
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class DataAggregator:
    """
    Aggregates data from multiple sources and providers.

    Provides:
    - Multi-source data combination
    - Data synchronization and alignment
    - Quality assessment across sources
    - Conflict resolution strategies
    """

    def __init__(self, primary_provider: str = "binance"):
        """
        Initialize data aggregator.

        Args:
            primary_provider: Primary data provider for conflict resolution
        """
        self.primary_provider = primary_provider
        self.factory = get_data_source_factory()
        self.data_handler = get_data_handler(primary_provider)

        _logger.info("Data aggregator initialized with primary provider: %s", primary_provider)

    def aggregate_data(
        self,
        symbol: str,
        interval: str,
        providers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategy: str = "primary_preferred"
    ) -> Optional[pd.DataFrame]:
        """
        Aggregate data from multiple providers.

        Args:
            symbol: Trading symbol
            interval: Data interval
            providers: List of provider names
            start_date: Start date for data range
            end_date: End date for data range
            strategy: Aggregation strategy ('primary_preferred', 'consensus', 'best_quality')

        Returns:
            Aggregated DataFrame or None if failed
        """
        try:
            # Collect data from all providers
            provider_data = {}
            for provider in providers:
                data_source = self.factory.get_or_create_data_source(provider)
                if data_source:
                    data = data_source.get_data_with_cache(
                        symbol, interval, start_date, end_date
                    )
                    if data is not None and not data.empty:
                        provider_data[provider] = data
                        _logger.info("Retrieved %d data points from %s", len(data), provider)
                    else:
                        _logger.warning("No data received from %s", provider)
                else:
                    _logger.error("Failed to create data source for %s", provider)

            if not provider_data:
                _logger.error("No data received from any provider")
                return None

            # Aggregate data based on strategy
            if strategy == "primary_preferred":
                return self._aggregate_primary_preferred(provider_data, symbol, interval)
            elif strategy == "consensus":
                return self._aggregate_consensus(provider_data, symbol, interval)
            elif strategy == "best_quality":
                return self._aggregate_best_quality(provider_data, symbol, interval)
            else:
                _logger.error("Unknown aggregation strategy: %s", strategy)
                return None

        except Exception:
            _logger.exception("Failed to aggregate data for %s:", symbol)
            return None

    def _aggregate_primary_preferred(
        self,
        provider_data: Dict[str, pd.DataFrame],
        symbol: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Aggregate data preferring the primary provider.

        Args:
            provider_data: Dictionary mapping provider names to DataFrames
            symbol: Trading symbol
            interval: Data interval

        Returns:
            Aggregated DataFrame
        """
        if self.primary_provider in provider_data:
            primary_data = provider_data[self.primary_provider].copy()

            # Fill gaps with data from other providers
            for provider, data in provider_data.items():
                if provider != self.primary_provider:
                    primary_data = self._fill_data_gaps(primary_data, data, symbol)

            return primary_data
        else:
            # Fall back to first available provider
            first_provider = list(provider_data.keys())[0]
            return provider_data[first_provider].copy()

    def _aggregate_consensus(
        self,
        provider_data: Dict[str, pd.DataFrame],
        symbol: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Aggregate data using consensus approach.

        Args:
            provider_data: Dictionary mapping provider names to DataFrames
            symbol: Trading symbol
            interval: Data interval

        Returns:
            Aggregated DataFrame
        """
        # Find common timestamp range
        common_start = max(data['timestamp'].min() for data in provider_data.values())
        common_end = min(data['timestamp'].max() for data in provider_data.values())

        # Filter all datasets to common range
        filtered_data = {}
        for provider, data in provider_data.items():
            mask = (data['timestamp'] >= common_start) & (data['timestamp'] <= common_end)
            filtered_data[provider] = data[mask].copy()

        # Calculate consensus values
        consensus_data = []
        timestamps = sorted(set.intersection(*[set(data['timestamp']) for data in filtered_data.values()]))

        for ts in timestamps:
            consensus_point = self._calculate_consensus_point(
                filtered_data, ts, symbol
            )
            if consensus_point is not None:
                consensus_data.append(consensus_point)

        if not consensus_data:
            _logger.warning("No consensus data points for %s", symbol)
            return pd.DataFrame()

        return pd.DataFrame(consensus_data)

    def _aggregate_best_quality(
        self,
        provider_data: Dict[str, pd.DataFrame],
        symbol: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Aggregate data selecting the best quality source for each time period.

        Args:
            provider_data: Dictionary mapping provider names to DataFrames
            symbol: Trading symbol
            interval: Data interval

        Returns:
            Aggregated DataFrame
        """
        # Assess quality of each provider's data
        quality_scores = {}
        for provider, data in provider_data.items():
            try:
                validation_result = self.data_handler.validate_and_score_data(data, symbol)
                quality_scores[provider] = validation_result['quality_score']['quality_score']
            except Exception as e:
                _logger.warning("Failed to assess quality for %s: %s",provider, e)
                quality_scores[provider] = 0.0

        # Sort providers by quality score
        sorted_providers = sorted(
            quality_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Use highest quality provider as base
        best_provider = sorted_providers[0][0]
        best_data = provider_data[best_provider].copy()

        # Fill gaps with next best provider
        for provider, _ in sorted_providers[1:]:
            best_data = self._fill_data_gaps(best_data, provider_data[provider], symbol)

        return best_data

    def _fill_data_gaps(
        self,
        base_data: pd.DataFrame,
        fill_data: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Fill gaps in base data with fill data.

        Args:
            base_data: Base DataFrame to fill gaps in
            fill_data: DataFrame to use for filling gaps
            symbol: Trading symbol

        Returns:
            DataFrame with gaps filled
        """
        if base_data.empty or fill_data.empty:
            return base_data

        # Find timestamps in fill_data that are not in base_data
        base_timestamps = set(base_data['timestamp'])
        fill_timestamps = set(fill_data['timestamp'])

        missing_timestamps = fill_timestamps - base_timestamps

        if not missing_timestamps:
            return base_data

        # Get missing data points
        missing_data = fill_data[fill_data['timestamp'].isin(missing_timestamps)].copy()

        # Combine data
        combined_data = pd.concat([base_data, missing_data], ignore_index=True)
        combined_data.sort_values('timestamp', inplace=True)
        combined_data.reset_index(drop=True, inplace=True)

        _logger.info("Filled %d gaps in %s data", len(missing_data), symbol)

        return combined_data

    def _calculate_consensus_point(
        self,
        provider_data: Dict[str, pd.DataFrame],
        timestamp: datetime,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate consensus values for a specific timestamp.

        Args:
            provider_data: Dictionary mapping provider names to DataFrames
            timestamp: Timestamp to calculate consensus for
            symbol: Trading symbol

        Returns:
            Consensus data point or None if calculation failed
        """
        try:
            # Collect values for this timestamp from all providers
            values = {}
            for provider, data in provider_data.items():
                mask = data['timestamp'] == timestamp
                if mask.any():
                    row = data[mask].iloc[0]
                    values[provider] = {
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }

            if not values:
                return None

            # Calculate consensus values
            consensus_point = {
                'timestamp': timestamp,
                'open': np.median([v['open'] for v in values.values()]),
                'high': np.max([v['high'] for v in values.values()]),
                'low': np.min([v['low'] for v in values.values()]),
                'close': np.median([v['close'] for v in values.values()]),
                'volume': np.sum([v['volume'] for v in values.values()]),
                'provider_count': len(values)
            }

            return consensus_point

        except Exception as e:
            _logger.warning("Failed to calculate consensus for %s: %s",timestamp, e)
            return None

    def synchronize_data(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        symbol: str,
        tolerance: timedelta = timedelta(seconds=1)
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Synchronize two datasets to common timestamps.

        Args:
            data1: First DataFrame
            data2: Second DataFrame
            symbol: Trading symbol
            tolerance: Time tolerance for matching timestamps

        Returns:
            Tuple of synchronized DataFrames
        """
        if data1.empty or data2.empty:
            return data1, data2

        # Find common timestamps within tolerance
        common_timestamps = []
        for ts1 in data1['timestamp']:
            for ts2 in data2['timestamp']:
                if abs((ts1 - ts2).total_seconds()) <= tolerance.total_seconds():
                    common_timestamps.append((ts1, ts2))

        if not common_timestamps:
            _logger.warning("No common timestamps found for %s", symbol)
            return data1, data2

        # Create synchronized datasets
        sync_data1 = []
        sync_data2 = []

        for ts1, ts2 in common_timestamps:
            row1 = data1[data1['timestamp'] == ts1].iloc[0]
            row2 = data2[data2['timestamp'] == ts2].iloc[0]

            sync_data1.append(row1.to_dict())
            sync_data2.append(row2.to_dict())

        sync_df1 = pd.DataFrame(sync_data1)
        sync_df2 = pd.DataFrame(sync_data2)

        _logger.info("Synchronized %d data points for %s", len(sync_df1), symbol)

        return sync_df1, sync_df2

    def compare_data_sources(
        self,
        symbol: str,
        interval: str,
        providers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Compare data quality and consistency across providers.

        Args:
            symbol: Trading symbol
            interval: Data interval
            providers: List of provider names
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'symbol': symbol,
            'interval': interval,
            'providers': providers,
            'data_points': {},
            'quality_scores': {},
            'consistency_metrics': {},
            'timestamp_ranges': {},
            'recommendations': []
        }

        # Collect data and metrics from each provider
        provider_data = {}
        for provider in providers:
            try:
                data_source = self.factory.get_or_create_data_source(provider)
                if data_source:
                    data = data_source.get_data_with_cache(
                        symbol, interval, start_date, end_date
                    )
                    if data is not None and not data.empty:
                        provider_data[provider] = data

                        # Calculate metrics
                        comparison['data_points'][provider] = len(data)
                        comparison['timestamp_ranges'][provider] = {
                            'start': data['timestamp'].min().isoformat(),
                            'end': data['timestamp'].max().isoformat()
                        }

                        # Quality assessment
                        validation_result = self.data_handler.validate_and_score_data(data, symbol)
                        comparison['quality_scores'][provider] = validation_result['quality_score']

                    else:
                        comparison['data_points'][provider] = 0
                        comparison['quality_scores'][provider] = {'quality_score': 0.0}

            except Exception:
                _logger.exception("Failed to analyze %s:", provider)
                comparison['data_points'][provider] = 0
                comparison['quality_scores'][provider] = {'quality_score': 0.0}

        # Calculate consistency metrics
        if len(provider_data) > 1:
            comparison['consistency_metrics'] = self._calculate_consistency_metrics(
                provider_data, symbol
            )

        # Generate recommendations
        comparison['recommendations'] = self._generate_recommendations(comparison)

        return comparison

    def _calculate_consistency_metrics(
        self,
        provider_data: Dict[str, pd.DataFrame],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Calculate consistency metrics across providers.

        Args:
            provider_data: Dictionary mapping provider names to DataFrames
            symbol: Trading symbol

        Returns:
            Dictionary with consistency metrics
        """
        metrics = {}

        try:
            # Find common timestamps
            all_timestamps = set.intersection(
                *[set(data['timestamp']) for data in provider_data.values()]
            )

            if not all_timestamps:
                metrics['common_timestamps'] = 0
                metrics['consistency_score'] = 0.0
                return metrics

            metrics['common_timestamps'] = len(all_timestamps)

            # Calculate price consistency
            price_differences = []
            for ts in sorted(all_timestamps):
                prices = []
                for data in provider_data.values():
                    row = data[data['timestamp'] == ts].iloc[0]
                    prices.append(row['close'])

                if len(prices) > 1:
                    max_price = max(prices)
                    min_price = min(prices)
                    if min_price > 0:
                        price_diff_pct = (max_price - min_price) / min_price * 100
                        price_differences.append(price_diff_pct)

            if price_differences:
                metrics['avg_price_difference_pct'] = np.mean(price_differences)
                metrics['max_price_difference_pct'] = np.max(price_differences)
                metrics['price_consistency_score'] = max(0, 100 - np.mean(price_differences))
            else:
                metrics['avg_price_difference_pct'] = 0.0
                metrics['max_price_difference_pct'] = 0.0
                metrics['price_consistency_score'] = 100.0

            # Overall consistency score
            metrics['consistency_score'] = metrics['price_consistency_score']

        except Exception as e:
            _logger.exception("Failed to calculate consistency metrics:")
            metrics['error'] = str(e)

        return metrics

    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on comparison results.

        Args:
            comparison: Comparison results dictionary

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check data availability
        available_providers = [
            p for p, count in comparison['data_points'].items()
            if count > 0
        ]

        if not available_providers:
            recommendations.append("No data available from any provider")
            return recommendations

        # Quality recommendations
        quality_scores = comparison['quality_scores']
        best_provider = max(quality_scores.items(), key=lambda x: x[1]['quality_score'])

        if best_provider[1]['quality_score'] < 80:
            recommendations.append(f"Data quality is low across providers. Best: {best_provider[0]} ({best_provider[1]['quality_score']:.1f})")

        # Consistency recommendations
        if 'consistency_metrics' in comparison and comparison['consistency_metrics']:
            consistency_score = comparison['consistency_metrics'].get('consistency_score', 0)
            if consistency_score < 90:
                recommendations.append(f"Low data consistency across providers ({consistency_score:.1f})")
            else:
                recommendations.append(f"Good data consistency across providers ({consistency_score:.1f})")

        # Provider recommendations
        if len(available_providers) > 1:
            recommendations.append(f"Multiple providers available: {', '.join(available_providers)}")
            recommendations.append(f"Consider using {best_provider[0]} as primary source")
        else:
            recommendations.append(f"Single provider available: {available_providers[0]}")

        return recommendations
