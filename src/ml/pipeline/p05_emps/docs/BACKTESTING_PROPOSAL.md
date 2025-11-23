# EMPS Historical Testing & Backtesting Proposal

**Version:** 1.0
**Date:** 2025-11-22
**Status:** Proposal for Implementation

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Why Historical Testing is Critical](#why-historical-testing-is-critical)
- [Proposed Testing Framework](#proposed-testing-framework)
- [Historical Test Cases](#historical-test-cases)
- [Implementation Plan](#implementation-plan)
- [Sentiment Integration Analysis](#sentiment-integration-analysis)
- [Expected Outcomes](#expected-outcomes)

---

## Executive Summary

### Current State

EMPS (Explosive Move Probability Score) is a quantitative system designed to detect explosive price movements using 5 intraday components:
- Volume Z-Score (45% weight)
- VWAP Deviation (25% weight)
- Realized Volatility Ratio (25% weight)
- Liquidity Score (5% weight)
- Social Proxy (optional - basic StockTwits count)

**Problem:** EMPS has **NOT been validated** against historical explosive moves. Current thresholds and weights are theoretical.

### Proposed Solution

Implement a comprehensive backtesting framework to:
1. **Validate detection accuracy** - Test against known explosive moves
2. **Optimize parameters** - Find optimal thresholds and weights
3. **Measure performance** - Quantify precision, recall, and timing
4. **Integrate sentiment** - Evaluate value of `src/common/sentiments` module

### Key Benefits

- ✅ **Validate EMPS works** - Prove it detects real explosive moves
- ✅ **Optimize parameters** - Data-driven threshold tuning
- ✅ **Quantify performance** - Measurable metrics (precision/recall)
- ✅ **Reduce false positives** - Improve signal quality
- ✅ **Enhance with sentiment** - Add social media intelligence

---

## Why Historical Testing is Critical

### 1. EMPS is Currently Unvalidated

**Current Parameters are Theoretical:**
```python
DEFAULTS = {
    'vol_zscore_thresh': 4.0,      # Why 4.0? Not tested
    'vwap_dev_thresh': 0.03,       # Why 3%? Not tested
    'rv_ratio_thresh': 1.8,        # Why 1.8? Not tested
    'combined_score_thresh': 0.6,  # Why 0.6? Not tested
    'weights': {
        'vol': 0.45,               # Why 45%? Not tested
        'vwap': 0.25,              # Why 25%? Not tested
        'rv': 0.25,                # Why 25%? Not tested
        'liquidity': 0.05,         # Why 5%? Not tested
    }
}
```

**Risk:** Without validation:
- May miss actual explosive moves (false negatives)
- May trigger on non-explosive moves (false positives)
- Weights may be sub-optimal
- Thresholds may need adjustment

### 2. Historical Data Proves Effectiveness

**Known Explosive Moves (2021-2024):**
- **GME** - January 2021 squeeze ($20 → $483)
- **AMC** - June 2021 spike ($9 → $72)
- **DWAC** - October 2021 (+357% in 2 days)
- **MULN** - March 2022 spike (+200% in 3 days)
- **HKD** - July 2022 explosion ($7.80 → $2,555)
- **BBBY** - August 2022 rally ($4.50 → $30)
- **NEGG** - Multiple 2021-2022 spikes
- **CLOV** - June 2021 squeeze ($6 → $28)
- **SPRT** - August-September 2021 squeeze ($2.50 → $59)
- **RDBX** - June 2022 squeeze ($2 → $18)
- **GETY** - July 2022 spike ($1.50 → $7.50)
- **APRN** - August 2022 squeeze ($3 → $8)
- **ENVX** - November 2021 spike ($13 → $35)
- **RELI** - December 2021 squeeze ($0.50 → $14)
- **ESSC** - December 2021 squeeze ($10 → $24)
- **PROG** - November 2021 squeeze ($0.90 → $6.20)
- **ATER** - September 2021 squeeze ($3 → $19)
- **BGFV** - November 2021 squeeze ($15 → $45)
- **SDC** - August-September 2021 squeeze ($6 → $7.50)
- **CEI** - September 2021 spike ($0.60 → $4.85)
- **AGRI** - June 2022 spike ($3 → $40)
- **SST** - April 2022 IPO spike ($10 → $37)
- **NILE** - March 2022 spike ($0.30 → $1.25)

**Question:** Would EMPS have flagged these BEFORE the major moves?

### 3. Optimize for Real-World Use

**Current unknowns:**
- What score threshold minimizes false positives?
- What component weights maximize detection?
- How early does EMPS detect moves (hours/days before peak)?
- What liquidity parameters work best?

**Backtesting answers these questions.**

---

## Proposed Testing Framework

### Approach 1: Detection Accuracy Testing (Recommended First)

**Goal:** Validate EMPS detects known explosive moves

**Method:**
1. Select 20-30 known explosive moves (2021-2024)
2. Fetch historical 5m intraday data for each
3. Run EMPS calculation on historical data
4. Measure:
   - **Detection rate:** Did EMPS flag the move?
   - **Timing:** How many hours/days before peak?
   - **Score distribution:** What scores did real moves get?
   - **False positive rate:** How often does EMPS flag normal days?

**Implementation:**
```python
# src/ml/pipeline/p05_emps/backtesting/detection_accuracy_test.py

from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

from src.ml.pipeline.p05_emps.emps import compute_emps_from_intraday
from src.ml.pipeline.p05_emps.emps_data_adapter import EMPSDataAdapter
from src.data.downloader.fmp_data_downloader import FMPDataDownloader


@dataclass
class ExplosiveMoveCase:
    """Historical explosive move test case."""
    ticker: str
    event_date: str           # Date of major move
    peak_price: float         # Peak price reached
    start_price: float        # Price before move
    description: str          # Brief description
    expected_detect: bool = True  # Should EMPS detect this?


# Test cases - Known explosive moves (20+ cases)
EXPLOSIVE_MOVES = [
    # 2021 - GME Squeeze (The OG)
    ExplosiveMoveCase(
        ticker='GME',
        event_date='2021-01-27',  # Peak day
        start_price=20.0,
        peak_price=483.0,
        description='GameStop short squeeze - The OG',
        expected_detect=True
    ),

    # 2021 - AMC Squeeze
    ExplosiveMoveCase(
        ticker='AMC',
        event_date='2021-06-02',
        start_price=9.0,
        peak_price=72.62,
        description='AMC Entertainment squeeze',
        expected_detect=True
    ),

    # 2021 - CLOV Squeeze
    ExplosiveMoveCase(
        ticker='CLOV',
        event_date='2021-06-09',
        start_price=6.0,
        peak_price=28.0,
        description='Clover Health short squeeze',
        expected_detect=True
    ),

    # 2021 - SPRT Squeeze
    ExplosiveMoveCase(
        ticker='SPRT',
        event_date='2021-08-27',
        start_price=2.50,
        peak_price=59.0,
        description='Support.com merger squeeze',
        expected_detect=True
    ),

    # 2021 - ATER Squeeze
    ExplosiveMoveCase(
        ticker='ATER',
        event_date='2021-09-10',
        start_price=3.0,
        peak_price=19.0,
        description='Aterian squeeze',
        expected_detect=True
    ),

    # 2021 - CEI Spike
    ExplosiveMoveCase(
        ticker='CEI',
        event_date='2021-09-27',
        start_price=0.60,
        peak_price=4.85,
        description='Camber Energy spike',
        expected_detect=True
    ),

    # 2021 - SDC Squeeze
    ExplosiveMoveCase(
        ticker='SDC',
        event_date='2021-09-13',
        start_price=6.0,
        peak_price=7.50,
        description='SmileDirectClub squeeze attempt',
        expected_detect=True
    ),

    # 2021 - DWAC Trump SPAC
    ExplosiveMoveCase(
        ticker='DWAC',
        event_date='2021-10-22',
        start_price=10.0,
        peak_price=175.0,
        description='Trump SPAC announcement',
        expected_detect=True
    ),

    # 2021 - PROG Squeeze
    ExplosiveMoveCase(
        ticker='PROG',
        event_date='2021-11-16',
        start_price=0.90,
        peak_price=6.20,
        description='Progenity squeeze',
        expected_detect=True
    ),

    # 2021 - BGFV Squeeze
    ExplosiveMoveCase(
        ticker='BGFV',
        event_date='2021-11-17',
        start_price=15.0,
        peak_price=45.0,
        description='Big 5 Sporting Goods squeeze',
        expected_detect=True
    ),

    # 2021 - ENVX Spike
    ExplosiveMoveCase(
        ticker='ENVX',
        event_date='2021-11-18',
        start_price=13.0,
        peak_price=35.0,
        description='Enovix IPO spike',
        expected_detect=True
    ),

    # 2021 - RELI Squeeze
    ExplosiveMoveCase(
        ticker='RELI',
        event_date='2021-12-13',
        start_price=0.50,
        peak_price=14.0,
        description='Reliance Global short squeeze',
        expected_detect=True
    ),

    # 2021 - ESSC Squeeze
    ExplosiveMoveCase(
        ticker='ESSC',
        event_date='2021-12-07',
        start_price=10.0,
        peak_price=24.0,
        description='East Stone Acquisition de-SPAC squeeze',
        expected_detect=True
    ),

    # 2022 - NILE Spike
    ExplosiveMoveCase(
        ticker='NILE',
        event_date='2022-03-25',
        start_price=0.30,
        peak_price=1.25,
        description='BitNile Holdings spike',
        expected_detect=True
    ),

    # 2022 - SST IPO Spike
    ExplosiveMoveCase(
        ticker='SST',
        event_date='2022-04-04',
        start_price=10.0,
        peak_price=37.0,
        description='System1 de-SPAC spike',
        expected_detect=True
    ),

    # 2022 - RDBX Squeeze
    ExplosiveMoveCase(
        ticker='RDBX',
        event_date='2022-06-07',
        start_price=2.0,
        peak_price=18.0,
        description='Redbox Entertainment squeeze',
        expected_detect=True
    ),

    # 2022 - AGRI Spike
    ExplosiveMoveCase(
        ticker='AGRI',
        event_date='2022-06-08',
        start_price=3.0,
        peak_price=40.0,
        description='AgriFORCE Growing Systems spike',
        expected_detect=True
    ),

    # 2022 - GETY Spike
    ExplosiveMoveCase(
        ticker='GETY',
        event_date='2022-07-08',
        start_price=1.50,
        peak_price=7.50,
        description='Getty Images IPO spike',
        expected_detect=True
    ),

    # 2022 - HKD IPO Mania (EXTREME)
    ExplosiveMoveCase(
        ticker='HKD',
        event_date='2022-08-02',
        start_price=7.80,
        peak_price=2555.0,
        description='AMTD Digital IPO explosion - Most extreme ever',
        expected_detect=True
    ),

    # 2022 - BBBY Gamma Squeeze
    ExplosiveMoveCase(
        ticker='BBBY',
        event_date='2022-08-16',
        start_price=4.50,
        peak_price=30.0,
        description='Bed Bath & Beyond squeeze',
        expected_detect=True
    ),

    # 2022 - APRN Squeeze
    ExplosiveMoveCase(
        ticker='APRN',
        event_date='2022-08-24',
        start_price=3.0,
        peak_price=8.0,
        description='Blue Apron squeeze',
        expected_detect=True
    ),

    # Control cases - Normal stocks (should NOT trigger)
    ExplosiveMoveCase(
        ticker='AAPL',
        event_date='2023-05-15',
        start_price=170.0,
        peak_price=175.0,
        description='Apple normal trading day',
        expected_detect=False
    ),

    ExplosiveMoveCase(
        ticker='MSFT',
        event_date='2023-06-20',
        start_price=330.0,
        peak_price=340.0,
        description='Microsoft normal trading day',
        expected_detect=False
    ),

    ExplosiveMoveCase(
        ticker='SPY',
        event_date='2023-07-10',
        start_price=440.0,
        peak_price=445.0,
        description='SPY ETF normal day',
        expected_detect=False
    ),

    ExplosiveMoveCase(
        ticker='TSLA',
        event_date='2023-08-15',
        start_price=240.0,
        peak_price=250.0,
        description='Tesla normal volatility day',
        expected_detect=False
    ),
]


class EMPSDetectionAccuracyTest:
    """Test EMPS detection accuracy on historical explosive moves."""

    def __init__(self, fmp: FMPDataDownloader):
        self.fmp = fmp
        self.adapter = EMPSDataAdapter(fmp)
        self.results = []

    def run_test(self, test_case: ExplosiveMoveCase) -> Dict[str, Any]:
        """
        Test EMPS on a single historical case.

        Returns:
            Dict with test results
        """
        print(f"\n{'='*70}")
        print(f"Testing: {test_case.ticker} - {test_case.description}")
        print(f"Event Date: {test_case.event_date}")
        print(f"Price Move: ${test_case.start_price} → ${test_case.peak_price} "
              f"({((test_case.peak_price/test_case.start_price - 1) * 100):.1f}%)")
        print(f"{'='*70}")

        # Fetch historical data around event
        event_date = datetime.strptime(test_case.event_date, '%Y-%m-%d')

        # Get 5 days before event (to see if EMPS detected early)
        start_date = event_date - timedelta(days=5)

        # Fetch intraday data
        # Note: FMP historical intraday may have limitations
        # May need alternative data source for older data
        df_intraday = self._fetch_historical_intraday(
            test_case.ticker,
            start_date,
            event_date
        )

        if df_intraday.empty:
            print(f"[WARNING] No historical data available for {test_case.ticker}")
            return {
                'ticker': test_case.ticker,
                'event_date': test_case.event_date,
                'detected': False,
                'error': 'No historical data'
            }

        # Get metadata
        metadata = self.adapter.fetch_ticker_metadata(test_case.ticker)

        # Compute EMPS
        df_emps = compute_emps_from_intraday(
            df_intraday,
            market_cap=metadata.get('market_cap'),
            float_shares=metadata.get('float_shares'),
            avg_volume=metadata.get('avg_volume'),
            ticker=test_case.ticker
        )

        # Analyze results
        analysis = self._analyze_detection(df_emps, test_case)

        # Print summary
        self._print_analysis(analysis, test_case)

        return analysis

    def _fetch_historical_intraday(self, ticker: str, start_date: datetime,
                                   end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical intraday data.

        Note: FMP may have limitations on historical intraday data.
        May need to use alternative sources or local cache.
        """
        # TODO: Implement historical data fetching
        # Options:
        # 1. FMP historical intraday (limited to recent months)
        # 2. Local cache of historical data
        # 3. Alternative data provider (Polygon, Alpaca, etc.)

        # Placeholder - would need actual implementation
        try:
            # Use adapter with date range
            df = self.adapter.fetch_intraday_for_emps(
                ticker,
                {
                    'interval': '5m',
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d')
                }
            )
            return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def _analyze_detection(self, df_emps: pd.DataFrame,
                          test_case: ExplosiveMoveCase) -> Dict[str, Any]:
        """Analyze if EMPS detected the explosive move."""

        # Find explosion flags
        explosions = df_emps[df_emps['explosion_flag'] == True]

        detected = len(explosions) > 0

        if detected:
            # Get first detection
            first_flag = explosions.iloc[0]
            max_score = df_emps['emps_score'].max()
            avg_score = explosions['emps_score'].mean()

            # Timing analysis
            event_date = datetime.strptime(test_case.event_date, '%Y-%m-%d')
            first_detect_time = first_flag['emps_timestamp']

            # Calculate hours before peak
            hours_before_peak = (event_date - first_detect_time).total_seconds() / 3600

            return {
                'ticker': test_case.ticker,
                'event_date': test_case.event_date,
                'detected': True,
                'first_detection_time': first_detect_time,
                'hours_before_peak': hours_before_peak,
                'max_emps_score': max_score,
                'avg_explosion_score': avg_score,
                'num_explosion_flags': len(explosions),
                'expected_detect': test_case.expected_detect,
                'correct': detected == test_case.expected_detect,
                'components': {
                    'max_vol_zscore': df_emps['vol_zscore'].max(),
                    'max_vwap_dev': df_emps['vwap_dev'].abs().max(),
                    'max_rv_ratio': df_emps['rv_ratio'].max(),
                }
            }
        else:
            # Not detected
            max_score = df_emps['emps_score'].max() if not df_emps.empty else 0.0

            return {
                'ticker': test_case.ticker,
                'event_date': test_case.event_date,
                'detected': False,
                'max_emps_score': max_score,
                'expected_detect': test_case.expected_detect,
                'correct': detected == test_case.expected_detect,
            }

    def _print_analysis(self, analysis: Dict[str, Any],
                       test_case: ExplosiveMoveCase):
        """Print analysis results."""

        if analysis['detected']:
            print(f"\n[DETECTED] EMPS flagged explosion")
            print(f"  First Detection: {analysis['first_detection_time']}")
            print(f"  Hours Before Peak: {analysis['hours_before_peak']:.1f} hours")
            print(f"  Max EMPS Score: {analysis['max_emps_score']:.3f}")
            print(f"  Avg Explosion Score: {analysis['avg_explosion_score']:.3f}")
            print(f"  Explosion Flags: {analysis['num_explosion_flags']}")
            print(f"\n  Component Peaks:")
            print(f"    Vol Z-Score: {analysis['components']['max_vol_zscore']:.2f}")
            print(f"    VWAP Dev: {analysis['components']['max_vwap_dev']:.3f}")
            print(f"    RV Ratio: {analysis['components']['max_rv_ratio']:.2f}")
        else:
            print(f"\n[NOT DETECTED] EMPS did not flag explosion")
            print(f"  Max EMPS Score: {analysis['max_emps_score']:.3f}")

        # Correctness
        if analysis['correct']:
            print(f"\n[CORRECT] Expected: {test_case.expected_detect}, "
                  f"Got: {analysis['detected']}")
        else:
            print(f"\n[INCORRECT] Expected: {test_case.expected_detect}, "
                  f"Got: {analysis['detected']}")

    def run_all_tests(self) -> pd.DataFrame:
        """Run all test cases and return summary DataFrame."""

        print("\n" + "="*70)
        print("EMPS DETECTION ACCURACY TEST SUITE")
        print("="*70)

        for test_case in EXPLOSIVE_MOVES:
            result = self.run_test(test_case)
            self.results.append(result)

        # Create summary DataFrame
        df_results = pd.DataFrame(self.results)

        # Print summary
        self._print_summary(df_results)

        return df_results

    def _print_summary(self, df: pd.DataFrame):
        """Print test summary."""

        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        total = len(df)
        detected = df['detected'].sum()
        correct = df['correct'].sum()

        # Split by expected
        expected_detect = df[df['expected_detect'] == True]
        expected_no_detect = df[df['expected_detect'] == False]

        true_positives = len(expected_detect[expected_detect['detected'] == True])
        false_negatives = len(expected_detect[expected_detect['detected'] == False])

        true_negatives = len(expected_no_detect[expected_no_detect['detected'] == False])
        false_positives = len(expected_no_detect[expected_no_detect['detected'] == True])

        print(f"\nOverall:")
        print(f"  Total Cases: {total}")
        print(f"  Detected: {detected}")
        print(f"  Accuracy: {(correct/total*100):.1f}%")

        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {true_positives} (correctly detected explosive moves)")
        print(f"  False Negatives: {false_negatives} (missed explosive moves)")
        print(f"  True Negatives: {true_negatives} (correctly ignored normal moves)")
        print(f"  False Positives: {false_positives} (incorrectly flagged normal moves)")

        if len(expected_detect) > 0:
            recall = true_positives / len(expected_detect)
            print(f"\nRecall (Detection Rate): {recall*100:.1f}%")

        if detected > 0:
            precision = true_positives / detected
            print(f"Precision: {precision*100:.1f}%")

        # Timing analysis
        detected_df = df[df['detected'] == True]
        if not detected_df.empty and 'hours_before_peak' in detected_df.columns:
            avg_hours = detected_df['hours_before_peak'].mean()
            print(f"\nAverage Detection Timing: {avg_hours:.1f} hours before peak")


# Example usage
if __name__ == "__main__":
    from src.data.downloader.fmp_data_downloader import FMPDataDownloader

    print("EMPS Detection Accuracy Test")
    print("="*70)

    # Initialize
    fmp = FMPDataDownloader()

    if not fmp.test_connection():
        print("ERROR: FMP connection failed")
        exit(1)

    # Run tests
    tester = EMPSDetectionAccuracyTest(fmp)
    results = tester.run_all_tests()

    # Save results
    results.to_csv('emps_detection_accuracy_results.csv', index=False)
    print(f"\nResults saved to: emps_detection_accuracy_results.csv")
```

**Expected Output:**
```
==================================================
EMPS DETECTION ACCURACY TEST SUITE
==================================================

Testing: GME - GameStop short squeeze
Event Date: 2021-01-27
Price Move: $20.0 → $483.0 (2315.0%)
==================================================

[DETECTED] EMPS flagged explosion
  First Detection: 2021-01-26 14:30:00
  Hours Before Peak: 18.5 hours
  Max EMPS Score: 0.892
  Avg Explosion Score: 0.743
  Explosion Flags: 24

  Component Peaks:
    Vol Z-Score: 8.45
    VWAP Dev: 0.078
    RV Ratio: 3.21

[CORRECT] Expected: True, Got: True

... [more tests] ...

==================================================
TEST SUMMARY
==================================================

Overall:
  Total Cases: 7
  Detected: 5
  Accuracy: 85.7%

Confusion Matrix:
  True Positives: 4 (correctly detected explosive moves)
  False Negatives: 1 (missed explosive moves)
  True Negatives: 2 (correctly ignored normal moves)
  False Positives: 0 (incorrectly flagged normal moves)

Recall (Detection Rate): 80.0%
Precision: 100.0%

Average Detection Timing: 16.3 hours before peak
```

---

### Approach 2: Parameter Optimization

**Goal:** Find optimal thresholds and weights

**Method:**
1. Use grid search or Bayesian optimization
2. Test different parameter combinations
3. Maximize: `precision * recall` (F1-score)
4. Minimize: false positive rate

**Implementation:**
```python
# src/ml/pipeline/p05_emps/backtesting/parameter_optimizer.py

from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd

from src.ml.pipeline.p05_emps.emps import DEFAULTS


class EMPSParameterOptimizer:
    """Optimize EMPS parameters using historical test cases."""

    def __init__(self, test_cases: List[ExplosiveMoveCase]):
        self.test_cases = test_cases
        self.best_params = None
        self.best_score = 0.0

    def define_param_grid(self) -> Dict[str, List[Any]]:
        """Define parameter search space."""

        return {
            'vol_zscore_thresh': [3.0, 3.5, 4.0, 4.5, 5.0],
            'vwap_dev_thresh': [0.02, 0.025, 0.03, 0.035, 0.04],
            'rv_ratio_thresh': [1.5, 1.6, 1.8, 2.0, 2.2],
            'combined_score_thresh': [0.5, 0.55, 0.6, 0.65, 0.7],
            'weight_vol': [0.40, 0.45, 0.50, 0.55],
            'weight_vwap': [0.20, 0.25, 0.30],
            'weight_rv': [0.20, 0.25, 0.30],
            # weight_liquidity = 1.0 - others
        }

    def evaluate_params(self, params: Dict[str, Any]) -> float:
        """
        Evaluate a parameter set.

        Returns:
            F1 score (harmonic mean of precision and recall)
        """

        # Build EMPS params dict
        emps_params = {
            **DEFAULTS,
            'vol_zscore_thresh': params['vol_zscore_thresh'],
            'vwap_dev_thresh': params['vwap_dev_thresh'],
            'rv_ratio_thresh': params['rv_ratio_thresh'],
            'combined_score_thresh': params['combined_score_thresh'],
            'weights': {
                'vol': params['weight_vol'],
                'vwap': params['weight_vwap'],
                'rv': params['weight_rv'],
                'liquidity': 1.0 - (params['weight_vol'] +
                                   params['weight_vwap'] +
                                   params['weight_rv']),
            }
        }

        # Run detection test with these params
        # ... (integrate with EMPSDetectionAccuracyTest)

        # Calculate metrics
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # f1 = 2 * (precision * recall) / (precision + recall)

        # Placeholder
        f1_score = 0.85  # Would be calculated from actual tests

        return f1_score

    def optimize(self) -> Dict[str, Any]:
        """Run grid search optimization."""

        param_grid = self.define_param_grid()
        grid = ParameterGrid(param_grid)

        print(f"Testing {len(grid)} parameter combinations...")

        for i, params in enumerate(grid):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(grid)}")

            score = self.evaluate_params(params)

            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                print(f"  New best! F1={score:.3f}, Params: {params}")

        print(f"\nOptimization complete!")
        print(f"Best F1 Score: {self.best_score:.3f}")
        print(f"Best Parameters: {self.best_params}")

        return self.best_params
```

---

### Approach 3: Integration with Existing Backtester

**Goal:** Use existing backtesting infrastructure

**Current backtester capabilities:**
- Supports custom strategies
- Walks forward validation
- Performance metrics (Sharpe, drawdown, etc.)
- JSON configuration

**EMPS as Entry Signal:**
```python
# src/backtester/strategies/emps_strategy.py

import backtrader as bt
from src.ml.pipeline.p05_emps.emps import compute_emps_from_intraday


class EMPSStrategy(bt.Strategy):
    """Backtrader strategy using EMPS for entry signals."""

    params = (
        ('emps_threshold', 0.6),
        ('hold_days', 3),
    )

    def __init__(self):
        self.emps_score = None
        self.days_held = 0

    def next(self):
        # Calculate EMPS on recent bars
        df_recent = self.get_recent_bars(n=200)  # Get recent intraday data

        df_emps = compute_emps_from_intraday(df_recent, ticker=self.data._name)

        latest_score = df_emps.iloc[-1]['emps_score']
        explosion_flag = df_emps.iloc[-1]['explosion_flag']

        # Entry: EMPS explosion flag
        if not self.position and explosion_flag:
            self.buy()
            self.days_held = 0

        # Exit: Hold for N days or stop loss
        elif self.position:
            self.days_held += 1

            # Exit after hold period
            if self.days_held >= self.params.hold_days:
                self.sell()

            # Stop loss
            elif self.data.close[0] < self.position.price * 0.95:
                self.sell()
```

**Config:**
```json
{
  "strategy": "EMPSStrategy",
  "tickers": ["GME", "AMC", "BBBY"],
  "start_date": "2021-01-01",
  "end_date": "2022-12-31",
  "initial_cash": 10000,
  "commission": 0.001,
  "params": {
    "emps_threshold": 0.6,
    "hold_days": 3
  }
}
```

---

## Historical Test Cases

### Confirmed Explosive Moves (Should Detect)

#### 2021 Meme Stock Era

**GME - January 2021**
- Date: 2021-01-27 (peak)
- Price: $20 → $483 (2,315% gain)
- Characteristics:
  - Volume: 30-50x normal
  - VWAP deviation: 10-15%
  - Volatility: Extreme acceleration
- **EMPS Should:** Detect 1-2 days before peak

**AMC - June 2021**
- Date: 2021-06-02
- Price: $9 → $72.62 (706% gain)
- Characteristics:
  - Volume: 20-40x normal
  - VWAP deviation: 8-12%
  - Volatility: High acceleration
- **EMPS Should:** Detect during early runup

#### 2021 SPAC Mania

**DWAC - October 2021**
- Date: 2021-10-22
- Price: $10 → $175 (1,650% gain)
- Trigger: Trump SPAC announcement
- **EMPS Should:** Detect on announcement day

**IRNT - September 2021**
- Date: 2021-09-16
- Price: $10 → $47 (370% gain)
- De-SPAC redemption squeeze
- **EMPS Should:** Detect during squeeze

#### 2022 Events

**HKD - August 2022**
- Date: 2022-08-02
- Price: $7.80 → $2,555 (32,656% gain)
- Most extreme IPO explosion in history
- **EMPS Should:** Definitely detect (ultimate test)

**BBBY - August 2022**
- Date: 2022-08-16
- Price: $4.50 → $30 (567% gain)
- RC Ventures + short squeeze
- **EMPS Should:** Detect early stages

### Control Cases (Should NOT Detect)

**AAPL - Normal Trading**
- Date: Various 2023 dates
- Normal 2-3% daily ranges
- **EMPS Should:** Not flag

**MSFT - Normal Trading**
- Date: Various 2023 dates
- Steady trend, no explosions
- **EMPS Should:** Not flag

**SPY - Market Days**
- Date: Various
- Broad market ETF
- **EMPS Should:** Not flag

---

## Implementation Plan

### Phase 1: Historical Data Collection (Week 1-2)

**Objective:** Acquire historical intraday data

**Tasks:**
1. **Evaluate data sources:**
   - FMP historical intraday (limited availability)
   - Polygon.io (good historical coverage)
   - Alpaca (free historical data)
   - Local CSV storage (cache for reuse)

2. **Download test case data:**
   - Fetch 5m bars for each test case
   - 5 days before event + event day
   - Store locally for repeated testing

3. **Create data loader:**
   ```python
   # src/ml/pipeline/p05_emps/backtesting/historical_data_loader.py

   class HistoricalDataLoader:
       """Load historical intraday data for backtesting."""

       def load_event_data(self, ticker: str, event_date: str,
                          days_before: int = 5) -> pd.DataFrame:
           """Load historical data around an event."""
           pass
   ```

**Success Criteria:**
- ✅ Historical data for all 20+ test cases
- ✅ Data quality validated (no gaps, correct format)
- ✅ Cached locally for fast access

---

### Phase 2: Detection Accuracy Testing (Week 3)

**Objective:** Validate EMPS detects explosive moves

**Tasks:**
1. Implement `EMPSDetectionAccuracyTest` (code above)
2. Run tests on all historical cases
3. Generate report with metrics:
   - Recall (detection rate)
   - Precision (accuracy)
   - Timing (hours before peak)
   - Component analysis

**Success Criteria:**
- ✅ Recall > 80% (detects 80%+ of explosive moves)
- ✅ Precision > 70% (70%+ of flags are real)
- ✅ Average detection: 12-24 hours before peak

---

### Phase 3: Parameter Optimization (Week 4-5)

**Objective:** Find optimal EMPS parameters

**Tasks:**
1. Implement `EMPSParameterOptimizer`
2. Run grid search (1,000-5,000 combinations)
3. Use Bayesian optimization for efficiency
4. Validate optimized params on holdout set

**Success Criteria:**
- ✅ Improved F1 score vs. defaults
- ✅ Reduced false positive rate
- ✅ Validated on separate test set

---

### Phase 4: Sentiment Integration (Week 6)

**Objective:** Evaluate sentiment module value

**Tasks:**
1. Integrate `src/common/sentiments` (see section below)
2. Re-run tests with sentiment enhancement
3. Compare performance: EMPS vs. EMPS+Sentiment
4. Optimize sentiment weight

**Success Criteria:**
- ✅ Sentiment improves detection by 5-10%
- ✅ Reduces false positives
- ✅ Provides early warning signal

---

### Phase 5: Backtrader Integration (Week 7-8)

**Objective:** Backtest EMPS as trading signal

**Tasks:**
1. Create `EMPSStrategy` for backtrader
2. Backtest on 2021-2024 data
3. Measure trading performance:
   - Win rate
   - Profit factor
   - Sharpe ratio
   - Max drawdown

**Success Criteria:**
- ✅ Win rate > 60%
- ✅ Profit factor > 2.0
- ✅ Sharpe ratio > 1.5

---

### Phase 6: Documentation & Deployment (Week 9)

**Objective:** Document findings and deploy

**Tasks:**
1. Create comprehensive backtest report
2. Update EMPS parameters with optimized values
3. Add backtesting module to codebase
4. Create usage documentation

**Deliverables:**
- `BACKTEST_RESULTS.md` - Comprehensive results
- `emps_backtest.py` - Reusable backtesting module
- Updated `DEFAULTS` with optimized parameters

---

## Sentiment Integration Analysis

### Current State: Basic Social Proxy

**EMPS currently uses:**
- StockTwits message count (raw count)
- No sentiment analysis
- No virality detection
- No bot filtering

```python
# emps.py - Current social proxy
def fetch_stocktwits_count(ticker):
    """Fetch raw message count from StockTwits."""
    # Simple count, no sentiment
    pass
```

### Proposed: Full Sentiment Integration

**`src/common/sentiments` provides:**

```python
@dataclass
class SentimentFeatures:
    ticker: str
    mentions_24h: int                    # Volume
    unique_authors_24h: int              # Authenticity
    mentions_growth_7d: Optional[float]  # Momentum
    positive_ratio_24h: Optional[float]  # Sentiment direction
    sentiment_score_24h: float           # -1 to +1
    sentiment_normalized: float          # 0 to 1 (ready for scoring)
    virality_index: float                # Viral coefficient
    bot_pct: float                       # Bot detection (0-1)
    data_quality: Dict[str, str]         # Provider health
    raw_payload: Dict[str, Any]          # Full data
```

**Key features:**
- ✅ **Multi-source** - StockTwits, Reddit, Twitter (optional)
- ✅ **Advanced NLP** - HuggingFace sentiment analysis
- ✅ **Bot detection** - Filters fake engagement
- ✅ **Virality metrics** - Detects viral spread
- ✅ **Quality scoring** - Data confidence levels

---

### Integration Proposal

**Add Sentiment as 6th Component:**

```python
# emps.py - Enhanced with sentiment

DEFAULTS = {
    # ... existing params ...

    # NEW: Sentiment component
    'sentiment': {
        'enabled': True,
        'weight': 0.10,  # 10% weight (reduce others to 90%)
        'min_mentions_threshold': 20,  # Minimum mentions for confidence
        'virality_threshold': 2.0,     # Viral coefficient > 2.0
        'bot_filter_max': 0.3,         # Max 30% bot activity
    },

    # Adjusted weights (total = 1.0)
    'weights': {
        'vol': 0.40,          # 45% → 40%
        'vwap': 0.22,         # 25% → 22%
        'rv': 0.22,           # 25% → 22%
        'liquidity': 0.06,    # 5% → 6%
        'sentiment': 0.10,    # NEW
    },
}


def compute_sentiment_score(ticker: str,
                            lookback_hours: int = 24) -> Dict[str, Any]:
    """
    Compute sentiment score using src/common/sentiments module.

    Returns:
        Dict with sentiment score and metadata
    """
    from src.common.sentiments.collect_sentiment_async import (
        collect_sentiment_batch_sync,
        SentimentFeatures
    )

    try:
        # Collect sentiment data
        results = collect_sentiment_batch_sync(
            tickers=[ticker],
            lookback_hours=lookback_hours
        )

        if not results or ticker not in results:
            return {
                'sentiment_score': 0.0,
                'sentiment_normalized': 0.5,  # Neutral
                'mentions': 0,
                'virality': 0.0,
                'quality': 'missing'
            }

        sentiment: SentimentFeatures = results[ticker]

        # Apply filters
        if sentiment.mentions_24h < DEFAULTS['sentiment']['min_mentions_threshold']:
            # Not enough data
            return {
                'sentiment_score': 0.0,
                'sentiment_normalized': 0.5,
                'mentions': sentiment.mentions_24h,
                'virality': sentiment.virality_index,
                'quality': 'insufficient_mentions'
            }

        if sentiment.bot_pct > DEFAULTS['sentiment']['bot_filter_max']:
            # Too many bots
            return {
                'sentiment_score': 0.0,
                'sentiment_normalized': 0.5,
                'mentions': sentiment.mentions_24h,
                'virality': sentiment.virality_index,
                'quality': 'high_bot_activity'
            }

        # Calculate composite sentiment score
        sentiment_score = (
            0.4 * sentiment.sentiment_normalized +        # Base sentiment
            0.3 * min(1.0, sentiment.virality_index / 3.0) +  # Virality
            0.3 * min(1.0, sentiment.mentions_24h / 100.0)    # Mentions volume
        )

        return {
            'sentiment_score': sentiment_score,
            'sentiment_normalized': sentiment.sentiment_normalized,
            'mentions': sentiment.mentions_24h,
            'virality': sentiment.virality_index,
            'bot_pct': sentiment.bot_pct,
            'quality': 'ok'
        }

    except Exception as e:
        logger.warning(f"Sentiment collection failed for {ticker}: {e}")
        return {
            'sentiment_score': 0.0,
            'sentiment_normalized': 0.5,
            'mentions': 0,
            'virality': 0.0,
            'quality': 'error'
        }


# Add to compute_emps_from_intraday
def compute_emps_from_intraday(df: pd.DataFrame,
                               ticker: str,
                               params: Optional[Dict] = None,
                               use_sentiment: bool = True,
                               **kwargs) -> pd.DataFrame:
    """Compute EMPS with optional sentiment."""

    # ... existing calculations ...

    # NEW: Add sentiment component
    if use_sentiment and params.get('sentiment', {}).get('enabled', True):
        sentiment_data = compute_sentiment_score(ticker)

        # Add to DataFrame (constant for all rows since daily metric)
        df['sentiment_score'] = sentiment_data['sentiment_score']
        df['sentiment_mentions'] = sentiment_data['mentions']
        df['sentiment_virality'] = sentiment_data['virality']
        df['sentiment_quality'] = sentiment_data['quality']

        # Incorporate into EMPS score
        sentiment_weight = params.get('weights', {}).get('sentiment', 0.10)

        df['emps_score'] = (
            df['emps_score'] * (1.0 - sentiment_weight) +  # Reduce existing
            df['sentiment_score'] * sentiment_weight        # Add sentiment
        )

    return df
```

---

### Expected Benefits

**1. Early Warning Signal**

Sentiment often precedes price moves:
- Reddit/StockTwits buzz → 12-24 hours → Price move
- EMPS detects sentiment spike → Alert before volume spike

**Example (GME):**
```
Jan 25, 2021: Reddit mentions explode (10x normal)
Jan 26, 2021: EMPS detects high sentiment + increasing volume
Jan 27, 2021: Massive price explosion

With sentiment: 48-hour advance warning
Without sentiment: 24-hour advance warning
```

**2. False Positive Reduction**

High EMPS score + low sentiment = Likely false positive
High EMPS score + high sentiment = Confirmed signal

**3. Virality Detection**

Virality index tracks exponential spread:
- Viral coefficient > 2.0 = Exponential growth
- Indicates crowd momentum building

---

### Backtesting Sentiment Value

**Test Plan:**

1. **Baseline:** EMPS without sentiment
2. **Enhanced:** EMPS with sentiment
3. **Compare:**
   - Detection rate (recall)
   - False positive rate
   - Early warning time
   - F1 score

**Expected Results:**
```
Metric                  | EMPS Only | EMPS + Sentiment | Improvement
------------------------|-----------|------------------|------------
Recall                  | 80%       | 85-90%           | +5-10%
Precision               | 70%       | 75-80%           | +5-10%
False Positive Rate     | 30%       | 20-25%           | -5-10%
Avg Hours Before Peak   | 18 hours  | 24-36 hours      | +6-18 hours
F1 Score                | 0.75      | 0.80-0.85        | +0.05-0.10
```

---

### Implementation Complexity

**Low Complexity:**
- ✅ `src/common/sentiments` is already built
- ✅ Clean API: `collect_sentiment_batch_sync()`
- ✅ Returns structured `SentimentFeatures`
- ✅ No additional dependencies needed

**Integration Steps:**

1. **Add import** (1 line)
```python
from src.common.sentiments.collect_sentiment_async import collect_sentiment_batch_sync
```

2. **Call sentiment API** (5 lines)
```python
results = collect_sentiment_batch_sync(tickers=[ticker])
sentiment = results.get(ticker)
```

3. **Incorporate into score** (10 lines)
```python
sentiment_score = calculate_sentiment_component(sentiment)
emps_score = combine_with_weights(emps_score, sentiment_score)
```

**Total effort:** 2-3 hours

---

### Recommendation: **YES, Integrate Sentiment**

**Reasons:**

1. ✅ **Already built** - `src/common/sentiments` is production-ready
2. ✅ **Low cost** - Easy integration, no new dependencies
3. ✅ **High value** - Early warning, false positive reduction
4. ✅ **Proven useful** - Sentiment precedes price moves
5. ✅ **Testable** - Can backtest to measure actual value

**Proposed Timeline:**
- Phase 4 (Week 6) of implementation plan
- After validating baseline EMPS
- Before final optimization

---

## Expected Outcomes

### Phase 1-2: Validation

**Goal:** Confirm EMPS works

**Expected Results:**
- Recall: 75-85% (detects most explosive moves)
- Precision: 65-75% (reasonable false positive rate)
- Timing: 12-24 hours advance warning

**If Results are Poor:**
- Adjust thresholds
- Re-weight components
- Add sentiment (Phase 4)

---

### Phase 3: Optimization

**Goal:** Optimize parameters

**Expected Improvements:**
- Recall: +5-10% (better detection)
- Precision: +10-15% (fewer false positives)
- F1 Score: +10-20%

---

### Phase 4: Sentiment Enhancement

**Goal:** Add social intelligence

**Expected Improvements:**
- Recall: +5% (catch more via early signals)
- Precision: +5-10% (filter false positives)
- Early warning: +6-12 hours

---

### Phase 5: Trading Performance

**Goal:** Validate as trading signal

**Expected Results:**
- Win rate: 55-65%
- Profit factor: 1.5-2.5
- Sharpe ratio: 1.0-1.8
- Max drawdown: -15% to -25%

---

## Conclusion

### Summary

**Historical testing is CRITICAL** to validate EMPS and should be implemented ASAP.

**Key Points:**

1. ✅ **EMPS is currently unvalidated** - No testing on real explosive moves
2. ✅ **Known explosive moves exist** - Perfect test cases (GME, AMC, HKD, etc.)
3. ✅ **Framework is straightforward** - Detection accuracy test + parameter optimization
4. ✅ **Sentiment integration makes sense** - Low cost, high value
5. ✅ **8-week timeline** - Reasonable scope, clear milestones

### Recommended Action Plan

**Immediate (Week 1-2):**
- Acquire historical data for test cases
- Implement detection accuracy test
- Run initial validation

**Short-term (Week 3-5):**
- Parameter optimization
- Integrate sentiment module
- Re-test with optimizations

**Medium-term (Week 6-9):**
- Backtrader integration
- Trading performance validation
- Documentation and deployment

### Success Criteria

**Minimum Viable:**
- Recall > 75%
- Precision > 65%
- F1 Score > 0.70

**Target:**
- Recall > 85%
- Precision > 75%
- F1 Score > 0.80

**Stretch:**
- Recall > 90%
- Precision > 80%
- F1 Score > 0.85

---

**Status:** Ready for Implementation
**Priority:** High
**Estimated Effort:** 8-9 weeks (1-2 engineers)
**Dependencies:** Historical intraday data access

---

**Next Steps:**

1. Review and approve this proposal
2. Select data provider for historical data
3. Begin Phase 1 (data collection)
4. Set up tracking for backtest results
