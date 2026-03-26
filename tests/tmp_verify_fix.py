import sys
from unittest.mock import patch
import pandas as pd
from src.ml.pipeline.p14_ath.run_ath_scan import main
from src.ml.pipeline.p14_ath.config import ATHPipelineConfig

def test_parsing():
    test_cases = [
        # Case 1: No arguments (should use ATHPipelineConfig defaults)
        {
            "argv": ["run_ath_scan.py"],
            "expected_count": 9  # Based on ATHPipelineConfig.create_default()
        },
        # Case 2: Space separated
        {
            "argv": ["run_ath_scan.py", "--tickers", "SPY", "AAPL"],
            "expected_count": 2
        },
        # Case 3: Comma separated
        {
            "argv": ["run_ath_scan.py", "--tickers", "SPY,VT,ORCL"],
            "expected_count": 3
        },
        # Case 4: Mixed
        {
            "argv": ["run_ath_scan.py", "--tickers", "SPY,VT", "AAPL", "MSFT,GOOGL"],
            "expected_count": 5
        }
    ]

    for case in test_cases:
        print(f"\nTesting with argv: {case['argv']}")
        with patch.object(sys, 'argv', case['argv']):
            # We need to mock ATHPipeline.run to avoid actual data fetching
            with patch('src.ml.pipeline.p14_ath.run_ath_scan.ATHPipeline') as MockPipeline:
                # Mock the return value of run()
                MockPipeline.return_value.run.return_value = pd.DataFrame()
                MockPipeline.return_value.results_dir = "dummy"
                
                main()
                
                # Check what config was passed to ATHPipeline
                config_passed = MockPipeline.call_args[0][0]
                actual_tickers = config_passed.tickers
                print(f"Actual tickers: {actual_tickers}")
                assert len(actual_tickers) == case['expected_count'], f"Expected {case['expected_count']}, got {len(actual_tickers)}"

if __name__ == "__main__":
    test_parsing()
    print("\nAll parsing tests passed!")
