#!/usr/bin/env python3
"""
Script to extract ticker lists from tickers_list.py into CSV files
"""

import os
import re
import pandas as pd

def extract_ticker_list_from_function(file_path, function_name):
    """Extract ticker list from a specific function in the file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the function
    pattern = rf'def {function_name}\(\):\s*return\s*\[(.*?)\]'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        # Try alternative pattern for functions that assign to variable first
        pattern = rf'def {function_name}\(\):\s*arr\s*=\s*\[(.*?)\]\s*return\s*arr'
        match = re.search(pattern, content, re.DOTALL)

    if match:
        ticker_list_str = match.group(1)
        # Extract ticker symbols
        tickers = re.findall(r'"([^"]+)"', ticker_list_str)
        return tickers

    return []

def main():
    # Create data/tickers directory if it doesn't exist
    os.makedirs('data/tickers', exist_ok=True)

    # Define the functions to extract
    functions = [
        'get_all_us_tickers',
        'get_us_delisted_tickers', 
        'get_us_small_cap_tickers',
        'get_us_medium_cap_tickers',
        'get_us_large_cap_tickers'
    ]

    for func_name in functions:
        print(f"Extracting {func_name}...")
        tickers = extract_ticker_list_from_function('src/screener/tickers_list.py', func_name)

        if tickers:
            # Create DataFrame
            df = pd.DataFrame({'ticker': tickers})

            # Save to CSV
            csv_filename = f"data/tickers/{func_name.replace('get_', '').replace('_', '_')}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"  Saved {len(tickers)} tickers to {csv_filename}")
        else:
            print(f"  No tickers found for {func_name}")

if __name__ == "__main__":
    main() 
