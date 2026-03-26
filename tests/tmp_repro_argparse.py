import argparse

def test_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default="SPY,VT,AAPL")
    
    # Case 1: No arguments (uses default)
    args1 = parser.parse_args([])
    print(f"No args: {args1.tickers} (type: {type(args1.tickers)})")
    
    # Case 2: One ticker
    args2 = parser.parse_args(["--tickers", "SPY"])
    print(f"One ticker: {args2.tickers} (type: {type(args2.tickers)})")

    # Case 3: Comma separated (user's likely intent)
    args3 = parser.parse_args(["--tickers", "SPY,VT"])
    print(f"Comma sep: {args3.tickers} (type: {type(args3.tickers)})")

if __name__ == "__main__":
    test_argparse()
