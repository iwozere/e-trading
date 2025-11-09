import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def analyze_drawdowns(ticker, years=5, threshold=0.10):
    """
    Analyzes stock price drops greater than a specified percentage from peaks

    Args:
        ticker (str): Stock ticker (e.g., 'AAPL')
        years (int): Number of years to analyze
        threshold (float): Drop threshold value (0.10 = 10%)

    Returns:
        dict: Analysis results
    """

    # Get data for the specified period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    try:
        # Load data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data available for ticker {ticker}")

        # Work with closing prices
        prices = data['Close'].copy()
        low_prices = data['Low'].copy()  # Add low prices for more accurate min tracking
        high_prices = data['High'].copy()  # Add high prices for more accurate max tracking

        # Calculate rolling maxima (peaks) using high prices
        rolling_max = high_prices.expanding().max()

        # Calculate drawdowns as percentages using low prices vs high price peaks for more accuracy
        drawdowns = (low_prices - rolling_max) / rolling_max

        # Count separate drop events
        drawdown_events = []
        in_drawdown = False
        current_event = None

        for date, dd in drawdowns.items():
            if dd <= -threshold and not in_drawdown:
                # Start of new drawdown event
                in_drawdown = True
                current_event = {
                    'start_date': date,
                    'start_price': high_prices[date],  # Use high price as start price
                    'peak_price': rolling_max[date],
                    'max_drawdown': dd,
                    'max_drawdown_date': date,
                    'min_price': low_prices[date]  # Initialize with low price of start day
                }
            elif dd <= -threshold and in_drawdown:
                # Continuation of drawdown event
                if dd < current_event['max_drawdown']:
                    current_event['max_drawdown'] = dd
                    current_event['max_drawdown_date'] = date
                    current_event['min_price'] = low_prices[date]
                elif low_prices[date] < current_event['min_price']:
                    # Update min_price if current low price is lower
                    current_event['min_price'] = low_prices[date]
            elif dd > -threshold and in_drawdown:
                # End of drawdown event
                in_drawdown = False
                current_event['end_date'] = date
                current_event['recovery_price'] = prices[date]
                current_event['duration_days'] = (date - current_event['start_date']).days
                drawdown_events.append(current_event)

        # If last event hasn't ended
        if in_drawdown and current_event:
            current_event['end_date'] = prices.index[-1]
            current_event['recovery_price'] = prices.iloc[-1]
            current_event['duration_days'] = (current_event['end_date'] - current_event['start_date']).days
            drawdown_events.append(current_event)

        # Statistics
        total_events = len(drawdown_events)
        max_drawdown = drawdowns.min()
        max_drawdown_date = drawdowns.idxmin()

        results = {
            'ticker': ticker,
            'period_years': years,
            'threshold_percent': threshold * 100,
            'total_drawdown_events': total_events,
            'max_drawdown_percent': max_drawdown * 100,
            'max_drawdown_date': max_drawdown_date,
            'drawdown_events': drawdown_events,
            'prices': prices,
            'drawdowns': drawdowns,
            'rolling_max': rolling_max
        }

        return results

    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None

def print_results(results):
    """Prints analysis results"""
    if not results:
        return

    print(f"\n{'='*60}")
    print(f"DRAWDOWN ANALYSIS FOR {results['ticker']}")
    print(f"{'='*60}")
    print(f"Analysis period: {results['period_years']} years")
    print(f"Threshold value: {results['threshold_percent']:.1f}%")
    print(f"Total drawdown events: {results['total_drawdown_events']}")
    print(f"Maximum drawdown: {results['max_drawdown_percent']:.2f}%")
    print(f"Maximum drawdown date: {results['max_drawdown_date'].strftime('%Y-%m-%d')}")

    if results['drawdown_events']:
        print("\nDRAWDOWN EVENT DETAILS:")
        print(f"{'#':<3} {'Start Date':<12} {'End Date':<12} {'Start Price':<12} {'Min Price':<12} {'Peak Price':<12} {'Price Delta':<12} {'Max Drawdown':<15} {'Days':<8}")
        print("-" * 115)

        for i, event in enumerate(results['drawdown_events'], 1):
            start_date = event['start_date'].strftime('%Y-%m-%d')
            end_date = event.get('end_date', 'Ongoing').strftime('%Y-%m-%d') if hasattr(event.get('end_date'), 'strftime') else 'Ongoing'
            start_price = f"${event['start_price']:.2f}"
            min_price = f"${event['min_price']:.2f}"
            peak_price = f"${event['peak_price']:.2f}"
            price_delta = f"${event['min_price'] - event['start_price']:.2f}"  # Min price - Start price (high)

            # Use the original peak-to-trough drawdown calculation
            max_dd = f"{event['max_drawdown']*100:.2f}%"
            duration = event.get('duration_days', 'N/A')

            print(f"{i:<3} {start_date:<12} {end_date:<12} {start_price:<12} {min_price:<12} {peak_price:<12} {price_delta:<12} {max_dd:<15} {duration:<8}")

def plot_drawdowns(results):
    """Creates price and drawdown charts"""
    if not results:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Price and peaks chart
    ax1.plot(results['prices'].index, results['prices'], label='Price', linewidth=1)
    ax1.plot(results['rolling_max'].index, results['rolling_max'],
             label='Rolling Maximum', linewidth=1, alpha=0.7)
    ax1.set_title(f'{results["ticker"]} Stock Price Over {results["period_years"]} Years')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdowns chart
    ax2.fill_between(results['drawdowns'].index,
                     results['drawdowns'] * 100, 0,
                     alpha=0.3, color='red')
    ax2.axhline(y=-results['threshold_percent'], color='red', linestyle='--',
                label=f'Threshold -{results["threshold_percent"]}%')
    ax2.set_title('Drawdowns Relative to Peaks')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def analyze_intraday_swings(ticker, years=5, swing_threshold=0.05):
    """
    Analyzes intraday price swings (high to low) that exceed a specified percentage

    Args:
        ticker (str): Stock ticker (e.g., 'AAPL')
        years (int): Number of years to analyze
        swing_threshold (float): Swing threshold value (0.05 = 5%)

    Returns:
        dict: Analysis results
    """

    # Get data for the specified period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    try:
        # Load data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data available for ticker {ticker}")

        # Work with high and low prices
        high_prices = data['High'].copy()
        low_prices = data['Low'].copy()

        # Calculate intraday swings as percentages
        swings = (low_prices - high_prices) / high_prices

        # Find all swings greater than threshold (negative because low < high)
        significant_swings = swings[swings <= -swing_threshold]

        # Create swing events
        swing_events = []

        for date, swing in significant_swings.items():
            swing_event = {
                'date': date,
                'high_price': high_prices[date],
                'low_price': low_prices[date],
                'swing_percent': swing * 100,
                'swing_amount': low_prices[date] - high_prices[date]
            }
            swing_events.append(swing_event)

        # Sort by swing magnitude (most negative first)
        swing_events.sort(key=lambda x: x['swing_percent'])

        results = {
            'ticker': ticker,
            'period_years': years,
            'swing_threshold_percent': swing_threshold * 100,
            'total_swing_events': len(swing_events),
            'swing_events': swing_events,
            'high_prices': high_prices,
            'low_prices': low_prices,
            'swings': swings
        }

        return results

    except Exception as e:
        print(f"Error analyzing intraday swings for {ticker}: {str(e)}")
        return None

def print_swing_results(results):
    """Prints intraday swing analysis results"""
    if not results:
        return

    print(f"\n{'='*60}")
    print(f"INTRADAY SWING ANALYSIS FOR {results['ticker']}")
    print(f"{'='*60}")
    print(f"Analysis period: {results['period_years']} years")
    print(f"Swing threshold: {results['swing_threshold_percent']:.1f}%")
    print(f"Total swing events: {results['total_swing_events']}")

    if results['swing_events']:
        print("\nINTRADAY SWING DETAILS:")
        print(f"{'#':<3} {'Date':<12} {'High Price':<12} {'Low Price':<12} {'Swing %':<10} {'Swing $':<10}")
        print("-" * 70)

        for i, event in enumerate(results['swing_events'], 1):
            date = event['date'].strftime('%Y-%m-%d')
            high_price = f"${event['high_price']:.2f}"
            low_price = f"${event['low_price']:.2f}"
            swing_percent = f"{event['swing_percent']:.2f}%"
            swing_amount = f"${event['swing_amount']:.2f}"

            print(f"{i:<3} {date:<12} {high_price:<12} {low_price:<12} {swing_percent:<10} {swing_amount:<10}")

# Main program
def main():
    """Main program function"""

    # Usage examples
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']  # Can add any tickers

    print("STOCK DRAWDOWN ANALYSIS")
    print("Analyzing price drops greater than 10% from previous peaks")

    # User input
    user_ticker = input(f"\nEnter ticker for analysis (or Enter for example with {tickers[0]}): ").strip().upper()

    if not user_ticker:
        user_ticker = tickers[0]

    try:
        years = int(input("Number of years to analyze (default 5): ") or "5")
        threshold = float(input("Drop threshold in % (default 5): ") or "5") / 100
        swing_threshold = float(input("Intraday swing threshold in % (default 5): ") or "5") / 100
    except ValueError:
        years = 5
        threshold = 0.10
        swing_threshold = 0.05

    # Analyze selected ticker
    results = analyze_drawdowns(user_ticker, years, threshold)
    swing_results = analyze_intraday_swings(user_ticker, years, swing_threshold)

    if results:
        print_results(results)

    if swing_results:
        print_swing_results(swing_results)

    if results:
        # Offer to show chart
        show_plot = input("\nShow chart? (y/n): ").lower().strip()
        if show_plot in ['y', 'yes']:
            plot_drawdowns(results)

        # Offer to analyze multiple tickers
        analyze_multiple = input("\nAnalyze multiple popular tickers? (y/n): ").lower().strip()
        if analyze_multiple in ['y', 'yes']:
            print(f"\nMultiple ticker analysis with {threshold*100}% threshold:")
            summary = []

            for ticker in tickers:
                result = analyze_drawdowns(ticker, years, threshold)
                if result:
                    summary.append({
                        'ticker': ticker,
                        'events': result['total_drawdown_events'],
                        'max_drawdown': result['max_drawdown_percent']
                    })

            # Print summary table
            print(f"\n{'Ticker':<8} {'Events':<10} {'Max Drawdown':<15}")
            print("-" * 35)
            for item in summary:
                print(f"{item['ticker']:<8} {item['events']:<10} {item['max_drawdown']:.2f}%")

if __name__ == "__main__":
    main()
