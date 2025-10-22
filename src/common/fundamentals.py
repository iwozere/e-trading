"""
Fundamentals business logic for unified access and normalization across providers.

Usage:
    from src.common.fundamentals import get_fundamentals
    fundamentals = get_fundamentals('AAPL', provider='yf')
"""
from src.data.downloader.data_downloader_factory import DataDownloaderFactory
from src.model.telegram_bot import Fundamentals
# Removed circular import - fundamentals should not depend on indicators service
from src.indicators.models import IndicatorCalculationRequest, IndicatorCategory
from src.data.data_manager import get_data_manager

PROVIDER_CODES = ['yf', 'av', 'fh', 'td', 'pg', 'bnc', 'cg']


async def get_fundamentals_unified(ticker: str, provider: str = None, **kwargs) -> Fundamentals:
    """
    Get fundamentals using the unified indicator service.
    This is the new recommended approach.
    """
    # First, try DataManager path which handles caching and multi-provider merge
    dm = get_data_manager()
    combined = dm.get_fundamentals(symbol=ticker)

    fundamental_data = {
        'ticker': ticker,
        'company_name': None,
        'current_price': None,
        'market_cap': None,
        'pe_ratio': None,
        'forward_pe': None,
        'dividend_yield': None,
        'earnings_per_share': None,
        'price_to_book': None,
        'return_on_equity': None,
        'return_on_assets': None,
        'debt_to_equity': None,
        'current_ratio': None,
        'quick_ratio': None,
        'revenue': None,
        'revenue_growth': None,
        'net_income': None,
        'net_income_growth': None,
        'free_cash_flow': None,
        'operating_margin': None,
        'profit_margin': None,
        'beta': None,
        'sector': None,
        'industry': None,
        'country': None,
        'exchange': None,
        'currency': None,
        'shares_outstanding': None,
        'float_shares': None,
        'short_ratio': None,
        'payout_ratio': None,
        'peg_ratio': None,
        'price_to_sales': None,
        'enterprise_value': None,
        'enterprise_value_to_ebitda': None,
        'data_source': None,
        'last_updated': None,
        'sources': {}
    }

    # If DataManager returned combined fundamentals, map known fields
    if isinstance(combined, dict) and combined:
        mapping = {
            'ticker': 'ticker',
            'company_name': 'company_name',
            'current_price': 'current_price',
            'market_cap': 'market_cap',
            'pe_ratio': 'pe_ratio',
            'forward_pe': 'forward_pe',
            'dividend_yield': 'dividend_yield',
            'earnings_per_share': 'earnings_per_share',
            'price_to_book': 'price_to_book',
            'return_on_equity': 'return_on_equity',
            'return_on_assets': 'return_on_assets',
            'debt_to_equity': 'debt_to_equity',
            'current_ratio': 'current_ratio',
            'quick_ratio': 'quick_ratio',
            'revenue': 'revenue',
            'revenue_growth': 'revenue_growth',
            'net_income': 'net_income',
            'net_income_growth': 'net_income_growth',
            'free_cash_flow': 'free_cash_flow',
            'operating_margin': 'operating_margin',
            'profit_margin': 'profit_margin',
            'beta': 'beta',
            'sector': 'sector',
            'industry': 'industry',
            'country': 'country',
            'exchange': 'exchange',
            'currency': 'currency',
            'shares_outstanding': 'shares_outstanding',
            'float_shares': 'float_shares',
            'short_ratio': 'short_ratio',
            'payout_ratio': 'payout_ratio',
            'peg_ratio': 'peg_ratio',
            'price_to_sales': 'price_to_sales',
            'enterprise_value': 'enterprise_value',
            'enterprise_value_to_ebitda': 'enterprise_value_to_ebitda',
            'data_source': 'data_source',
            'last_updated': 'last_updated',
        }
        for src_key, dest_key in mapping.items():
            if src_key in combined and combined[src_key] is not None:
                fundamental_data[dest_key] = combined[src_key]
    else:
        # Fall back to indicator service if DataManager had no data
        indicator_service = get_unified_indicator_service()

        request = IndicatorCalculationRequest(
            ticker=ticker,
            indicators=["PE_RATIO", "PB_RATIO", "PS_RATIO", "PEG_RATIO", "ROE", "ROA", "DEBT_TO_EQUITY", "CURRENT_RATIO", "QUICK_RATIO", "OPERATING_MARGIN", "PROFIT_MARGIN", "REVENUE_GROWTH", "NET_INCOME_GROWTH", "FREE_CASH_FLOW", "DIVIDEND_YIELD", "PAYOUT_RATIO", "MARKET_CAP", "ENTERPRISE_VALUE"],
            timeframe="1d",
            period="1y",
            provider=provider
        )

        indicator_set = await indicator_service.get_indicators(request)

        indicator_mapping = {
            'PE_RATIO': 'pe_ratio',
            'FORWARD_PE': 'forward_pe',
            'PB_RATIO': 'price_to_book',
            'PS_RATIO': 'price_to_sales',
            'PEG_RATIO': 'peg_ratio',
            'ROE': 'return_on_equity',
            'ROA': 'return_on_assets',
            'DEBT_TO_EQUITY': 'debt_to_equity',
            'CURRENT_RATIO': 'current_ratio',
            'QUICK_RATIO': 'quick_ratio',
            'OPERATING_MARGIN': 'operating_margin',
            'PROFIT_MARGIN': 'profit_margin',
            'REVENUE_GROWTH': 'revenue_growth',
            'NET_INCOME_GROWTH': 'net_income_growth',
            'FREE_CASH_FLOW': 'free_cash_flow',
            'DIVIDEND_YIELD': 'dividend_yield',
            'PAYOUT_RATIO': 'payout_ratio',
            'MARKET_CAP': 'market_cap',
            'ENTERPRISE_VALUE': 'enterprise_value',
        }

        for name, indicator in indicator_set.fundamental_indicators.items():
            if name in indicator_mapping:
                field_name = indicator_mapping[name]
                fundamental_data[field_name] = indicator.value

    # Enrich with traditional fundamentals if still missing key metadata
    try:
        traditional_fundamentals = get_fundamentals(ticker, provider)
        if traditional_fundamentals:
            for key in ['company_name','current_price','sector','industry','country','exchange','currency','earnings_per_share','revenue','net_income','beta','shares_outstanding','float_shares','short_ratio','enterprise_value_to_ebitda']:
                val = getattr(traditional_fundamentals, key, None)
                if val is not None and fundamental_data.get(key) is None:
                    fundamental_data[key] = val
    except Exception:
        # Ignore failures; we already have best-effort data
        pass

    return Fundamentals(**fundamental_data)


def get_fundamentals(ticker: str, provider: str = None, **kwargs) -> Fundamentals:
    """
    Retrieve fundamentals for a ticker using the specified data provider.
    If provider is None, try all providers and merge results for the most complete Fundamentals.
    """
    # Allow all reasonable ticker lengths (supports tickers like GOOGL, BRK.B, etc.)

    if provider:
        prov = provider
        downloader = DataDownloaderFactory.create_downloader(prov, **kwargs)
        if downloader is None:
            raise ValueError(f"Unknown or unsupported provider: {prov}")
        result = downloader.get_fundamentals(ticker)
        # If result is already Fundamentals, return as is
        if isinstance(result, Fundamentals):
            return result
        # Otherwise, try to normalize
        return normalize_fundamentals({prov: result})
    else:
        provider_results = {}
        for code in PROVIDER_CODES:
            try:
                downloader = DataDownloaderFactory.create_downloader(code, **kwargs)
                if downloader:
                    result = downloader.get_fundamentals(ticker)
                    # Accept both Fundamentals and dicts
                    provider_results[code] = result.__dict__ if hasattr(result, '__dict__') else result
            except Exception:
                continue  # Skip providers that fail
        return normalize_fundamentals(provider_results)

def normalize_fundamentals(sources: dict) -> Fundamentals:
    """
    Combine and normalize fundamental data from different providers into a standard schema.
    Priority: yfinance (yf) > Alpha Vantage (av) > Finnhub (fh) > Twelve Data (td) > Polygon (pg)
    Returns a Fundamentals object with the most complete data.

    Args:
        sources: Dictionary with provider codes as keys and data objects as values
                e.g., {'yf': yf_data, 'av': av_data, 'fh': fh_data}

    Returns:
        Fundamentals object with merged data
    """
    field_sources = {}

    def get_first_with_source(field_name, *provider_fields):
        """
        Get the first non-None value from providers, recording the source.

        Args:
            field_name: Name of the field being populated
            *provider_fields: Tuples of (provider_code, field_key) to check
        """
        for provider_code, field_key in provider_fields:
            if provider_code in sources:
                data = sources[provider_code]
                if isinstance(data, dict) and field_key in data:
                    value = data[field_key]
                    if value is not None and value != '':
                        field_sources[field_name] = provider_code
                        return value
        return None

    fundamentals = Fundamentals(
        ticker=get_first_with_source(
            "ticker",
            ("yf", "ticker"),
            ("av", "Symbol"),
            ("fh", "ticker"),
            ("td", "ticker"),
            ("pg", "ticker"),
        ),
        company_name=get_first_with_source(
            "company_name",
            ("yf", "company_name"),
            ("av", "Name"),
            ("pg", "name"),
        ),
        current_price=get_first_with_source(
            "current_price",
            ("yf", "current_price"),
            ("av", "current_price"),
            ("fh", "current_price"),
            ("td", "current_price"),
            ("pg", "current_price"),
        ),
        market_cap=get_first_with_source(
            "market_cap",
            ("yf", "market_cap"),
            ("av", "MarketCapitalization"),
            ("fh", "market_cap"),
            ("td", "market_cap"),
            ("pg", "market_cap"),
        ),
        pe_ratio=get_first_with_source(
            "pe_ratio",
            ("yf", "pe_ratio"),
            ("yf", "trailingPE"),
            ("av", "PERatio"),
            ("fh", "pe_ratio"),
            ("td", "pe_ratio"),
            ("pg", "pe_ratio"),
        ),
        forward_pe=get_first_with_source(
            "forward_pe",
            ("yf", "forward_pe"),
            ("av", "ForwardPE"),
        ),
        dividend_yield=get_first_with_source(
            "dividend_yield",
            ("yf", "dividend_yield"),
            ("av", "DividendYield"),
        ),
        earnings_per_share=get_first_with_source(
            "earnings_per_share",
            ("yf", "earnings_per_share"),
            ("yf", "trailingEps"),
            ("av", "EPS"),
            ("fh", "earnings_per_share"),
            ("td", "earnings_per_share"),
        ),
        price_to_book=get_first_with_source(
            "price_to_book",
            ("yf", "price_to_book"),
            ("yf", "priceToBook"),
            ("av", "PriceToBookRatio"),
        ),
        return_on_equity=get_first_with_source(
            "return_on_equity",
            ("yf", "return_on_equity"),
            ("av", "ReturnOnEquityTTM"),
        ),
        return_on_assets=get_first_with_source(
            "return_on_assets",
            ("yf", "return_on_assets"),
            ("av", "ReturnOnAssetsTTM"),
        ),
        debt_to_equity=get_first_with_source(
            "debt_to_equity",
            ("yf", "debt_to_equity"),
            ("av", "DebtToEquityRatio"),
        ),
        current_ratio=get_first_with_source(
            "current_ratio",
            ("yf", "current_ratio"),
            ("av", "CurrentRatio"),
        ),
        quick_ratio=get_first_with_source(
            "quick_ratio",
            ("yf", "quick_ratio"),
            ("av", "QuickRatio"),
        ),
        revenue=get_first_with_source(
            "revenue",
            ("yf", "revenue"),
            ("av", "RevenueTTM"),
            ("fh", "revenue"),
            ("td", "revenue"),
        ),
        revenue_growth=get_first_with_source(
            "revenue_growth",
            ("yf", "revenue_growth"),
            ("av", "RevenueGrowthTTM"),
        ),
        net_income=get_first_with_source(
            "net_income",
            ("yf", "net_income"),
            ("av", "NetIncomeTTM"),
        ),
        net_income_growth=get_first_with_source(
            "net_income_growth",
            ("yf", "net_income_growth"),
            ("av", "NetIncomeGrowthTTM"),
        ),
        free_cash_flow=get_first_with_source(
            "free_cash_flow",
            ("yf", "free_cash_flow"),
            ("av", "FreeCashFlowTTM"),
        ),
        operating_margin=get_first_with_source(
            "operating_margin",
            ("yf", "operating_margin"),
            ("av", "OperatingMarginTTM"),
        ),
        profit_margin=get_first_with_source(
            "profit_margin",
            ("yf", "profit_margin"),
            ("av", "ProfitMarginTTM"),
        ),
        beta=get_first_with_source(
            "beta",
            ("yf", "beta"),
            ("av", "Beta"),
        ),
        sector=get_first_with_source(
            "sector",
            ("yf", "sector"),
            ("av", "Sector"),
        ),
        industry=get_first_with_source(
            "industry",
            ("yf", "industry"),
            ("av", "Industry"),
        ),
        country=get_first_with_source(
            "country",
            ("yf", "country"),
            ("av", "Country"),
        ),
        exchange=get_first_with_source(
            "exchange",
            ("yf", "exchange"),
            ("av", "Exchange"),
        ),
        currency=get_first_with_source(
            "currency",
            ("yf", "currency"),
            ("av", "Currency"),
        ),
        shares_outstanding=get_first_with_source(
            "shares_outstanding",
            ("yf", "shares_outstanding"),
            ("av", "SharesOutstanding"),
        ),
        float_shares=get_first_with_source(
            "float_shares",
            ("yf", "float_shares"),
            ("av", "FloatShares"),
        ),
        short_ratio=get_first_with_source(
            "short_ratio",
            ("yf", "short_ratio"),
            ("av", "ShortRatio"),
        ),
        payout_ratio=get_first_with_source(
            "payout_ratio",
            ("yf", "payout_ratio"),
            ("av", "PayoutRatio"),
        ),
        peg_ratio=get_first_with_source(
            "peg_ratio",
            ("yf", "peg_ratio"),
            ("av", "PEGRatio"),
        ),
        price_to_sales=get_first_with_source(
            "price_to_sales",
            ("yf", "price_to_sales"),
            ("av", "PriceToSalesRatioTTM"),
        ),
        enterprise_value=get_first_with_source(
            "enterprise_value",
            ("yf", "enterprise_value"),
            ("av", "MarketCapitalization"),
        ),
        enterprise_value_to_ebitda=get_first_with_source(
            "enterprise_value_to_ebitda",
            ("yf", "enterprise_value_to_ebitda"),
            ("av", "EVToEBITDA"),
        ),
        data_source=get_first_with_source(
            "data_source",
            ("yf", "data_source"),
            ("av", "data_source"),
            ("fh", "data_source"),
            ("td", "data_source"),
            ("pg", "data_source"),
        ),
        last_updated=get_first_with_source(
            "last_updated",
            ("yf", "last_updated"),
            ("av", "last_updated"),
            ("fh", "last_updated"),
            ("td", "last_updated"),
            ("pg", "last_updated"),
        ),
        sources=field_sources
    )
    return fundamentals

def format_fundamental_analysis(fundamentals) -> str:
    """
    Format the fundamentals section for display in Telegram/email.
    Handles missing values gracefully. Includes all available fields.
    """
    if not fundamentals:
        return ""
    lines = []
    # Price & Market
    if fundamentals.current_price is not None:
        lines.append(f"💵 Price: ${fundamentals.current_price:.2f}")
    if fundamentals.market_cap is not None:
        lines.append(f"💸 Market Cap: ${(fundamentals.market_cap/1e9):.2f}B")
    if fundamentals.pe_ratio is not None or fundamentals.forward_pe is not None:
        pe = f"{fundamentals.pe_ratio:.2f}" if fundamentals.pe_ratio is not None else "-"
        fpe = f"{fundamentals.forward_pe:.2f}" if fundamentals.forward_pe is not None else "-"
        lines.append(f"🏦 P/E: {pe}, Forward P/E: {fpe}")
    if fundamentals.earnings_per_share is not None or fundamentals.dividend_yield is not None:
        eps = f"${fundamentals.earnings_per_share:.2f}" if fundamentals.earnings_per_share is not None else "-"
        dy = f"{fundamentals.dividend_yield*100:.2f}%" if fundamentals.dividend_yield is not None else "-"
        lines.append(f"📊 EPS: {eps}, Div Yield: {dy}")
    # Profitability
    if fundamentals.operating_margin is not None:
        lines.append(f"🧮 Operating Margin: {fundamentals.operating_margin:.2f}")
    if fundamentals.profit_margin is not None:
        lines.append(f"💰 Profit Margin: {fundamentals.profit_margin:.2f}")
    if fundamentals.return_on_equity is not None:
        lines.append(f"🔄 ROE: {fundamentals.return_on_equity:.2f}")
    if fundamentals.return_on_assets is not None:
        lines.append(f"🔄 ROA: {fundamentals.return_on_assets:.2f}")
    # Balance Sheet
    if fundamentals.price_to_book is not None:
        lines.append(f"📚 Price/Book: {fundamentals.price_to_book:.2f}")
    if fundamentals.debt_to_equity is not None:
        lines.append(f"🏦 Debt/Equity: {fundamentals.debt_to_equity:.2f}")
    if fundamentals.current_ratio is not None:
        lines.append(f"💧 Current Ratio: {fundamentals.current_ratio:.2f}")
    if fundamentals.quick_ratio is not None:
        lines.append(f"⚡ Quick Ratio: {fundamentals.quick_ratio:.2f}")
    # Growth
    if fundamentals.revenue is not None:
        lines.append(f"📈 Revenue: ${fundamentals.revenue/1e6:.2f}M")
    if fundamentals.revenue_growth is not None:
        lines.append(f"📈 Revenue Growth: {fundamentals.revenue_growth:.2f}")
    if fundamentals.net_income is not None:
        lines.append(f"💵 Net Income: ${fundamentals.net_income/1e6:.2f}M")
    if fundamentals.net_income_growth is not None:
        lines.append(f"💵 Net Income Growth: {fundamentals.net_income_growth:.2f}")
    if fundamentals.free_cash_flow is not None:
        lines.append(f"💸 Free Cash Flow: ${fundamentals.free_cash_flow/1e6:.2f}M")
    # Other ratios
    if fundamentals.beta is not None:
        lines.append(f"📉 Beta: {fundamentals.beta:.2f}")
    if fundamentals.shares_outstanding is not None:
        lines.append(f"🧾 Shares Outstanding: {fundamentals.shares_outstanding/1e6:.2f}M")
    if fundamentals.float_shares is not None:
        lines.append(f"🧾 Float Shares: {fundamentals.float_shares/1e6:.2f}M")
    if fundamentals.short_ratio is not None:
        lines.append(f"📉 Short Ratio: {fundamentals.short_ratio:.2f}")
    if fundamentals.payout_ratio is not None:
        lines.append(f"💵 Payout Ratio: {fundamentals.payout_ratio:.2f}")
    if fundamentals.peg_ratio is not None:
        lines.append(f"📈 PEG Ratio: {fundamentals.peg_ratio:.2f}")
    if fundamentals.price_to_sales is not None:
        lines.append(f"💲 Price/Sales: {fundamentals.price_to_sales:.2f}")
    if fundamentals.enterprise_value is not None:
        lines.append(f"🏢 Enterprise Value: ${fundamentals.enterprise_value/1e9:.2f}B")
    if fundamentals.enterprise_value_to_ebitda is not None:
        lines.append(f"🏢 EV/EBITDA: {fundamentals.enterprise_value_to_ebitda:.2f}")
    # Company Info
    if fundamentals.sector:
        lines.append(f"🏭 Sector: {fundamentals.sector}")
    if fundamentals.industry:
        lines.append(f"🏢 Industry: {fundamentals.industry}")
    if fundamentals.country:
        lines.append(f"🌎 Country: {fundamentals.country}")
    if fundamentals.exchange:
        lines.append(f"💹 Exchange: {fundamentals.exchange}")
    if fundamentals.currency:
        lines.append(f"💱 Currency: {fundamentals.currency}")
    # Data source and update info
    if fundamentals.data_source:
        lines.append(f"🔗 Data Source: {fundamentals.data_source}")
    if fundamentals.last_updated:
        lines.append(f"🕒 Last Updated: {fundamentals.last_updated}")
    return '\n'.join(lines) + '\n'
