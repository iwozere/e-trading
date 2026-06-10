-- Verification queries for edgar_floats.sql update
-- Run these queries after applying the edgar_floats.sql script

-- 1. Check how many records were updated (should have non-null float_shares)
SELECT 
    COUNT(*) as total_records,
    COUNT(float_shares) as records_with_float_shares,
    COUNT(*) - COUNT(float_shares) as records_without_float_shares,
    ROUND(COUNT(float_shares) * 100.0 / COUNT(*), 2) as coverage_percentage
FROM ss_finra_short_interest;

-- 2. Sample of updated records
SELECT 
    ticker, 
    settlement_date, 
    float_shares, 
    short_interest_shares,
    CASE 
        WHEN float_shares > 0 THEN ROUND(short_interest_shares * 100.0 / float_shares, 4)
        ELSE NULL 
    END as short_interest_pct_calculated
FROM ss_finra_short_interest 
WHERE float_shares IS NOT NULL 
ORDER BY settlement_date DESC, ticker 
LIMIT 10;

-- 3. Check for any tickers that got updated
SELECT 
    ticker,
    COUNT(*) as total_records,
    COUNT(float_shares) as records_with_float_shares,
    MIN(settlement_date) as earliest_date,
    MAX(settlement_date) as latest_date
FROM ss_finra_short_interest 
WHERE float_shares IS NOT NULL
GROUP BY ticker 
ORDER BY records_with_float_shares DESC 
LIMIT 20;

-- 4. Identify any potential data quality issues
SELECT 
    ticker,
    settlement_date,
    float_shares,
    short_interest_shares
FROM ss_finra_short_interest 
WHERE float_shares IS NOT NULL 
  AND short_interest_shares > float_shares  -- Short interest can't exceed float
ORDER BY ticker, settlement_date;

-- 5. Summary statistics
SELECT 
    'Float Shares Statistics' as metric,
    MIN(float_shares) as min_value,
    MAX(float_shares) as max_value,
    AVG(float_shares) as avg_value,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY float_shares) as median_value
FROM ss_finra_short_interest 
WHERE float_shares IS NOT NULL;