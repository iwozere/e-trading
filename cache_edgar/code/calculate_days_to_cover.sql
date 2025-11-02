-- SQL script to calculate and update days_to_cover in ss_finra_short_interest table
-- Formula: days_to_cover = short_interest_shares / average_daily_volume
-- Uses avg_volume_14d from ss_snapshot table when available

-- Start transaction
BEGIN;

-- Update days_to_cover using volume data from ss_snapshot table
-- This joins FINRA data with the most recent snapshot data for each ticker
UPDATE ss_finra_short_interest 
SET 
    days_to_cover = CASE 
        WHEN s.avg_volume_14d > 0 THEN 
            ROUND(ss_finra_short_interest.short_interest_shares::numeric / s.avg_volume_14d::numeric, 2)
        ELSE NULL 
    END,
    updated_at = CURRENT_TIMESTAMP
FROM (
    -- Get the most recent snapshot for each ticker with volume data
    SELECT DISTINCT ON (ticker) 
        ticker, 
        avg_volume_14d,
        run_date
    FROM ss_snapshot 
    WHERE avg_volume_14d IS NOT NULL 
      AND avg_volume_14d > 0
    ORDER BY ticker, run_date DESC
) s
WHERE ss_finra_short_interest.ticker = s.ticker
  AND ss_finra_short_interest.short_interest_shares IS NOT NULL 
  AND ss_finra_short_interest.short_interest_shares > 0
  AND ss_finra_short_interest.days_to_cover IS NULL  -- Only update if not already calculated
  AND ss_finra_short_interest.settlement_date >= s.run_date - INTERVAL '30 days';  -- Only use recent volume data

-- Alternative approach: Update using a fixed average volume estimate
-- This is a fallback for tickers without snapshot data
-- Note: This uses a conservative estimate and should be replaced with actual volume data

-- Get statistics on the update
SELECT 
    'Days to Cover Update Statistics' as operation,
    COUNT(*) as total_records_updated,
    MIN(days_to_cover) as min_days,
    MAX(days_to_cover) as max_days,
    AVG(days_to_cover) as avg_days,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_to_cover) as median_days
FROM ss_finra_short_interest 
WHERE days_to_cover IS NOT NULL;

-- Show sample of updated records
SELECT 
    ticker,
    settlement_date,
    short_interest_shares,
    days_to_cover,
    CASE 
        WHEN days_to_cover > 10 THEN 'HIGH (>10 days)'
        WHEN days_to_cover > 5 THEN 'MEDIUM (5-10 days)'
        WHEN days_to_cover > 2 THEN 'LOW (2-5 days)'
        ELSE 'VERY LOW (<2 days)'
    END as days_to_cover_level
FROM ss_finra_short_interest 
WHERE days_to_cover IS NOT NULL 
ORDER BY days_to_cover DESC 
LIMIT 20;

-- Commit transaction
COMMIT;

-- Summary report
SELECT 
    'Summary Report' as report_type,
    COUNT(*) as total_records,
    COUNT(days_to_cover) as records_with_days_to_cover,
    COUNT(*) - COUNT(days_to_cover) as records_without_days_to_cover,
    ROUND(COUNT(days_to_cover) * 100.0 / COUNT(*), 2) as coverage_percentage
FROM ss_finra_short_interest;

-- Records that still need volume data
SELECT 
    'Records Missing Volume Data' as issue_type,
    COUNT(*) as count
FROM ss_finra_short_interest f
LEFT JOIN (
    SELECT DISTINCT ticker 
    FROM ss_snapshot 
    WHERE avg_volume_14d IS NOT NULL 
      AND avg_volume_14d > 0
) s ON f.ticker = s.ticker
WHERE f.short_interest_shares IS NOT NULL 
  AND f.short_interest_shares > 0
  AND s.ticker IS NULL;

-- Script completed successfully