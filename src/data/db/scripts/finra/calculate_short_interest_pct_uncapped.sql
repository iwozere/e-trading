-- SQL script to calculate and update short_interest_pct in ss_finra_short_interest table
-- Formula: short_interest_pct = (short_interest_shares / float_shares) * 100
-- This version allows values > 100% (requires running fix_short_interest_pct_constraint.sql first)

-- Start transaction
BEGIN;

-- Update short_interest_pct where we have both short_interest_shares and float_shares
-- No capping - allows true market values > 100%
UPDATE ss_finra_short_interest 
SET 
    short_interest_pct = CASE 
        WHEN float_shares > 0 THEN 
            ROUND((short_interest_shares::numeric / float_shares::numeric) * 100, 4)
        ELSE NULL 
    END,
    updated_at = CURRENT_TIMESTAMP
WHERE 
    short_interest_shares IS NOT NULL 
    AND float_shares IS NOT NULL 
    AND float_shares > 0
    AND short_interest_pct IS NULL;  -- Only update if not already calculated

-- Get statistics on the update
SELECT 
    'Short Interest Percentage Update Statistics' as operation,
    COUNT(*) as total_records_updated,
    MIN(short_interest_pct) as min_pct,
    MAX(short_interest_pct) as max_pct,
    AVG(short_interest_pct) as avg_pct,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY short_interest_pct) as median_pct,
    COUNT(*) FILTER (WHERE short_interest_pct > 100) as records_over_100_pct
FROM ss_finra_short_interest 
WHERE short_interest_pct IS NOT NULL;

-- Show sample of updated records, including high short interest cases
SELECT 
    ticker,
    settlement_date,
    short_interest_shares,
    float_shares,
    short_interest_pct,
    CASE 
        WHEN short_interest_pct > 100 THEN 'EXTREME (>100%)'
        WHEN short_interest_pct > 50 THEN 'VERY HIGH (50-100%)'
        WHEN short_interest_pct > 20 THEN 'HIGH (20-50%)'
        WHEN short_interest_pct > 10 THEN 'MEDIUM (10-20%)'
        WHEN short_interest_pct > 5 THEN 'LOW (5-10%)'
        ELSE 'VERY LOW (<5%)'
    END as short_interest_level
FROM ss_finra_short_interest 
WHERE short_interest_pct IS NOT NULL 
ORDER BY short_interest_pct DESC 
LIMIT 20;

-- Show extreme cases (>100%) for review
SELECT 
    'Extreme Short Interest Cases (>100%)' as case_type,
    ticker,
    settlement_date,
    short_interest_shares,
    float_shares,
    short_interest_pct,
    ROUND(short_interest_shares::numeric / 1000000, 2) as short_interest_millions,
    ROUND(float_shares::numeric / 1000000, 2) as float_millions
FROM ss_finra_short_interest 
WHERE short_interest_pct > 100
ORDER BY short_interest_pct DESC
LIMIT 10;

-- Commit transaction
COMMIT;

-- Summary report
SELECT 
    'Summary Report' as report_type,
    COUNT(*) as total_records,
    COUNT(short_interest_pct) as records_with_pct,
    COUNT(*) - COUNT(short_interest_pct) as records_without_pct,
    ROUND(COUNT(short_interest_pct) * 100.0 / COUNT(*), 2) as coverage_percentage,
    COUNT(*) FILTER (WHERE short_interest_pct > 100) as extreme_cases_over_100_pct
FROM ss_finra_short_interest;

-- Records that still need float_shares data
SELECT 
    'Records Missing Float Shares' as issue_type,
    COUNT(*) as count
FROM ss_finra_short_interest 
WHERE short_interest_shares IS NOT NULL 
  AND (float_shares IS NULL OR float_shares = 0);

-- Script completed successfully