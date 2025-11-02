-- SQL script to calculate and update short_interest_pct in ss_finra_short_interest table
-- Formula: short_interest_pct = (short_interest_shares / float_shares) * 100
-- Only updates records where both short_interest_shares and float_shares are available

-- Start transaction
BEGIN;

-- Update short_interest_pct where we have both short_interest_shares and float_shares
-- Note: Values are capped at 100% due to database constraint, but this may not reflect true market conditions
UPDATE ss_finra_short_interest 
SET 
    short_interest_pct = CASE 
        WHEN float_shares > 0 THEN 
            -- Cap at 100% due to database constraint (short interest can actually exceed 100% in real markets)
            LEAST(100.0, ROUND((short_interest_shares::numeric / float_shares::numeric) * 100, 4))
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
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY short_interest_pct) as median_pct
FROM ss_finra_short_interest 
WHERE short_interest_pct IS NOT NULL;

-- Show sample of updated records
SELECT 
    ticker,
    settlement_date,
    short_interest_shares,
    float_shares,
    short_interest_pct,
    CASE 
        WHEN short_interest_pct > 20 THEN 'HIGH (>20%)'
        WHEN short_interest_pct > 10 THEN 'MEDIUM (10-20%)'
        WHEN short_interest_pct > 5 THEN 'LOW (5-10%)'
        ELSE 'VERY LOW (<5%)'
    END as short_interest_level
FROM ss_finra_short_interest 
WHERE short_interest_pct IS NOT NULL 
ORDER BY short_interest_pct DESC 
LIMIT 20;

-- Commit transaction
COMMIT;

-- Summary report
SELECT 
    'Summary Report' as report_type,
    COUNT(*) as total_records,
    COUNT(short_interest_pct) as records_with_pct,
    COUNT(*) - COUNT(short_interest_pct) as records_without_pct,
    ROUND(COUNT(short_interest_pct) * 100.0 / COUNT(*), 2) as coverage_percentage
FROM ss_finra_short_interest;

-- Records that still need float_shares data
SELECT 
    'Records Missing Float Shares' as issue_type,
    COUNT(*) as count
FROM ss_finra_short_interest 
WHERE short_interest_shares IS NOT NULL 
  AND (float_shares IS NULL OR float_shares = 0);

-- Script completed successfully