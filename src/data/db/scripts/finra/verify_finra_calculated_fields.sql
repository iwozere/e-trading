-- Verification queries for FINRA calculated fields
-- Run these queries after populating short_interest_pct and days_to_cover

-- 1. Overall coverage statistics
SELECT 
    'Field Coverage Statistics' as report_type,
    COUNT(*) as total_records,
    COUNT(short_interest_pct) as records_with_pct,
    COUNT(days_to_cover) as records_with_days_to_cover,
    COUNT(float_shares) as records_with_float_shares,
    ROUND(COUNT(short_interest_pct) * 100.0 / COUNT(*), 2) as pct_coverage,
    ROUND(COUNT(days_to_cover) * 100.0 / COUNT(*), 2) as days_coverage,
    ROUND(COUNT(float_shares) * 100.0 / COUNT(*), 2) as float_coverage
FROM ss_finra_short_interest;

-- 2. Short interest percentage statistics
SELECT 
    'Short Interest % Statistics' as metric_type,
    COUNT(*) as records_count,
    MIN(short_interest_pct) as min_pct,
    MAX(short_interest_pct) as max_pct,
    AVG(short_interest_pct) as avg_pct,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY short_interest_pct) as median_pct,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY short_interest_pct) as p95_pct
FROM ss_finra_short_interest 
WHERE short_interest_pct IS NOT NULL;

-- 3. Days to cover statistics
SELECT 
    'Days to Cover Statistics' as metric_type,
    COUNT(*) as records_count,
    MIN(days_to_cover) as min_days,
    MAX(days_to_cover) as max_days,
    AVG(days_to_cover) as avg_days,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_to_cover) as median_days,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY days_to_cover) as p95_days
FROM ss_finra_short_interest 
WHERE days_to_cover IS NOT NULL;

-- 4. Distribution of short interest levels
SELECT 
    CASE 
        WHEN short_interest_pct >= 30 THEN 'VERY HIGH (≥30%)'
        WHEN short_interest_pct >= 20 THEN 'HIGH (20-30%)'
        WHEN short_interest_pct >= 10 THEN 'MEDIUM (10-20%)'
        WHEN short_interest_pct >= 5 THEN 'LOW (5-10%)'
        ELSE 'VERY LOW (<5%)'
    END as short_interest_level,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM ss_finra_short_interest 
WHERE short_interest_pct IS NOT NULL
GROUP BY 
    CASE 
        WHEN short_interest_pct >= 30 THEN 'VERY HIGH (≥30%)'
        WHEN short_interest_pct >= 20 THEN 'HIGH (20-30%)'
        WHEN short_interest_pct >= 10 THEN 'MEDIUM (10-20%)'
        WHEN short_interest_pct >= 5 THEN 'LOW (5-10%)'
        ELSE 'VERY LOW (<5%)'
    END
ORDER BY 
    MIN(short_interest_pct) DESC;

-- 5. Distribution of days to cover levels
SELECT 
    CASE 
        WHEN days_to_cover >= 10 THEN 'HIGH (≥10 days)'
        WHEN days_to_cover >= 5 THEN 'MEDIUM (5-10 days)'
        WHEN days_to_cover >= 2 THEN 'LOW (2-5 days)'
        ELSE 'VERY LOW (<2 days)'
    END as days_to_cover_level,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM ss_finra_short_interest 
WHERE days_to_cover IS NOT NULL
GROUP BY 
    CASE 
        WHEN days_to_cover >= 10 THEN 'HIGH (≥10 days)'
        WHEN days_to_cover >= 5 THEN 'MEDIUM (5-10 days)'
        WHEN days_to_cover >= 2 THEN 'LOW (2-5 days)'
        ELSE 'VERY LOW (<2 days)'
    END
ORDER BY 
    MIN(days_to_cover) DESC;

-- 6. Top short interest candidates
SELECT 
    ticker,
    settlement_date,
    short_interest_shares,
    float_shares,
    short_interest_pct,
    days_to_cover,
    CASE 
        WHEN short_interest_pct >= 20 AND days_to_cover >= 5 THEN 'HIGH SQUEEZE POTENTIAL'
        WHEN short_interest_pct >= 15 AND days_to_cover >= 3 THEN 'MEDIUM SQUEEZE POTENTIAL'
        WHEN short_interest_pct >= 10 THEN 'LOW SQUEEZE POTENTIAL'
        ELSE 'MINIMAL SQUEEZE POTENTIAL'
    END as squeeze_potential
FROM ss_finra_short_interest 
WHERE short_interest_pct IS NOT NULL 
  AND days_to_cover IS NOT NULL
ORDER BY short_interest_pct DESC, days_to_cover DESC 
LIMIT 20;

-- 7. Recent data quality check
SELECT 
    'Recent Data Quality' as check_type,
    COUNT(*) as total_recent_records,
    COUNT(short_interest_pct) as recent_with_pct,
    COUNT(days_to_cover) as recent_with_days,
    COUNT(float_shares) as recent_with_float,
    ROUND(COUNT(short_interest_pct) * 100.0 / COUNT(*), 2) as recent_pct_coverage,
    ROUND(COUNT(days_to_cover) * 100.0 / COUNT(*), 2) as recent_days_coverage
FROM ss_finra_short_interest 
WHERE settlement_date >= CURRENT_DATE - INTERVAL '90 days';

-- 8. Identify records still missing data
SELECT 
    'Missing Data Analysis' as analysis_type,
    COUNT(*) FILTER (WHERE short_interest_pct IS NULL AND float_shares IS NOT NULL AND float_shares > 0) as missing_pct_with_float,
    COUNT(*) FILTER (WHERE days_to_cover IS NULL AND short_interest_shares > 0) as missing_days_with_shares,
    COUNT(*) FILTER (WHERE float_shares IS NULL OR float_shares = 0) as missing_float_shares,
    COUNT(*) FILTER (WHERE short_interest_pct IS NULL AND days_to_cover IS NULL) as missing_both_calculated
FROM ss_finra_short_interest;

-- 9. Sample of complete records
SELECT 
    ticker,
    settlement_date,
    short_interest_shares,
    float_shares,
    short_interest_pct,
    days_to_cover,
    data_quality_score,
    updated_at
FROM ss_finra_short_interest 
WHERE short_interest_pct IS NOT NULL 
  AND days_to_cover IS NOT NULL
  AND settlement_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY short_interest_pct DESC 
LIMIT 10;