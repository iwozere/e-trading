-- Migration script to fix the short_interest_pct constraint
-- Short interest can legitimately exceed 100% in real market conditions
-- This script removes the upper limit constraint

-- Start transaction
BEGIN;

-- Drop the existing constraint that limits short_interest_pct to 100%
ALTER TABLE ss_finra_short_interest 
DROP CONSTRAINT IF EXISTS valid_percentage;

-- Add a new constraint that allows values > 100%
-- We keep the lower bound at 0% and set a reasonable upper bound of 1000%
ALTER TABLE ss_finra_short_interest 
ADD CONSTRAINT valid_percentage_updated 
CHECK (short_interest_pct >= 0 AND short_interest_pct <= 1000);

-- Also update the model constraint to match
ALTER TABLE ss_finra_short_interest 
DROP CONSTRAINT IF EXISTS check_short_interest_pct;

ALTER TABLE ss_finra_short_interest 
ADD CONSTRAINT check_short_interest_pct 
CHECK (short_interest_pct >= 0 AND short_interest_pct <= 1000);

-- Commit the changes
COMMIT;

-- Verify the new constraints
SELECT 
    conname as constraint_name,
    pg_get_constraintdef(oid) as constraint_definition
FROM pg_constraint 
WHERE conrelid = 'ss_finra_short_interest'::regclass 
  AND conname LIKE '%percentage%' OR conname LIKE '%short_interest_pct%';

-- Show some statistics about current data
SELECT 
    'Current Data Analysis' as analysis_type,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE short_interest_pct > 100) as records_over_100_pct,
    MAX(short_interest_pct) as max_short_interest_pct,
    AVG(short_interest_pct) as avg_short_interest_pct
FROM ss_finra_short_interest 
WHERE short_interest_pct IS NOT NULL;

-- Script completed successfully