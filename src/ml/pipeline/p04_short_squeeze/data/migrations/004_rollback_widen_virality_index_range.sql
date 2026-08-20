-- Rollback Migration: Restore virality_index's [0,1] range constraint
-- Version: 4.0
-- Date: 2026-08-20
-- Description: Rolls back 004_widen_virality_index_range.sql
-- WARNING: this will fail if any row already has virality_index > 1 (expected once the Rev 2
-- unsigned-reach formula has been running) -- clean up or cap such rows before rolling back.

BEGIN;

ALTER TABLE ss_deep_metrics DROP CONSTRAINT IF EXISTS check_virality_range;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'check_virality_range') THEN
        ALTER TABLE ss_deep_metrics
            ADD CONSTRAINT check_virality_range CHECK (virality_index >= 0 AND virality_index <= 1);
    END IF;
END $$;

DROP INDEX IF EXISTS idx_ss_deep_metrics_virality;
CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_virality ON ss_deep_metrics(virality_index DESC) WHERE virality_index > 0.5;

COMMIT;

-- Verification query
-- SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'check_virality_range';
