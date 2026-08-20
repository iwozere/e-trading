-- Migration: Widen virality_index's valid range from [0,1] to [0, +inf)
-- Version: 4.0
-- Date: 2026-08-20
-- Description: Rev 2 Phase 3 redefined virality_index (sentiment-spec-rev2.md §2.5.5) from a
-- sentiment-weighted ~0..1 score into unsigned reach only: Σengagement / sqrt(unique_authors+1).
-- That quantity is unbounded above -- a popular ticker with real engagement routinely exceeds 1.
-- The original check_virality_range constraint (migration 001) still enforces <= 1 and would
-- reject every such row. This migration relaxes it to >= 0 only.
--
-- The virality_index column itself is FLOAT (double precision), not NUMERIC, so there is no
-- column-width limit to widen -- only the check constraint needs to change.

BEGIN;

ALTER TABLE ss_deep_metrics DROP CONSTRAINT IF EXISTS check_virality_range;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'check_virality_range') THEN
        ALTER TABLE ss_deep_metrics
            ADD CONSTRAINT check_virality_range CHECK (virality_index >= 0);
    END IF;
END $$;

-- The old partial index assumed the 0..1 scale (WHERE virality_index > 0.5, i.e. "top half").
-- Rebuild it without a scale-dependent predicate -- any positive value is now a normal reading,
-- not necessarily high.
DROP INDEX IF EXISTS idx_ss_deep_metrics_virality;
CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_virality ON ss_deep_metrics(virality_index DESC);

COMMENT ON COLUMN ss_deep_metrics.virality_index IS
'Unsigned reach: Σengagement / sqrt(unique_authors + 1). Unbounded above -- redefined from a 0..1 sentiment-weighted score in Rev 2 (sentiment-spec-rev2.md §2.5.5); orthogonal to sentiment_24h''s direction.';

COMMIT;

-- Verification query
-- SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'check_virality_range';
