# EMPS2 Future Roadmap & TODO

## Performance & Scalability
- [ ] **Parallel Fundamental Filtering**: Implement `ProcessPoolExecutor` or `ThreadPoolExecutor` to bypass 60/min Finnhub limits by using multiple keys or more efficient connection management.
- [ ] **Database Integration**: Migrating results from flat CSV files to a proper PostgreSQL/TimescaleDB schema for better historical querying.
- [ ] **Incremental Updates**: Scan only tickers that have new data or significant price action since the last run.

## Filtering Enhancements
- [ ] **Sector/Industry Rotation Filter**: Prioritize tickers in currently trending sectors.
- [ ] **Advanced UOA metrics**: Include "Golden Sweeps" and dark pool print correlations.
- [ ] **Machine Learning Scoring**: Replace heuristic phase detection with a trained classifier for "Explosive Potential".

## UX & Delivery
- [ ] **Web Dashboard**: Create a simple Vite/Next.js interface to visualize `09_final_universe.csv`.
- [ ] **Detailed Telegram Reports**: In-bot charts generated using `matplotlib` on alert trigger.
- [ ] **Interactive Commands**: Allow users to query ticker metrics directly from Telegram (e.g., `/score AAPL`).

## Technical Debt
- [ ] Standardize logging across all downloaders.
- [ ] Implement full unit test coverage for `alerts.py` and `emps2_pipeline.py`.
- [ ] Refactor `volatility_filter.py` into smaller, more testable components.

---
**Last Updated**: 2026-01-07