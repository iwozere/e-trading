# Requirements

## Python Dependencies
- `aiohttp` >= 3.8.0
- `asyncio` (built-in)
- `transformers` >= 4.20.0 (optional, for HuggingFace sentiment)
- `torch` >= 1.12.0 (optional, for HuggingFace sentiment)

## External Dependencies
- `src.notification` - For logging infrastructure
- `src.data` - For data provider patterns

## External Services
- **StockTwits API**: Public API for social sentiment data
- **Pushshift API**: Reddit data aggregation service
- **HuggingFace Models**: Pre-trained sentiment analysis models

## System Requirements
- Memory requirements: 512MB minimum, 2GB recommended for HF models
- CPU requirements: Multi-core recommended for async processing
- Network: Stable internet connection for API calls

## Security Requirements
- Rate limiting compliance with external APIs
- No API keys stored in code (environment variables)
- Input validation for all external data

## Performance Requirements
- Response time targets: < 5 seconds for batch sentiment analysis
- Throughput requirements: 100+ tickers per minute
- Concurrent request limits: Configurable per provider