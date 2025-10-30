# Design

## Purpose
The sentiment analysis module provides real-time sentiment scoring for financial instruments by aggregating data from multiple social media sources. It serves the short squeeze detection pipeline and other trading strategies that require sentiment-based signals.

## Architecture
The system follows an async adapter pattern that allows concurrent data collection from multiple providers while maintaining rate limit compliance and error resilience.

### High-Level Architecture
- **Async Collectors**: Individual adapters for each data source
- **Aggregation Engine**: Combines multiple sentiment signals
- **HuggingFace Integration**: Optional ML-based sentiment analysis
- **Batch Processing**: Efficient concurrent processing of multiple tickers

### Component Design
- **AsyncStocktwitsAdapter**: Social sentiment from StockTwits platform
- **AsyncPushshiftAdapter**: Reddit sentiment via Pushshift API
- **AsyncHFSentiment**: ML-based sentiment using HuggingFace transformers
- **SentimentFeatures**: Standardized output dataclass

## Data Flow
1. **Input**: List of ticker symbols
2. **Collection**: Concurrent API calls to multiple providers
3. **Processing**: Heuristic and ML-based sentiment analysis
4. **Aggregation**: Weighted combination of multiple signals
5. **Output**: Normalized sentiment features per ticker

## Design Decisions

### Technology Choices
- **Async/Await Pattern**: Enables high-concurrency data collection
- **aiohttp**: Non-blocking HTTP client for API calls
- **HuggingFace Transformers**: State-of-the-art sentiment models
- **ThreadPoolExecutor**: Isolates blocking ML inference from async loop

### Architecture Patterns
- **Adapter Pattern**: Standardized interface for different data sources
- **Factory Pattern**: Configurable provider instantiation
- **Semaphore Pattern**: Rate limiting and concurrency control
- **Circuit Breaker**: Graceful degradation on provider failures

### Performance Considerations
- **Concurrent Processing**: Multiple tickers processed simultaneously
- **Rate Limiting**: Respects API provider limits
- **Caching**: Avoids redundant API calls within time windows
- **Lazy Loading**: HF models loaded only when needed

### Security Decisions
- **Input Validation**: All ticker symbols sanitized
- **Error Isolation**: Provider failures don't affect other providers
- **Rate Limit Compliance**: Prevents API abuse
- **No Credential Storage**: Environment-based configuration