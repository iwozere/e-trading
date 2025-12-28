python -m unittest tests/test_indicator.py

If you want to run all tests in the tests/ directory, you can use:
python -m unittest discover -s tests



# Test Suite Documentation

This directory contains comprehensive unit tests for all data downloaders and live feeds in the e-trading platform.

## Overview

The test suite covers:
- **Data Downloaders**: All 7 data downloader implementations
- **Live Feeds**: All 4 live feed implementations  
- **Factories**: Data downloader and live feed factories
- **Integration**: End-to-end testing scenarios

## Test Structure

### Data Downloader Tests

#### `test_data_downloaders.py`
Comprehensive tests for all data downloader implementations:

- **YahooDataDownloader**: OHLCV and fundamental data retrieval
- **AlphaVantageDataDownloader**: API integration and rate limiting
- **FinnhubDataDownloader**: Real-time data and comprehensive fundamentals
- **PolygonDataDownloader**: US market data and basic fundamentals
- **TwelveDataDataDownloader**: Global coverage and basic data
- **BinanceDataDownloader**: Cryptocurrency data (no fundamentals)
- **CoinGeckoDataDownloader**: Cryptocurrency data (no fundamentals)

**Test Coverage:**
- ✅ OHLCV data retrieval
- ✅ Fundamental data retrieval
- ✅ API key validation
- ✅ Error handling
- ✅ Data format validation
- ✅ Rate limiting
- ✅ Provider-specific features

#### `test_base_data_downloader.py`
Tests for the base data downloader functionality:

- ✅ File operations (save/load)
- ✅ Multiple symbol downloads
- ✅ Data validation
- ✅ Error handling

### Live Feed Tests

#### `test_live_feeds.py`
Comprehensive tests for all live feed implementations:

- **BinanceLiveDataFeed**: WebSocket-based real-time data
- **YahooLiveDataFeed**: Polling-based stock data
- **IBKRLiveDataFeed**: Professional trading platform integration
- **CoinGeckoLiveDataFeed**: Polling-based cryptocurrency data

**Test Coverage:**
- ✅ Historical data loading
- ✅ Real-time connection handling
- ✅ Data streaming
- ✅ Error handling and reconnection
- ✅ Backtrader integration
- ✅ WebSocket and polling mechanisms
- ✅ Status monitoring

#### `test_live_data_feeds.py`
Integration tests for live data feeds:

- ✅ End-to-end data flow
- ✅ Multiple data source testing
- ✅ Performance testing
- ✅ Error recovery

### Factory Tests

#### `test_data_downloader_factory.py`
Tests for the DataDownloaderFactory:

- ✅ Provider code mapping
- ✅ Downloader creation
- ✅ Environment variable support
- ✅ Error handling
- ✅ Provider information

#### `test_data_feed_factory.py`
Tests for the DataFeedFactory:

- ✅ Live feed creation
- ✅ Configuration handling
- ✅ Source information
- ✅ Error handling

## Running Tests

### Prerequisites

Install testing dependencies:
```bash
pip install -r requirements-test.txt
```

### Basic Test Execution

Run all tests:
```bash
python run_data_tests.py
```

Run specific test categories:
```bash
# Data downloaders only
python run_data_tests.py --downloaders

# Live feeds only
python run_data_tests.py --live-feeds

# Factories only
python run_data_tests.py --factories
```

### Advanced Test Execution

Run with verbose output:
```bash
python run_data_tests.py --verbose
```

Run with coverage report:
```bash
python run_data_tests.py --coverage
```

Run specific test file:
```bash
python run_data_tests.py --test-file tests/test_data_downloaders.py
```

Run specific test class:
```bash
python run_data_tests.py --test-file tests/test_data_downloaders.py --test-class TestYahooDataDownloader
```

Run specific test method:
```bash
python run_data_tests.py --test-file tests/test_data_downloaders.py --test-class TestYahooDataDownloader --test-method test_get_fundamentals
```

### Using pytest directly

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src/data --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_data_downloaders.py
```

Run specific test class:
```bash
pytest tests/test_data_downloaders.py::TestYahooDataDownloader
```

Run specific test method:
```bash
pytest tests/test_data_downloaders.py::TestYahooDataDownloader::test_get_fundamentals
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Speed**: Fast execution
- **Dependencies**: Mocked external services
- **Coverage**: High coverage of individual functions

### Integration Tests
- **Purpose**: Test component interactions
- **Speed**: Medium execution time
- **Dependencies**: Some real API calls (with rate limiting)
- **Coverage**: End-to-end workflows

### Mock Tests
- **Purpose**: Test with simulated external services
- **Speed**: Very fast execution
- **Dependencies**: No external calls
- **Coverage**: API interaction patterns

## Test Data

### Mock Data
All tests use realistic mock data that mimics real API responses:

- **OHLCV Data**: Properly formatted price and volume data
- **Fundamental Data**: Complete financial metrics
- **API Responses**: Realistic JSON structures
- **Error Scenarios**: Network failures, rate limits, invalid data

### Test Symbols
- **Stocks**: AAPL, MSFT, GOOGL, AMZN
- **Cryptocurrencies**: BTCUSDT, ETHUSDT, bitcoin, ethereum
- **Indices**: ^GSPC, ^DJI, ^IXIC

## Coverage Requirements

### Minimum Coverage
- **Line Coverage**: 80%
- **Branch Coverage**: 75%
- **Function Coverage**: 90%

### Coverage Reports
Coverage reports are generated in multiple formats:
- **Terminal**: `--cov-report=term-missing`
- **HTML**: `--cov-report=html:htmlcov`
- **XML**: `--cov-report=xml`

## Environment Variables

### Required for Testing
```bash
# API Keys (for integration tests)
ALPHA_VANTAGE_API_KEY=your_key
FINNHUB_API_KEY=your_key
POLYGON_API_KEY=your_key
TWELVE_DATA_API_KEY=your_key
BINANCE_KEY=your_key
BINANCE_SECRET=your_secret
```

### Test Environment
The test suite automatically sets test API keys when not provided.

## Test Configuration

### pytest.ini
Configuration file with:
- Test discovery patterns
- Output formatting
- Coverage settings
- Environment variables
- Timeout settings

### Test Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow running tests
- `@pytest.mark.api`: API-dependent tests
- `@pytest.mark.mock`: Mocked tests

## Continuous Integration

### GitHub Actions
Tests are automatically run on:
- Pull requests
- Push to main branch
- Scheduled runs

### CI Pipeline
1. **Install dependencies**
2. **Run unit tests**
3. **Run integration tests**
4. **Generate coverage report**
5. **Upload artifacts**

## Debugging Tests

### Verbose Output
```bash
pytest -vvv
```

### Debug Mode
```bash
pytest --pdb
```

### Show Local Variables
```bash
pytest -l
```

### Stop on First Failure
```bash
pytest -x
```

## Performance Testing

### Benchmark Tests
```bash
pytest --benchmark-only
```

### Performance Monitoring
- Response time tracking
- Memory usage monitoring
- Rate limit compliance

## Security Testing

### Bandit Integration
```bash
bandit -r src/data/
```

### Security Checks
- API key exposure
- Input validation
- Error message security

## Contributing

### Adding New Tests

1. **Follow naming conventions**:
   - Test files: `test_*.py`
   - Test classes: `Test*`
   - Test methods: `test_*`

2. **Use appropriate markers**:
   ```python
   @pytest.mark.unit
   def test_something():
       pass
   ```

3. **Mock external dependencies**:
   ```python
   @patch('requests.get')
   def test_api_call(self, mock_get):
       mock_get.return_value = MockResponse()
   ```

4. **Add comprehensive assertions**:
   ```python
   def assert_valid_data(self, data):
       self.assertIsInstance(data, pd.DataFrame)
       self.assertGreater(len(data), 0)
   ```

### Test Best Practices

1. **Arrange-Act-Assert**: Structure tests clearly
2. **Descriptive names**: Use clear test method names
3. **Isolation**: Each test should be independent
4. **Mocking**: Mock external dependencies
5. **Coverage**: Aim for high test coverage
6. **Documentation**: Document complex test scenarios

### Test Data Management

1. **Use fixtures**: For common test data
2. **Clean up**: Remove temporary files
3. **Realistic data**: Use realistic mock data
4. **Edge cases**: Test boundary conditions

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src` is in Python path
2. **API Rate Limits**: Use appropriate delays in tests
3. **Network Issues**: Mock external calls
4. **Missing Dependencies**: Install `requirements-test.txt`

### Debug Commands

```bash
# Check test discovery
pytest --collect-only

# Run with maximum verbosity
pytest -vvv --tb=long

# Profile test execution
pytest --durations=10

# Check coverage details
pytest --cov=src/data --cov-report=term-missing
```

## Test Results

### Success Criteria
- All tests pass
- Coverage meets minimum requirements
- No security vulnerabilities
- Performance within acceptable limits

### Reporting
- Test results summary
- Coverage reports
- Performance benchmarks
- Security scan results

## Maintenance

### Regular Tasks
- Update mock data to match API changes
- Review and update test dependencies
- Monitor test performance
- Update coverage requirements

### Test Review
- Monthly test suite review
- Quarterly coverage analysis
- Annual test strategy update 