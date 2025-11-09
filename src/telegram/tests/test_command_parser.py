import pytest
from src.telegram.command_parser import parse_command

@pytest.mark.parametrize("text,expected", [
    ("/report AAPL", dict(command="report", args={"tickers": "AAPL", "email": False, "indicators": None, "period": None, "interval": None, "provider": None})),
    ("/report btcusdt -email", dict(command="report", args={"tickers": "BTCUSDT", "email": True, "indicators": None, "period": None, "interval": None, "provider": None})),
    ("/report AAPL BTCUSDT -email -indicators=RSI,MACD,MA50", dict(command="report", args={"tickers": "AAPL", "email": True, "indicators": "rsi,macd,ma50", "period": None, "interval": None, "provider": None})),
    ("/report tsla -period=3mo -interval=15m", dict(command="report", args={"tickers": "TSLA", "email": False, "indicators": None, "period": "3mo", "interval": "15m", "provider": None})),
    ("/report btcusdt --provider=bnc --period=1y --interval=1d", dict(command="report", args={"tickers": "BTCUSDT", "email": False, "indicators": None, "period": "1y", "interval": "1d", "provider": "bnc"})),
    ("/report aapl --unknownflag --email", dict(command="report", args={"tickers": "AAPL", "email": True, "indicators": None, "period": None, "interval": None, "provider": None}, extra_flags={"unknownflag": True})),
    ("/report aapl --indicators=RSI", dict(command="report", args={"tickers": "AAPL", "email": False, "indicators": "rsi", "period": None, "interval": None, "provider": None})),
    ("/report aapl --provider=YF", dict(command="report", args={"tickers": "AAPL", "email": False, "indicators": None, "period": None, "interval": None, "provider": None})),
    ("/report aapl -period=2y -interval=1d -provider=yf -email -indicators=RSI,MACD", dict(command="report", args={"tickers": "AAPL", "email": True, "indicators": "rsi,macd", "period": "2y", "interval": "1d", "provider": "yf"})),
    ("/alerts add btcusdt 65000 above", dict(command="alerts", positionals=["ADD", "BTCUSDT", "65000", "ABOVE"])),
    ("/alerts delete 2", dict(command="alerts", positionals=["DELETE", "2"])),
    ("/schedules add aapl 09:00 -email", dict(command="schedules", positionals=["ADD", "AAPL", "09:00"], extra_flags={"email": True})),
    ("/admin setlimit alerts 10", dict(command="admin", positionals=["SETLIMIT", "ALERTS", "10"])),
    ("/feedback Please add moving averages!", dict(command="feedback", positionals=["PLEASE", "ADD", "MOVING", "AVERAGES!"]))
])
def test_parse_command_all_cases(text, expected):
    result = parse_command(text)
    assert result.command == expected["command"]
    # Check args (tickers always upper, others lower)
    if "args" in expected:
        for k, v in expected["args"].items():
            if k == "tickers":
                assert result.args[k].isupper()
            elif v is not None:
                assert (result.args[k] == v)
    # Check positionals (tickers always upper)
    if "positionals" in expected:
        for idx, val in enumerate(expected["positionals"]):
            if idx == 1 and result.positionals[idx].isalpha():  # likely ticker
                assert result.positionals[idx].isupper()
            else:
                assert result.positionals[idx].lower() == val.lower()
    # Check extra flags
    if "extra_flags" in expected:
        for k, v in expected["extra_flags"].items():
            assert result.extra_flags[k] == v

def test_parse_command_case_insensitivity():
    # Command and flags should be lower, tickers upper
    result = parse_command("/REPORT aapl -EMAIL -PROVIDER=YF")
    assert result.command == "report"
    assert result.args["tickers"] == "AAPL"
    assert result.args["email"] is True
    assert result.args["provider"] == "yf"

# Test unknown/extra flags
def test_parse_command_extra_flags():
    result = parse_command("/report AAPL -foo=bar -baz -email")
    assert result.extra_flags["foo"] == "bar"
    assert result.extra_flags["baz"] is True
    assert result.args["email"] is True
    assert result.args["tickers"] == "AAPL"
