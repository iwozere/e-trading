@echo off
REM HMM-LSTM Backtesting Runner for Windows
REM This script runs the HMM-LSTM backtesting optimizer

echo Starting HMM-LSTM Backtesting...
echo.

REM Set Python path and run the optimizer
python src/backtester/optimizer/hmm_lstm.py %*

echo.
echo HMM-LSTM Backtesting completed.
pause
