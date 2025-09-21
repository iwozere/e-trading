#!/bin/bash
# HMM-LSTM Backtesting Runner for Unix/Linux
# This script runs the HMM-LSTM backtesting optimizer

echo "Starting HMM-LSTM Backtesting..."
echo

# Set Python path and run the optimizer
python3 src/backtester/optimizer/hmm_lstm.py "$@"

echo
echo "HMM-LSTM Backtesting completed."
