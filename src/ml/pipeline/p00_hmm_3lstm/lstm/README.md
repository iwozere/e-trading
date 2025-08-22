# Regime-Based LSTM Time Series Forecasting

This project provides a script to train Long Short-Term Memory (LSTM) models for time series forecasting. Its key feature is training separate, specialized models for different market regimes (e.g., bullish, bearish, sideways) to potentially improve prediction accuracy.

The script automates the entire pipeline: it processes input data, trains a model for each regime, saves the trained model weights, and logs the hyperparameters used for complete reproducibility.

## Key Features

-   **Regime-Specific Training**: Automatically trains a distinct LSTM model for each market condition defined in the data, allowing for expert models.
-   **Automated Data Processing**: Reads CSV files, applies standard scaling to features, and creates time-series sequences for training.
-   **Reproducibility**: Saves the exact hyperparameters for each training run in a structured JSON file alongside the model.
-   **Configurable**: Easily adjust model architecture (layers, hidden units), training parameters (epochs, learning rate), and feature sets from a central configuration section.
-   **PyTorch-Based**: Built with PyTorch, a modern and powerful deep learning framework.

## Project Structure

```
.
├── results/
│   ├── BTCUSDT_4h_20220101_20250707.csv             (Input Data)
│   └── LSTM_BTCUSDT_4h_20220101_20250707_bullish.json  (Output Params)
│   └── LSTM_BTCUSDT_4h_20220101_20250707_bearish.json  (Output Params)
│   └── ...
├── scripts/
│   └── train_lstm_all.py                            (The Main Training Script)
├── src/
│   └── ml/
│       └── lstm/
│           └── model/
│               ├── BTCUSDT_4h_20220101_20250707_bullish.pt (Output Model)
│               ├── BTCUSDT_4h_20220101_20250707_bearish.pt (Output Model)
│               └── ...
└── README.md
```

## Setup and Installation

#### Prerequisites

-   Python 3.8+
-   A new virtual environment is highly recommended.

#### Installation

1.  Clone the repository or download the project files.

2.  Create a `requirements.txt` file with the following content:
    ```txt
    numpy
    pandas
    scikit-learn
    torch
    ```

3.  Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Place Your Data**:
    -   Prepare your input data as one or more `.csv` files.
    -   Each CSV file **must** contain a `regime` column with integer labels (e.g., `0` for bearish, `1` for sideways) and all the feature columns specified in the script's `FEATURE_COLS` list.
    -   Place the CSV file(s) into the `results/` directory.

2.  **Configure the Script (Optional)**:
    -   Open `scripts/train_lstm_all.py`.
    -   Adjust parameters in the `---- Config ----` section as needed (e.g., `EPOCHS`, `WINDOW`, `HIDDEN_DIM`).

3.  **Execute the Training Script**:
    -   Run the script from the root directory of the project:
    ```bash
    python scripts/train_lstm_all.py
    ```
    The script will find all `.csv` files in `results/`, process each one, and train models for every regime found in the data.

## Output Artifacts

For each input CSV and each regime, the script generates two files:

1.  **Trained Model (`.pt`)**: A PyTorch state dictionary containing the model's trained weights.
    -   **Location**: `src/ml/lstm/model/`
    -   **Example**: `BTCUSDT_4h_20220101_20250707_bullish.pt`

2.  **Hyperparameters (`.json`)**: A JSON log of all settings used for the training run.
    -   **Location**: `results/`
    -   **Example**: `LSTM_BTCUSDT_4h_20220101_20250707_bullish.json`

## Future Improvements & Next Steps

This project provides a solid foundation for training. The following steps outline how to evaluate, optimize, and deploy these models to create a complete trading or analysis system.

### 1. Add Evaluation Script to Test Predictive Accuracy

Currently, the script only tracks training loss. A dedicated evaluation script is needed to measure the model's performance on unseen data.

-   **Action**: Create a new script, e.g., `scripts/evaluate_lstm.py`.
-   **Functionality**:
    1.  **Data Splitting**: Modify `prepare_data` to split data into training and testing sets (e.g., 80/20 split). The training script would use the training set, and the evaluation script would use the test set.
    2.  **Load Models**: The script should load a trained `.pt` model and its corresponding `.json` parameter file.
    3.  **Make Predictions**: Use the model to predict values on the test dataset.
    4.  **Calculate Metrics**: Compare predictions against actual values and compute key regression metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).

### 2. Integrate Regime-Based LSTM Prediction in Backtrader

To test the economic significance of the predictions, integrate them into a backtesting engine like **Backtrader**.

-   **Action**: Create a `strategies/lstm_strategy.py` file.
-   **Functionality**:
    1.  **Custom Strategy**: Create a new `bt.Strategy` class.
    2.  **Model Loading**: In the strategy's `__init__`, load all regime models (bullish, bearish, sideways).
    3.  **Dynamic Prediction**: In the `next` method, determine the current regime for the latest data point.
    4.  **Select & Predict**: Load the appropriate LSTM model for that regime and generate a prediction for the next period's return.
    5.  **Trading Logic**: Implement trading rules based on the prediction. For example:
        -   If `predicted_log_return > 0.001`, enter a long position.
        -   If `predicted_log_return < -0.001`, enter a short position.
    6.  **Run Backtest**: Use a runner script to execute the backtest and analyze the performance (Sharpe ratio, drawdown, final portfolio value).

### 3. Add Optuna Optimization for LSTM Hyperparameters

The current hyperparameters are hard-coded. Using a library like **Optuna** can automatically find the best combination for a given regime.

-   **Action**: Create an optimization script, e.g., `scripts/optimize_lstm.py`.
-   **Functionality**:
    1.  **Define Objective Function**: Create a function that takes an Optuna `trial` object as input.
    2.  **Suggest Hyperparameters**: Inside the function, use `trial.suggest_...` to propose values for parameters like `LR`, `HIDDEN_DIM`, `NUM_LAYERS`, `WINDOW`, and `BATCH_SIZE`.
    3.  **Train & Evaluate**: Train the LSTM model using these suggested parameters on a validation set.
    4.  **Return Score**: Return a performance metric (e.g., validation loss or MSE) that Optuna should minimize.
    5.  **Run Study**: Create and run an Optuna `study` to find and report the best parameters.

### 4. Deploy Real-Time or Streaming Inference Pipeline

To use the model for live trading, it must be deployed in a real-time environment.

-   **Action**: Design a streaming application using tools like Kafka, Redis Streams, or a simple WebSocket client.
-   **Pipeline Components**:
    1.  **Data Ingestion**: A service that connects to a live data feed (e.g., a crypto exchange's WebSocket API) to get real-time candle data.
    2.  **Feature & Regime Engine**: A component that takes the latest data, calculates all necessary features (`log_return`, `rsi`, etc.), and determines the current market regime.
    3.  **Model Server/Inference Service**: A service (e.g., a simple Flask/FastAPI endpoint) that, upon receiving features and a regime, loads the correct `.pt` model and returns a prediction.
    4.  **Decision Engine**: A final component that takes the prediction and decides whether to send a trade order to an exchange API.
