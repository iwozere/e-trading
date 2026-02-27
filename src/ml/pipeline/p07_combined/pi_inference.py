import onnxruntime as ort
import pandas as pd
import numpy as np
import json
import talib
from pathlib import Path

class P07PiClient:
    """
    Minimal Inference Client for Raspberry Pi.
    Dependencies: onnxruntime, pandas, ta-lib (C core)
    """

    def __init__(self, model_path: Path, metadata_path: Path):
        self.session = ort.InferenceSession(str(model_path))
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata['feature_names']
        # Scaler params for manual reconstruction to avoid sklearn dependency
        self.means = np.array(self.metadata['scaler_params']['mean'])
        self.scales = np.array(self.metadata['scaler_params']['scale'])

    def preprocess(self, latest_ohlcv: pd.DataFrame) -> np.ndarray:
        """
        Calculate features manually or via ta-lib to match research.
        """
        # 1. Feature calculation (Simplified, mapping metadata to TA-Lib)
        # 2. Scaling
        # raw_features = ... (vector of values in correct order)
        # scaled = (raw_features - self.means) / self.scales
        return np.zeros((1, len(self.feature_names))).astype(np.float32)

    def predict(self, feature_vector: np.ndarray) -> int:
        """Run ONNX inference."""
        inputs = {self.session.get_inputs()[0].name: feature_vector}
        probs = self.session.run(None, inputs)[0]
        # probs: [Sell, Hold, Buy]
        return int(np.argmax(probs) - 1) # -1, 0, 1

if __name__ == "__main__":
    # Minimal example
    # client = P07PiClient(Path("model.onnx"), Path("metadata.json"))
    # signal = client.predict(client.preprocess(latest_data))
    pass
