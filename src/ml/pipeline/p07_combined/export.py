import onnxmltools
from onnxconverter_common.data_types import FloatTensorType
import joblib
from pathlib import Path
from typing import List

def export_to_onnx(xgb_model: Any, X_sample: pd.DataFrame, output_path: Path):
    """
    Exports a trained XGBoost model to ONNX format.
    Requires: onnxmltools, xgboost, onnxruntime
    """
    num_features = X_sample.shape[1]
    initial_type = [('float_input', FloatTensorType([None, num_features]))]

    # XGBoost to ONNX
    onnx_model = onnxmltools.convert_xgboost(xgb_model, initial_types=initial_type)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model exported to {output_path}")

def save_metadata(feature_names: List[str], scaler_params: Dict, output_path: Path):
    """Save metadata required for parity in production."""
    import json
    metadata = {
        "feature_names": feature_names,
        "scaler_params": scaler_params, # e.g. mean_, scale_ from StandardScaler
        "pipeline_version": "p07_combined_v1.0"
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)
