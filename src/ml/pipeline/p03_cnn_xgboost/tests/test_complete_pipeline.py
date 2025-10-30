"""
Test script to verify the complete CNN + XGBoost pipeline structure.

This script tests that all pipeline stages can be imported and the pipeline
orchestrator is properly configured.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

def test_pipeline_imports():
    """Test that all pipeline stages can be imported."""
    print("Testing pipeline imports...")

    try:
        # Test data loader
        from x_01_data_loader import DataLoader
        print("✅ DataLoader imported successfully")

        # Test CNN training
        from x_02_train_cnn import CNNTrainer, CNN1D
        print("✅ CNNTrainer and CNN1D imported successfully")

        # Test embedding generation
        from x_03_generate_embeddings import EmbeddingGenerator
        print("✅ EmbeddingGenerator imported successfully")

        # Test TA features
        from x_04_ta_features import TAFeatureEngineer
        print("✅ TAFeatureEngineer imported successfully")

        # Test XGBoost optimization
        from x_05_optuna_xgboost import XGBoostOptimizer
        print("✅ XGBoostOptimizer imported successfully")

        # Test XGBoost training
        from x_06_train_xgboost import XGBoostTrainer
        print("✅ XGBoostTrainer imported successfully")

        # Test validation
        from x_07_validate_model import ModelValidator
        print("✅ ModelValidator imported successfully")

        # Test pipeline orchestrator
        from run_pipeline import CNNXGBoostPipeline
        print("✅ CNNXGBoostPipeline imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_config_loading():
    """Test that configuration can be loaded."""
    print("\nTesting configuration loading...")

    try:
        from src.util.config import load_config

        config_path = Path("config/pipeline/p03.yaml")
        if config_path.exists():
            config = load_config(str(config_path))
            print("✅ Configuration loaded successfully")
            print(f"   - Data section: {'data' in config}")
            print(f"   - CNN section: {'cnn' in config}")
            print(f"   - XGBoost section: {'xgboost' in config}")
            print(f"   - Validation section: {'validation' in config}")
            return True
        else:
            print("❌ Configuration file not found")
            return False

    except Exception as e:
        print(f"❌ Configuration loading error: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist."""
    print("\nTesting directory structure...")

    required_dirs = [
        "src/ml/pipeline/p03_cnn_xgboost",
        "src/ml/pipeline/p03_cnn_xgboost/docs",
        "src/ml/pipeline/p03_cnn_xgboost/models",
        "src/ml/pipeline/p03_cnn_xgboost/models/cnn",
        "src/ml/pipeline/p03_cnn_xgboost/models/xgboost",
        "src/ml/pipeline/p03_cnn_xgboost/tests",
        "config/pipeline"
    ]

    all_exist = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory}")
            all_exist = False

    return all_exist

def test_documentation_files():
    """Test that documentation files exist."""
    print("\nTesting documentation files...")

    required_files = [
        "src/ml/pipeline/p03_cnn_xgboost/docs/README.md",
        "src/ml/pipeline/p03_cnn_xgboost/docs/Design.md",
        "src/ml/pipeline/p03_cnn_xgboost/docs/Requirements.md",
        "src/ml/pipeline/p03_cnn_xgboost/docs/Tasks.md",
        "config/pipeline/p03.yaml"
    ]

    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False

    return all_exist

def test_pipeline_stages():
    """Test that all pipeline stage files exist."""
    print("\nTesting pipeline stage files...")

    stage_files = [
        "src/ml/pipeline/p03_cnn_xgboost/x_01_data_loader.py",
        "src/ml/pipeline/p03_cnn_xgboost/x_02_train_cnn.py",
        "src/ml/pipeline/p03_cnn_xgboost/x_03_generate_embeddings.py",
        "src/ml/pipeline/p03_cnn_xgboost/x_04_ta_features.py",
        "src/ml/pipeline/p03_cnn_xgboost/x_05_optuna_xgboost.py",
        "src/ml/pipeline/p03_cnn_xgboost/x_06_train_xgboost.py",
        "src/ml/pipeline/p03_cnn_xgboost/x_07_validate_model.py",
        "src/ml/pipeline/p03_cnn_xgboost/run_pipeline.py"
    ]

    all_exist = True
    for file_path in stage_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False

    return all_exist

def main():
    """Run all tests."""
    print("=" * 60)
    print("CNN + XGBoost Pipeline Structure Test")
    print("=" * 60)

    tests = [
        ("Pipeline Imports", test_pipeline_imports),
        ("Configuration Loading", test_config_loading),
        ("Directory Structure", test_directory_structure),
        ("Documentation Files", test_documentation_files),
        ("Pipeline Stages", test_pipeline_stages)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The pipeline structure is complete.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
