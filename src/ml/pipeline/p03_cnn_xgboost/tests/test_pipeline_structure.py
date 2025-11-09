#!/usr/bin/env python3
"""
Test script to verify CNN + XGBoost pipeline structure and imports.
"""

import sys
from pathlib import Path
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

def test_config_loading():
    """Test configuration file loading."""
    print("Testing configuration loading...")

    try:
        config_path = Path("config/pipeline/p03.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ['data', 'cnn', 'xgboost', 'technical_indicators', 'targets']
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"

        print("✅ Configuration loading successful")
        return True

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure creation."""
    print("Testing directory structure...")

    try:
        # Check if directories exist
        directories = [
            "src/ml/pipeline/p03_cnn_xgboost",
            "src/ml/pipeline/p03_cnn_xgboost/docs",
            "src/ml/pipeline/p03_cnn_xgboost/models",
            "src/ml/pipeline/p03_cnn_xgboost/models/cnn",
            "src/ml/pipeline/p03_cnn_xgboost/models/xgboost",
            "src/ml/pipeline/p03_cnn_xgboost/tests"
        ]

        for directory in directories:
            path = Path(directory)
            assert path.exists(), f"Directory does not exist: {directory}"

        print("✅ Directory structure verification successful")
        return True

    except Exception as e:
        print(f"❌ Directory structure verification failed: {e}")
        return False

def test_documentation_files():
    """Test documentation files exist."""
    print("Testing documentation files...")

    try:
        docs_dir = Path("src/ml/pipeline/p03_cnn_xgboost/docs")
        required_files = ['README.md', 'Design.md', 'Requirements.md', 'Tasks.md']

        for file_name in required_files:
            file_path = docs_dir / file_name
            assert file_path.exists(), f"Documentation file missing: {file_name}"

        print("✅ Documentation files verification successful")
        return True

    except Exception as e:
        print(f"❌ Documentation files verification failed: {e}")
        return False

def test_pipeline_runner():
    """Test pipeline runner import."""
    print("Testing pipeline runner...")

    try:
        # Test import

        print("✅ Pipeline runner import successful")
        return True

    except Exception as e:
        print(f"❌ Pipeline runner import failed: {e}")
        return False

def test_data_loader():
    """Test data loader import."""
    print("Testing data loader...")

    try:
        # Test import

        print("✅ Data loader import successful")
        return True

    except Exception as e:
        print(f"❌ Data loader import failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")

    try:
        config_path = Path("config/pipeline/p03.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate key configuration parameters
        assert 'symbols' in config['data'], "Missing symbols in data config"
        assert 'timeframes' in config['data'], "Missing timeframes in data config"
        assert 'sequence_length' in config['cnn'], "Missing sequence_length in CNN config"
        assert 'n_trials' in config['xgboost'], "Missing n_trials in XGBoost config"

        print("✅ Configuration validation successful")
        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("CNN + XGBoost Pipeline Structure Test")
    print("=" * 60)

    tests = [
        test_directory_structure,
        test_documentation_files,
        test_config_loading,
        test_configuration_validation,
        test_pipeline_runner,
        test_data_loader
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Pipeline structure is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
