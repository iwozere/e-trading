# src/common/sentiments/tests/test_configuration_management.py
"""
Unit tests for configuration management and validation.

Tests cover:
- Configuration loading from environment variables
- Configuration validation and normalization
- Default configuration handling
- Configuration merging and overrides
"""

import unittest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.collect_sentiment_async import (
    get_default_config, validate_config, _load_config_from_env, DEFAULT_CONFIG
)


class TestConfigurationManagement(unittest.TestCase):
    """Test cases for configuration management functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Store original environment variables
        self.original_env = dict(os.environ)

    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_config_structure(self):
        """Test default configuration structure and values."""
        config = get_default_config()

        # Check required sections exist
        required_sections = ["providers", "batching", "weights", "heuristic", "caching"]
        for section in required_sections:
            self.assertIn(section, config, f"Missing required section: {section}")

        # Check provider settings
        self.assertIn("stocktwits", config["providers"])
        self.assertIn("reddit_pushshift", config["providers"])
        self.assertIn("hf_enabled", config["providers"])

        # Check batching settings
        self.assertIn("concurrency", config["batching"])
        self.assertIn("rate_limit_delay_sec", config["batching"])

        # Check weights
        self.assertIn("stocktwits", config["weights"])
        self.assertIn("reddit", config["weights"])
        self.assertIn("heuristic_vs_hf", config["weights"])

        # Check heuristic settings
        self.assertIn("positive_tokens", config["heuristic"])
        self.assertIn("negative_tokens", config["heuristic"])
        self.assertIn("engagement_weight_formula", config["heuristic"])

        # Check caching settings
        self.assertIn("redis_enabled", config["caching"])
        self.assertIn("memory_max_size", config["caching"])

    def test_default_config_values(self):
        """Test default configuration values are reasonable."""
        config = get_default_config()

        # Check numeric values are positive
        self.assertGreater(config["lookback_hours"], 0)
        self.assertGreater(config["batching"]["concurrency"], 0)
        self.assertGreaterEqual(config["batching"]["rate_limit_delay_sec"], 0)

        # Check weights are valid
        self.assertGreater(config["weights"]["stocktwits"], 0)
        self.assertGreater(config["weights"]["reddit"], 0)
        self.assertGreaterEqual(config["weights"]["heuristic_vs_hf"], 0)
        self.assertLessEqual(config["weights"]["heuristic_vs_hf"], 1)

        # Check boolean values
        self.assertIsInstance(config["providers"]["stocktwits"], bool)
        self.assertIsInstance(config["providers"]["reddit_pushshift"], bool)
        self.assertIsInstance(config["providers"]["hf_enabled"], bool)

    @patch.dict(os.environ, {
        "SENTIMENT_STOCKTWITS_ENABLED": "false",
        "SENTIMENT_REDDIT_ENABLED": "true",
        "SENTIMENT_HF_ENABLED": "true",
        "SENTIMENT_LOOKBACK_HOURS": "48",
        "SENTIMENT_CONCURRENCY": "16",
        "SENTIMENT_WEIGHT_STOCKTWITS": "0.3",
        "SENTIMENT_WEIGHT_REDDIT": "0.7"
    })
    def test_environment_variable_override(self):
        """Test configuration override from environment variables."""
        config = get_default_config()

        # Check environment overrides
        self.assertFalse(config["providers"]["stocktwits"])
        self.assertTrue(config["providers"]["reddit_pushshift"])
        self.assertTrue(config["providers"]["hf_enabled"])
        self.assertEqual(config["lookback_hours"], 48)
        self.assertEqual(config["batching"]["concurrency"], 16)
        self.assertEqual(config["weights"]["stocktwits"], 0.3)
        self.assertEqual(config["weights"]["reddit"], 0.7)

    @patch.dict(os.environ, {
        "SENTIMENT_POSITIVE_TOKENS": "moon,rocket,lambo,tendies",
        "SENTIMENT_NEGATIVE_TOKENS": "crash,dump,rekt,fud",
        "SENTIMENT_REDIS_HOST": "redis.example.com",
        "SENTIMENT_REDIS_PORT": "6380",
        "SENTIMENT_CACHE_MEMORY_SIZE": "2000"
    })
    def test_complex_environment_overrides(self):
        """Test complex environment variable overrides."""
        config = get_default_config()

        # Check token lists
        expected_positive = ["moon", "rocket", "lambo", "tendies"]
        expected_negative = ["crash", "dump", "rekt", "fud"]
        self.assertEqual(config["heuristic"]["positive_tokens"], expected_positive)
        self.assertEqual(config["heuristic"]["negative_tokens"], expected_negative)

        # Check Redis settings
        self.assertEqual(config["caching"]["redis_host"], "redis.example.com")
        self.assertEqual(config["caching"]["redis_port"], 6380)
        self.assertEqual(config["caching"]["memory_max_size"], 2000)

    def test_load_config_from_env_empty(self):
        """Test loading configuration from empty environment."""
        # Clear relevant environment variables
        env_vars_to_clear = [key for key in os.environ.keys() if key.startswith("SENTIMENT_")]
        for var in env_vars_to_clear:
            del os.environ[var]

        config = _load_config_from_env()

        # Should return default values
        self.assertTrue(config["providers"]["stocktwits"])
        self.assertTrue(config["providers"]["reddit_pushshift"])
        self.assertFalse(config["providers"]["hf_enabled"])

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        valid_config = {
            "providers": {
                "stocktwits": True,
                "reddit_pushshift": True,
                "hf_enabled": False
            },
            "lookback_hours": 24,
            "batching": {
                "concurrency": 8,
                "rate_limit_delay_sec": 0.3
            },
            "weights": {
                "stocktwits": 0.4,
                "reddit": 0.6,
                "heuristic_vs_hf": 0.5
            },
            "heuristic": {
                "positive_tokens": ["moon", "rocket"],
                "negative_tokens": ["crash", "dump"],
                "engagement_weight_formula": "sqrt"
            }
        }

        # Should not raise exception
        validated_config = validate_config(valid_config)
        self.assertIsInstance(validated_config, dict)

    def test_validate_config_invalid_type(self):
        """Test configuration validation with invalid type."""
        with self.assertRaises(ValueError) as context:
            validate_config("not a dict")

        self.assertIn("must be a dictionary", str(context.exception))

    def test_validate_config_missing_sections(self):
        """Test configuration validation with missing required sections."""
        incomplete_config = {
            "providers": {"stocktwits": True},
            # Missing other required sections
        }

        with self.assertRaises(ValueError) as context:
            validate_config(incomplete_config)

        self.assertIn("Missing required config section", str(context.exception))

    def test_validate_config_invalid_values(self):
        """Test configuration validation with invalid values."""
        # Test negative lookback hours
        invalid_config = {
            "providers": {"stocktwits": True, "reddit_pushshift": True},
            "lookback_hours": -5,  # Invalid
            "batching": {"concurrency": 8, "rate_limit_delay_sec": 0.3},
            "weights": {"stocktwits": 0.4, "reddit": 0.6},
            "heuristic": {"positive_tokens": [], "negative_tokens": []}
        }

        with self.assertRaises(ValueError) as context:
            validate_config(invalid_config)

        self.assertIn("lookback_hours must be positive", str(context.exception))

    def test_validate_config_zero_concurrency(self):
        """Test configuration validation with zero concurrency."""
        invalid_config = {
            "providers": {"stocktwits": True, "reddit_pushshift": True},
            "lookback_hours": 24,
            "batching": {"concurrency": 0, "rate_limit_delay_sec": 0.3},  # Invalid
            "weights": {"stocktwits": 0.4, "reddit": 0.6},
            "heuristic": {"positive_tokens": [], "negative_tokens": []}
        }

        with self.assertRaises(ValueError) as context:
            validate_config(invalid_config)

        self.assertIn("concurrency must be positive", str(context.exception))

    def test_validate_config_zero_weights(self):
        """Test configuration validation with zero provider weights."""
        invalid_config = {
            "providers": {"stocktwits": True, "reddit_pushshift": True},
            "lookback_hours": 24,
            "batching": {"concurrency": 8, "rate_limit_delay_sec": 0.3},
            "weights": {"stocktwits": 0.0, "reddit": 0.0},  # Invalid - sum to zero
            "heuristic": {"positive_tokens": [], "negative_tokens": []}
        }

        with self.assertRaises(ValueError) as context:
            validate_config(invalid_config)

        self.assertIn("Provider weights must sum to a positive value", str(context.exception))

    def test_validate_config_weight_normalization(self):
        """Test configuration validation normalizes weights."""
        config_with_unnormalized_weights = {
            "providers": {"stocktwits": True, "reddit_pushshift": True},
            "lookback_hours": 24,
            "batching": {"concurrency": 8, "rate_limit_delay_sec": 0.3},
            "weights": {"stocktwits": 0.8, "reddit": 1.2},  # Sum to 2.0
            "heuristic": {"positive_tokens": [], "negative_tokens": []}
        }

        validated_config = validate_config(config_with_unnormalized_weights)

        # Weights should be normalized to sum to 1.0
        weight_sum = (validated_config["weights"]["stocktwits"] +
                     validated_config["weights"]["reddit"])
        self.assertAlmostEqual(weight_sum, 1.0, places=3)

        # Relative proportions should be maintained
        self.assertAlmostEqual(validated_config["weights"]["stocktwits"], 0.4, places=3)
        self.assertAlmostEqual(validated_config["weights"]["reddit"], 0.6, places=3)

    def test_config_deep_merge(self):
        """Test deep merging of configuration dictionaries."""
        # Test that nested dictionaries are properly merged
        base_config = get_default_config()

        override_config = {
            "providers": {
                "hf_enabled": True  # Only override this one setting
            },
            "batching": {
                "concurrency": 16  # Only override this one setting
            }
        }

        # Simulate the merging logic from get_default_config
        merged_config = dict(base_config)
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value

        # Check that only specified values were overridden
        self.assertTrue(merged_config["providers"]["hf_enabled"])
        self.assertTrue(merged_config["providers"]["stocktwits"])  # Should remain unchanged
        self.assertEqual(merged_config["batching"]["concurrency"], 16)
        self.assertEqual(merged_config["batching"]["rate_limit_delay_sec"], 0.3)  # Should remain unchanged

    @patch.dict(os.environ, {
        "SENTIMENT_HF_MODEL": "custom-model-name",
        "SENTIMENT_HF_DEVICE": "0",
        "SENTIMENT_HF_WORKERS": "4"
    })
    def test_huggingface_config_override(self):
        """Test HuggingFace configuration override."""
        config = get_default_config()

        self.assertEqual(config["hf"]["model_name"], "custom-model-name")
        self.assertEqual(config["hf"]["device"], 0)
        self.assertEqual(config["hf"]["max_workers"], 4)

    @patch.dict(os.environ, {
        "SENTIMENT_REDIS_ENABLED": "false",
        "SENTIMENT_REDIS_PASSWORD": "secret123",
        "SENTIMENT_CACHE_WARMING": "false"
    })
    def test_caching_config_override(self):
        """Test caching configuration override."""
        config = get_default_config()

        self.assertFalse(config["caching"]["redis_enabled"])
        self.assertEqual(config["caching"]["redis_password"], "secret123")
        self.assertFalse(config["caching"]["warming_enabled"])

    def test_config_immutability(self):
        """Test that DEFAULT_CONFIG is not modified by get_default_config."""
        original_default = dict(DEFAULT_CONFIG)

        # Get config and modify it
        config = get_default_config()
        config["lookback_hours"] = 999
        config["providers"]["stocktwits"] = False

        # DEFAULT_CONFIG should remain unchanged
        self.assertEqual(DEFAULT_CONFIG, original_default)

    def test_boolean_environment_parsing(self):
        """Test parsing of boolean values from environment variables."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("1", False),  # Only "true" should be True
            ("0", False),
            ("yes", False),
            ("no", False),
            ("", False)
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"SENTIMENT_STOCKTWITS_ENABLED": env_value}):
                config = _load_config_from_env()
                self.assertEqual(config["providers"]["stocktwits"], expected,
                               f"Failed for env_value: '{env_value}'")

    def test_numeric_environment_parsing(self):
        """Test parsing of numeric values from environment variables."""
        with patch.dict(os.environ, {
            "SENTIMENT_LOOKBACK_HOURS": "48",
            "SENTIMENT_CONCURRENCY": "16",
            "SENTIMENT_RATE_DELAY": "0.5",
            "SENTIMENT_WEIGHT_STOCKTWITS": "0.3"
        }):
            config = _load_config_from_env()

            self.assertEqual(config["lookback_hours"], 48)
            self.assertEqual(config["batching"]["concurrency"], 16)
            self.assertEqual(config["batching"]["rate_limit_delay_sec"], 0.5)
            self.assertEqual(config["weights"]["stocktwits"], 0.3)

    def test_list_environment_parsing(self):
        """Test parsing of comma-separated lists from environment variables."""
        with patch.dict(os.environ, {
            "SENTIMENT_POSITIVE_TOKENS": "moon,rocket,lambo,  tendies  ,hodl",
            "SENTIMENT_NEGATIVE_TOKENS": "crash, dump ,rekt,  fud"
        }):
            config = _load_config_from_env()

            # Should handle whitespace and empty strings
            expected_positive = ["moon", "rocket", "lambo", "tendies", "hodl"]
            expected_negative = ["crash", "dump", "rekt", "fud"]

            self.assertEqual(config["heuristic"]["positive_tokens"], expected_positive)
            self.assertEqual(config["heuristic"]["negative_tokens"], expected_negative)

    def test_config_validation_edge_cases(self):
        """Test configuration validation edge cases."""
        # Test with minimal valid config
        minimal_config = {
            "providers": {"stocktwits": True, "reddit_pushshift": False},
            "lookback_hours": 24,  # Add required field
            "batching": {"concurrency": 1, "rate_limit_delay_sec": 0},
            "weights": {"stocktwits": 1.0, "reddit": 0.0},
            "heuristic": {"positive_tokens": [], "negative_tokens": []}
        }

        validated = validate_config(minimal_config)
        self.assertIsInstance(validated, dict)

        # Test with missing optional fields but required ones present
        config_missing_optional = {
            "providers": {"stocktwits": True, "reddit_pushshift": True},
            "lookback_hours": 48,  # Add required field
            "batching": {"concurrency": 8, "rate_limit_delay_sec": 0.3},
            "weights": {"stocktwits": 0.5, "reddit": 0.5},
            "heuristic": {"positive_tokens": [], "negative_tokens": []}
            # Missing other optional fields like caching, hf, etc.
        }

        validated = validate_config(config_missing_optional)
        self.assertIsInstance(validated, dict)

    def test_environment_loading_error_handling(self):
        """Test error handling in environment configuration loading."""
        # Mock an exception during environment loading
        with patch('os.getenv', side_effect=Exception("Environment error")):
            config = get_default_config()

            # Should fall back to defaults without crashing
            self.assertIsInstance(config, dict)
            self.assertIn("providers", config)


if __name__ == "__main__":
    unittest.main()