"""
Data validation utilities for the CNN + XGBoost pipeline.

This module provides utilities to validate and ensure consistent data types
and quality across all pipeline stages.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from pathlib import Path
import logging

_logger = logging.getLogger(__name__)


def validate_target_columns(df: pd.DataFrame, target_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that target columns exist and have proper data types.
    
    Args:
        df: DataFrame to validate
        target_columns: List of expected target column names
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check if all target columns exist
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        issues.append(f"Missing target columns: {missing_targets}")
    
    # Check data types for existing target columns
    for col in target_columns:
        if col in df.columns:
            # Check if column contains non-numeric data
            try:
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                issues.append(f"Target column {col} contains non-numeric data")
                continue
            
            # Check for NaN values
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                issues.append(f"Target column {col} has {nan_count} NaN values")
            
            # Check for infinite values
            inf_count = np.isinf(pd.to_numeric(df[col], errors='coerce')).sum()
            if inf_count > 0:
                issues.append(f"Target column {col} has {inf_count} infinite values")
    
    return len(issues) == 0, issues


def convert_targets_to_numeric(df: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
    """
    Convert target columns to numeric types and handle conversion issues.
    
    Args:
        df: DataFrame with target columns
        target_columns: List of target column names
        
    Returns:
        DataFrame with converted target columns
    """
    df_copy = df.copy()
    
    for col in target_columns:
        if col in df_copy.columns:
            # Convert to numeric, coercing errors to NaN
            numeric_values = pd.to_numeric(df_copy[col], errors='coerce')
            
            # Count NaN values after conversion
            nan_count = numeric_values.isna().sum()
            if nan_count > 0:
                _logger.warning("Found %d NaN values in target %s after conversion", nan_count, col)
                
                # Fill NaN values with mode or 0
                if len(numeric_values.mode()) > 0:
                    fill_value = numeric_values.mode().iloc[0]
                else:
                    fill_value = 0
                
                numeric_values = numeric_values.fillna(fill_value)
                _logger.info("Filled %d NaN values in target %s with value %s", nan_count, col, fill_value)
            
            # Convert to integer type
            df_copy[col] = numeric_values.astype(int)
            
            # Validate unique values
            unique_values = df_copy[col].unique()
            _logger.debug("Target %s unique values after conversion: %s", col, unique_values)
    
    return df_copy


def validate_feature_columns(df: pd.DataFrame, exclude_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate feature columns for data quality issues.
    
    Args:
        df: DataFrame to validate
        exclude_columns: Columns to exclude from feature validation
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Get feature columns
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    if not feature_columns:
        issues.append("No feature columns found")
        return False, issues
    
    # Check for NaN values in features
    total_nan = df[feature_columns].isna().sum().sum()
    if total_nan > 0:
        issues.append(f"Found {total_nan} NaN values in feature columns")
    
    # Check for infinite values in features
    numeric_features = pd.to_numeric(df[feature_columns].select_dtypes(include=[np.number]), errors='coerce')
    total_inf = np.isinf(numeric_features).sum().sum()
    if total_inf > 0:
        issues.append(f"Found {total_inf} infinite values in feature columns")
    
    # Check for constant columns
    constant_columns = []
    for col in feature_columns:
        if df[col].nunique() <= 1:
            constant_columns.append(col)
    
    if constant_columns:
        issues.append(f"Found constant columns: {constant_columns}")
    
    return len(issues) == 0, issues


def validate_dataframe_structure(df: pd.DataFrame, expected_columns: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate basic DataFrame structure.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names (optional)
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check if DataFrame is empty
    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    # Check for duplicate columns
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        issues.append(f"Found duplicate columns: {duplicate_columns}")
    
    # Check for expected columns if provided
    if expected_columns:
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing expected columns: {missing_columns}")
    
    return len(issues) == 0, issues


def log_data_quality_report(df: pd.DataFrame, file_path: Path, target_columns: List[str]) -> None:
    """
    Log a comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze
        file_path: Path to the data file
        target_columns: List of target column names
    """
    _logger.info("=== Data Quality Report for %s ===", file_path.name)
    _logger.info("DataFrame shape: %s", df.shape)
    _logger.info("Memory usage: %.2f MB", df.memory_usage(deep=True).sum() / 1024 / 1024)
    
    # Validate structure
    is_valid, issues = validate_dataframe_structure(df)
    if not is_valid:
        for issue in issues:
            _logger.error("Structure issue: %s", issue)
    
    # Validate targets
    is_valid, issues = validate_target_columns(df, target_columns)
    if not is_valid:
        for issue in issues:
            _logger.error("Target issue: %s", issue)
    
    # Validate features
    exclude_columns = target_columns + ["date", "timestamp", "sequence_start_idx", "sequence_end_idx"]
    is_valid, issues = validate_feature_columns(df, exclude_columns)
    if not is_valid:
        for issue in issues:
            _logger.warning("Feature issue: %s", issue)
    
    # Log target statistics
    for col in target_columns:
        if col in df.columns:
            unique_values = df[col].value_counts().sort_index()
            _logger.info("Target %s distribution: %s", col, dict(unique_values))
    
    _logger.info("=== End Data Quality Report ===")
