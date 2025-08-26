"""
Correlation Check Module

Implements correlation-based position limits.
"""
import numpy as np

def check_correlation_limit(correlation_matrix: np.ndarray, threshold: float) -> bool:
    """
    Check if any pairwise correlation exceeds the threshold.
    Args:
        correlation_matrix (np.ndarray): Correlation matrix of asset returns
        threshold (float): Maximum allowed absolute correlation
    Returns:
        bool: True if all correlations are within limit, False if any exceed
    """
    n = correlation_matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if abs(correlation_matrix[i, j]) > threshold:
                return False
    return True 
