"""
Synthetic Stock Returns Generator using Cholesky Decomposition

Generates synthetic stock return data with known covariance structure.
"""

import numpy as np


def generate_synthetic_returns(n_stocks=100, n_days=252, random_seed=42):
    """
    Generate synthetic stock returns using Cholesky decomposition.
    
    Uses Cholesky decomposition to transform uncorrelated normal data
    into correlated returns: Σ = L @ L.T, then X = Z @ L.T
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks
    n_days : int
        Number of trading days
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : np.ndarray, shape (n_days, n_stocks)
        Synthetic returns matrix
    true_cov : np.ndarray, shape (n_stocks, n_stocks)
        True covariance matrix (ground truth)
    metadata : dict
        Generation parameters
    """
    np.random.seed(random_seed)
    
    # Create random positive definite covariance matrix
    A = np.random.randn(n_stocks, n_stocks)
    true_cov = A @ A.T
    
    # Scale to realistic variance levels
    scale_factor = 0.0005 / np.mean(np.diag(true_cov))
    true_cov = true_cov * scale_factor
    
    # Cholesky decomposition: Σ = L @ L.T
    try:
        L = np.linalg.cholesky(true_cov)
    except np.linalg.LinAlgError:
        true_cov += 1e-6 * np.eye(n_stocks)
        L = np.linalg.cholesky(true_cov)
    
    # Generate uncorrelated data
    Z = np.random.randn(n_days, n_stocks)
    
    # Transform to correlated data
    X = Z @ L.T
    
    metadata = {
        'n_stocks': n_stocks,
        'n_days': n_days,
        'random_seed': random_seed,
        'mean_variance': np.mean(np.diag(true_cov)),
        'mean_correlation': np.mean(np.abs(true_cov - np.diag(np.diag(true_cov)))),
    }
    
    return X, true_cov, metadata
