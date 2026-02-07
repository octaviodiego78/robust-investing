"""
Evaluation and Benchmarking of Covariance Estimators

Provides tools to evaluate and compare covariance estimators using
condition number and Frobenius error norm.
"""

import numpy as np
import pandas as pd
import time
from estimators import Estimators
from dataset_generator import generate_synthetic_returns


def compute_condition_number(cov_matrix):
    """
    Compute condition number of covariance matrix.
    
    Condition number = λ_max / λ_min
    Lower is better (measures numerical stability).
    
    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix
        
    Returns:
    --------
    condition_number : float
        Ratio of largest to smallest eigenvalue
    """
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    condition_number = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)
    return condition_number


def compute_frobenius_error(estimated_cov, true_cov):
    """
    Compute Frobenius norm of estimation error.
    
    Frobenius error = ||Estimated - True||_F
    Lower is better (measures estimation accuracy).
    
    Parameters:
    -----------
    estimated_cov : np.ndarray
        Estimated covariance matrix
    true_cov : np.ndarray
        True covariance matrix
        
    Returns:
    --------
    error : float
        Frobenius norm of difference
    """
    diff = estimated_cov - true_cov
    error = np.linalg.norm(diff, 'fro')
    return error


def benchmark_estimators(X, true_cov, estimators_obj):
    """
    Run all estimators and collect performance metrics.
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    true_cov : np.ndarray
        True covariance matrix
    estimators_obj : Estimators
        Instance of Estimators class
        
    Returns:
    --------
    results : pd.DataFrame
        Results with columns: estimator_name, condition_number, 
        frobenius_error, relative_error, computation_time
    """
    results = []
    
    estimator_configs = [
        ('Sample Covariance', estimators_obj.sample_covariance, {}),
        ('Ledoit-Wolf', estimators_obj.ledoit_wolf, {}),
        ("Tyler's M", estimators_obj.tylers_m, {}),
        ("Tyler's + Shrinkage", estimators_obj.tylers_shrinkage, {'shrinkage': 0.1}),
        ('T-REX', estimators_obj.trex, {'threshold': 'auto'}),
    ]
    
    for name, method, kwargs in estimator_configs:
        start_time = time.time()
        
        try:
            estimated_cov = method(X, **kwargs)
            computation_time = time.time() - start_time
            
            cond_num = compute_condition_number(estimated_cov)
            frob_error = compute_frobenius_error(estimated_cov, true_cov)
            relative_error = frob_error / np.linalg.norm(true_cov, 'fro')
            
            results.append({
                'estimator_name': name,
                'condition_number': cond_num,
                'frobenius_error': frob_error,
                'relative_error': relative_error,
                'computation_time': computation_time,
            })
            
        except Exception as e:
            results.append({
                'estimator_name': name,
                'condition_number': np.nan,
                'frobenius_error': np.nan,
                'relative_error': np.nan,
                'computation_time': np.nan,
            })
    
    results_df = pd.DataFrame(results)
    return results_df


def main():
    """
    Run complete evaluation pipeline.
    """

    print("Covariance Estimator Evaluation")
    
    # Generate synthetic data
    n_stocks = 100
    n_days = 252
    
    print(f"\nGenerating data: {n_stocks} stocks, {n_days} days (p/n = {n_stocks/n_days:.2f})")
    X, true_cov, metadata = generate_synthetic_returns(
        n_stocks=n_stocks,
        n_days=n_days,
        random_seed=42
    )
    
    # Initialize estimators
    est = Estimators()
    
    # Run benchmarks
    print("\nRunning benchmarks...\n")
    results = benchmark_estimators(X, true_cov, est)
    
    # Display results
    print("Results")
    print(results.to_string(index=False))
    
    # Best estimators
    print("\nBest Estimators")
    best_cond = results.loc[results['condition_number'].idxmin()]
    best_error = results.loc[results['frobenius_error'].idxmin()]
    fastest = results.loc[results['computation_time'].idxmin()]
    
    print(f"Best Condition Number: {best_cond['estimator_name']} ({best_cond['condition_number']:.2f})")
    print(f"Lowest Frobenius Error: {best_error['estimator_name']} ({best_error['frobenius_error']:.4f})")
    print(f"Fastest: {fastest['estimator_name']} ({fastest['computation_time']:.4f}s)")
    
    # Save results
    results.to_csv('benchmark_results.csv', index=False)
    print(f"\n✓ Results saved to benchmark_results.csv")


if __name__ == "__main__":
    main()
