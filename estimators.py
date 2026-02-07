"""
Covariance Matrix Estimators
"""

import numpy as np
from scipy import linalg


class Estimators:
    """
    A collection of covariance matrix estimation methods.
    
    All methods take a data matrix X of shape (n_samples, n_features) and return
    a covariance matrix of shape (n_features, n_features).
    """
    
    def sample_covariance(self, X):
        """
        Standard sample covariance estimator.
        
        Formula: S = (1/n) * X^T @ X (assuming X is centered)
        
        - Computes the maximum likelihood estimator of the covariance matrix

        """
        n_samples, n_features = X.shape
        
        # Center the data (subtract mean from each feature)]
        X_centered = X - np.mean(X, axis=0)
        
        # Compute covariance: (1/n) * X^T @ X
        # Each element (i,j) represents the covariance between features i and j
        cov = (X_centered.T @ X_centered) / n_samples
        
        return cov
    
    def ledoit_wolf(self, X):
        """
        Ledoit-Wolf shrinkage estimator.
        
        Formula: Σ_shrink = δ*F + (1-δ)*S
        where:
        - S is the sample covariance
        - F is the shrinkage target (scaled identity matrix)
        - δ is the optimal shrinkage intensity (computed analytically)

        """
        n_samples, n_features = X.shape
        
        # sample covariance
        X_centered = X - np.mean(X, axis=0)
        S = (X_centered.T @ X_centered) / n_samples
        
        # Step 2: Define shrinkage target F
        # We use a scaled identity matrix: F = μ * I
        # where μ is the average variance (trace of S divided by p)
        mu = np.trace(S) / n_features  # Average variance
        F = mu * np.eye(n_features)  # Target: uncorrelated assets with equal variance
        
        # Step 3: Compute optimal shrinkage intensity δ
        # This involves estimating the expected losses of S and F
        # The formula comes from minimizing E[||Σ_shrink - Σ_true||^2]
        
        # Compute delta (shrinkage intensity)
        # This is a simplified version of the Ledoit-Wolf formula
        delta_numerator = 0.0
        delta_denominator = 0.0
        
        # For each sample, compute contribution to shrinkage intensity
        for i in range(n_samples):
            x = X_centered[i:i+1, :].T  # Column vector
            
            # Contribution to numerator: ||x*x^T - S||^2
            sample_cov_i = x @ x.T
            delta_numerator += np.sum((sample_cov_i - S) ** 2)
        
        delta_numerator /= n_samples ** 2
        
        # Denominator: ||S - F||^2
        delta_denominator = np.sum((S - F) ** 2)
        
        # Optimal shrinkage intensity (capped between 0 and 1)
        if delta_denominator > 1e-10:
            delta = min(delta_numerator / delta_denominator, 1.0)
        else:
            delta = 1.0  # If S ≈ F, shrink completely to F
        
        delta = max(delta, 0.0)  # Ensure non-negative
        
        # Step 4: Compute shrunk covariance
        # Linear combination of target and sample covariance
        cov_shrunk = delta * F + (1 - delta) * S
        
        return cov_shrunk
    
    def tylers_m(self, X, tol=1e-6, max_iter=100):
        """
        Tyler's M-estimator of scatter.
        
        Robust covariance estimator using iterative fixed-point algorithm.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data matrix
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        M : np.ndarray, shape (n_features, n_features)
            Tyler's M-estimator
        """
        n_samples, n_features = X.shape
        X_centered = X - np.mean(X, axis=0)
        
        # Initialize with sample covariance
        M = self.sample_covariance(X)
        M += 1e-6 * np.eye(n_features)
        
        # Fixed-point iteration
        for iteration in range(max_iter):
            M_old = M.copy()
            
            try:
                L = linalg.cholesky(M, lower=True)
            except linalg.LinAlgError:
                M += 1e-4 * np.eye(n_features)
                L = linalg.cholesky(M, lower=True)
            
            M_new = np.zeros((n_features, n_features))
            
            for i in range(n_samples):
                x = X_centered[i, :]
                y = linalg.solve_triangular(L, x, lower=True)
                mahal_dist = np.dot(y, y)
                weight = 1.0 / (mahal_dist + 1e-10)
                M_new += weight * np.outer(x, x)
            
            M = (n_features / n_samples) * M_new
            
            change = np.linalg.norm(M - M_old, 'fro') / (np.linalg.norm(M_old, 'fro') + 1e-10)
            if change < tol:
                break
        
        return M
    
    def tylers_shrinkage(self, X, shrinkage=0.1, tol=1e-6, max_iter=100):
        """
        Tyler's M-estimator with shrinkage toward identity.
        
        Combines robustness of Tyler's M with regularization.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data matrix
        shrinkage : float
            Shrinkage intensity (0 to 1)
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        M_shrunk : np.ndarray, shape (n_features, n_features)
            Shrunk Tyler's M-estimator
        """
        n_features = X.shape[1]
        M = self.tylers_m(X, tol=tol, max_iter=max_iter)
        I = np.eye(n_features)
        M_shrunk = (1 - shrinkage) * M + shrinkage * I
        return M_shrunk
    
    def trex(self, X, threshold='auto'):
        """
        T-REX: Thresholded Random Effects eXpectation estimator.
        
        Sparse covariance estimator using thresholding.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data matrix
        threshold : float or 'auto'
            Threshold value for sparsification
            
        Returns:
        --------
        cov_sparse : np.ndarray, shape (n_features, n_features)
            Sparse covariance matrix
        """
        n_samples, n_features = X.shape
        
        # Start with Ledoit-Wolf as baseline
        S = self.ledoit_wolf(X)
        
        # Determine threshold
        if threshold == 'auto':
            threshold = np.sqrt(np.log(n_features) / n_samples)
            typical_cov = np.median(np.abs(S[np.triu_indices(n_features, k=1)]))
            threshold *= typical_cov
        
        # Apply soft thresholding to off-diagonal elements
        cov_sparse = S.copy()
        n = n_features
        for i in range(n):
            for j in range(i+1, n):
                if np.abs(cov_sparse[i, j]) < threshold:
                    cov_sparse[i, j] = 0.0
                    cov_sparse[j, i] = 0.0
                else:
                    sign = np.sign(cov_sparse[i, j])
                    cov_sparse[i, j] = sign * (np.abs(cov_sparse[i, j]) - threshold)
                    cov_sparse[j, i] = cov_sparse[i, j]
        
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(cov_sparse))
        if min_eig < 1e-6:
            cov_sparse += (1e-6 - min_eig) * np.eye(n_features)
        
        return cov_sparse
