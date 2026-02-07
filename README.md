# Robust Covariance Matrix Estimation

Implementation and comparison of covariance matrix estimators for portfolio optimization.

## Project Structure

```
robust-investing/
├── requirements.txt          # Python dependencies
├── estimators.py            # Covariance estimator implementations
├── dataset_generator.py     # Synthetic data generation using Cholesky
├── evaluation.py            # Benchmarking and metrics
└── report.ipynb            # Comprehensive analysis and documentation
```

## Estimators

1. **Sample Covariance** - Standard MLE estimator
2. **Ledoit-Wolf** - Shrinkage toward identity
3. **Tyler's M** - Robust M-estimator
4. **Tyler's + Shrinkage** - Tyler's M with regularization
5. **T-REX** - Sparse thresholded estimator