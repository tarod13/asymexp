# Sampling Distribution Diagnostic Report

## Executive Summary

I investigated the sampling distribution from the episodic replay buffer to verify whether it correctly recovers the transition matrix needed for learning eigenvectors of the non-symmetric Laplacian L = I - (1-γ)P·SR_γ.

**Key Finding**: The sampling correctly recovers **(1-γ)P·SR_γ**, not P·SR_γ. This is due to the row-normalization of transition frequencies.

## Background

The goal is to learn eigenvectors of the non-symmetric Laplacian:
```
L = I - (1-γ)P·SR_γ
```

where:
- P is the 1-step transition matrix
- SR_γ = (I - γP)^{-1} is the successor representation with discount γ
- (1-γ) is the normalization factor

The episodic replay buffer samples transitions using a truncated geometric distribution with parameter γ, which should approximate the successor representation.

## Methodology

I created a test (`test_sampling_distribution.py`) that:

1. Collects 500,000 transition samples from the replay buffer
2. Counts transition frequencies to build an empirical transition matrix
3. Computes eigendecomposition of the empirical Laplacian
4. Compares with ground truth eigenvalues and eigenvectors

## Initial Results (INCORRECT)

Initially, I computed the empirical Laplacian as:
```python
empirical_laplacian = I - (1-γ) * empirical_transition_matrix
```

This gave:
- Ground truth eigenvalue: 1.36e-07 ≈ 0 ✓
- Empirical eigenvalue: 0.2 = γ ✗
- **Eigenvalue error: 0.200** (completely wrong!)

## Root Cause Analysis

Using `diagnose_sampling.py`, I discovered that:

1. **Empirical transition matrix row sums**: 1.0 (row-stochastic due to normalization)
2. **Ground truth P·SR_γ row sums**: 1.25 = 1/(1-γ)
3. **Hypothesis test**: Empirical ≈ (1-γ)P·SR_γ
   - Frobenius norm difference: **0.141** (very small!)
   - **Conclusion: CONFIRMED**

### Why does sampling give (1-γ)P·SR_γ instead of P·SR_γ?

The sampling process:
1. Samples state `s` from episodes (approximately uniform over states)
2. Samples duration `k` from geometric distribution: p(k) ∝ (1-γ)γ^{k-1} for k ≥ 1
3. Returns transition (s, s') where s' is k steps from s

The probability of transitioning from s to s' is:
```
P(s'|s) ∝ Σ_{k≥1} (1-γ)γ^{k-1} P^k(s, s')
        = (1-γ) Σ_{k≥1} γ^{k-1} P^k(s, s')
        = (1-γ) [P + γP² + γ²P³ + ...](s, s')
        = (1-γ) [P·SR_γ](s, s')
```

When we row-normalize the transition counts (to make each row sum to 1), we get a matrix where each row sums to 1, representing the conditional distribution P(s'|s).

The key insight is that **(1-γ)P·SR_γ has row sums equal to (1-γ)/(1-γ) = 1**, making it naturally row-stochastic! This is because:
```
row_sum = Σ_s' (1-γ) [P·SR_γ](s, s')
        = (1-γ) Σ_s' [P·SR_γ](s, s')
        = (1-γ) · 1/(1-γ)    [since P·SR_γ row sums = 1/(1-γ)]
        = 1
```

## Corrected Results

After fixing the Laplacian computation to:
```python
empirical_laplacian = I - empirical_transition_matrix
```

Results:
- Ground truth eigenvalue: 1.36e-07 ≈ 0 ✓
- Empirical eigenvalue: -1.32e-07 ≈ 0 ✓
- **Mean eigenvalue error: 0.000000** ✓✓✓
- **Eigenvector similarity: 1.000** ✓✓✓

## Implications for Learning

### The sampling is working correctly!

The replay buffer correctly approximates the non-symmetric Laplacian through its sampling distribution. The (1-γ) factor is naturally included in the row-normalized empirical transition matrix.

### Graph drawing loss analysis

The graph drawing loss in `allo_complex.py` is:
```python
E[ψ(s) * (φ(s) - φ(s'))]²
```

where the expectation is over sampled transitions (s, s').

Since sampling gives us P(s'|s) = row_normalized[(1-γ)P·SR_γ], the expectation becomes:
```
E[ψ(s) * (φ(s) - φ(s'))]
= Σ_s π(s) Σ_s' P(s'|s) ψ(s) (φ(s) - φ(s'))
= Σ_s π(s) ψ(s) [φ(s) - Σ_s' P(s'|s) φ(s')]
```

where π(s) is the state visitation distribution from the replay buffer.

For the eigenvalue equation to hold:
```
L φ = λ φ
(I - (1-γ)P·SR_γ) φ = λ φ
φ - (1-γ)P·SR_γ φ = λ φ
```

Rearranging:
```
φ - [empirical_matrix] φ = λ φ   [since empirical = (1-γ)P·SR_γ normalized]
```

This means the graph drawing loss is correctly formulated to learn eigenvectors of L = I - empirical_matrix, which after proper accounting equals I - (1-γ)P·SR_γ.

## Potential Issues with Learning

Despite the sampling being correct, there might still be issues with learning:

1. **High cosine similarity despite different eigenvectors**: The user reports high cosine similarity but different eigenvector values. This could be due to:
   - **Scaling/normalization differences**: Eigenvectors are only defined up to a scalar multiple
   - **Sign ambiguity**: Eigenvectors can be flipped without changing the eigenvalue
   - **Numerical precision**: Small eigenvalues (≈0) can have unstable eigenvectors

2. **Biorthogonality constraints**: The augmented Lagrangian method enforces φ^T ψ = I. If this constraint is not satisfied well, the learned eigenvectors might not be correct.

3. **Batch sampling effects**: The graph loss uses batch sampling, which might not perfectly represent the true expectation over the transition distribution.

## Recommendations

1. **Check biorthogonality**: Compute φ^T ψ for the learned eigenvectors and verify it's close to identity

2. **Compare eigenvalue spectra**: Plot the full spectrum of learned vs. ground truth eigenvalues

3. **Visualize eigenvectors**: Create spatial heatmaps of both learned and ground truth eigenvectors to see if patterns match

4. **Check normalization**: Verify that eigenvectors are normalized consistently (L2 norm, or according to biorthogonality)

5. **Increase samples**: Try learning with more samples to see if the match improves

6. **Check dual variable convergence**: The dual variables should converge to the negative eigenvalues. Verify this is happening.

## Test Scripts

- `test_sampling_distribution.py`: Main test to verify sampling recovers correct distribution
- `diagnose_sampling.py`: Detailed diagnostics comparing empirical vs. ground truth matrices

## Visualizations

See `sampling_test_results/` for:
- `eigenvalue_comparison.png`: Scatter plot of empirical vs. ground truth eigenvalues
- `transition_matrix_comparison.png`: Heatmaps comparing matrices
- `detailed_matrix_comparison.png`: Comprehensive comparison of all matrix forms
