# Complex Eigenvector Handler (allo_complex.py)

This file extends the `allo.py` implementation to handle **complex eigenvectors** for non-symmetric Laplacians.

## Key Differences from allo.py

### 1. Network Architecture
- **Output**: 4 sets of features instead of 1:
  - `left_real`: Left eigenvectors, real part (φ_real)
  - `left_imag`: Left eigenvectors, imaginary part (φ_imag)
  - `right_real`: Right eigenvectors, real part (ψ_real)
  - `right_imag`: Right eigenvectors, imaginary part (ψ_imag)
- **Total outputs**: `4 × num_features` (instead of `num_features`)

### 2. Loss Function Components

#### Graph Drawing Loss
For complex eigenvectors, the graph drawing loss is:
```
L_graph = E[(φ_real(s) - ψ_real(s'))² + (φ_imag(s) - ψ_imag(s'))²]
```

This implements `Re(φ^T L ψ)` via Monte Carlo sampling, where:
- φ represents left eigenvectors
- ψ represents right eigenvectors
- L is encoded implicitly through the transition dynamics

#### Biorthogonality Constraints
Complex eigenvectors satisfy `φ^T ψ = I`, which decomposes into:

1. **Real part constraint**: `Re(φ^T ψ) = I`
   ```
   (1/N) Σ [φ_real^T ψ_real - φ_imag^T ψ_imag] = I
   ```

2. **Imaginary part constraint**: `Im(φ^T ψ) = 0`
   ```
   (1/N) Σ [φ_real^T ψ_imag + φ_imag^T ψ_real] = 0
   ```

### 3. Mathematical Formulation

For a non-symmetric matrix L with complex eigenvectors:
- Left eigenvectors: u = u_real + i·u_imag (satisfy u^T L = λ u^T)
- Right eigenvectors: v = v_real + i·v_imag (satisfy L v = λ v)
- Biorthogonality: u_i^T v_j = δ_ij

The objective minimizes:
```
Σ_i [u_{i,real}^T L v_{i,real} + u_{i,imag}^T L v_{i,imag}]
```

subject to biorthogonality constraints.

### 4. Training Details

The loss function includes:
1. **Graph drawing loss**: Couples left and right eigenvectors through transition dynamics
2. **Dual variables**: Enforce real part of biorthogonality constraint
3. **Barrier loss**: Quadratic penalty for constraint violations
4. **Imaginary part penalty**: Ensures Im(φ^T ψ) = 0

### 5. Saved Outputs

The training saves 4 separate files for learned eigenvectors:
- `final_learned_left_real.npy`
- `final_learned_left_imag.npy`
- `final_learned_right_real.npy`
- `final_learned_right_imag.npy`

## Usage

Run the same way as `allo.py`:
```bash
python exp_alcl/allo_complex.py --env_type room4 --num_eigenvectors 10
```

All command-line arguments from `allo.py` are supported.

## Implementation Notes

1. **Shared backbone**: All 4 output heads share the same hidden layers
2. **Stop gradient**: Biorthogonality constraints use stop_gradient on one side to prevent collapse
3. **Visualization**: Uses right eigenvectors (real parts) for comparison with ground truth
4. **Cosine similarity**: Compares learned right_real with ground truth real eigenvectors

## Mathematical Background

This implementation handles the general case where the Laplacian matrix is **non-symmetric**, which arises in:
- Directed graphs
- Non-reversible Markov chains
- Asymmetric successor representations

For symmetric matrices, left and right eigenvectors coincide, and imaginary parts vanish.
