#!/usr/bin/env python3
"""
Test the new eigenvector normalization function.
"""

import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

sys.path.append(str(Path(__file__).parent))

from exp_alcl.allo_complex import (
    compute_complex_cosine_similarities,
    compute_complex_cosine_similarities_with_normalization
)


def test_normalization():
    """Test the normalization function with simple synthetic data."""
    print("Testing eigenvector normalization...")

    # Create simple test data
    num_states = 10
    num_eigenvectors = 2

    # Create some random eigenvectors
    np.random.seed(42)

    # Right eigenvectors (complex)
    right_real = np.random.randn(num_states, num_eigenvectors)
    right_imag = np.random.randn(num_states, num_eigenvectors)

    # Left eigenvectors (complex) - make them similar to right for testing
    left_real = np.random.randn(num_states, num_eigenvectors)
    left_imag = np.random.randn(num_states, num_eigenvectors)

    # Sampling probabilities (uniform distribution for simplicity)
    sampling_probs = np.ones(num_states) / num_states

    # Convert to JAX arrays
    right_real = jnp.array(right_real)
    right_imag = jnp.array(right_imag)
    left_real = jnp.array(left_real)
    left_imag = jnp.array(left_imag)
    sampling_probs = jnp.array(sampling_probs)

    print(f"\nInput shapes:")
    print(f"  Right eigenvectors: {right_real.shape}")
    print(f"  Left eigenvectors: {left_real.shape}")
    print(f"  Sampling probs: {sampling_probs.shape}")

    # Test the normalization function
    # Compare learned vs learned (should give perfect similarity)
    results = compute_complex_cosine_similarities_with_normalization(
        learned_left_real=left_real,
        learned_left_imag=left_imag,
        learned_right_real=right_real,
        learned_right_imag=right_imag,
        gt_left_real=left_real,
        gt_left_imag=left_imag,
        gt_right_real=right_real,
        gt_right_imag=right_imag,
        sampling_probs=sampling_probs
    )

    print(f"\nSelf-comparison results (should be close to 1.0):")
    print(f"  Left similarity avg: {results['left_cosine_sim_avg']:.6f}")
    print(f"  Right similarity avg: {results['right_cosine_sim_avg']:.6f}")

    for i in range(num_eigenvectors):
        left_sim = results[f'left_cosine_sim_{i}']
        right_sim = results[f'right_cosine_sim_{i}']
        print(f"  Eigenvector {i}: left={left_sim:.6f}, right={right_sim:.6f}")

    # Test with scaled versions (arbitrary complex scaling)
    # This tests if the normalization properly handles the scaling freedom
    scale_factor_real = 2.5
    scale_factor_imag = 1.3

    # Scale the eigenvectors by a complex number
    scaled_right_real = scale_factor_real * right_real - scale_factor_imag * right_imag
    scaled_right_imag = scale_factor_real * right_imag + scale_factor_imag * right_real
    scaled_left_real = scale_factor_real * left_real - scale_factor_imag * left_imag
    scaled_left_imag = scale_factor_real * left_imag + scale_factor_imag * left_real

    results_scaled = compute_complex_cosine_similarities_with_normalization(
        learned_left_real=scaled_left_real,
        learned_left_imag=scaled_left_imag,
        learned_right_real=scaled_right_real,
        learned_right_imag=scaled_right_imag,
        gt_left_real=left_real,
        gt_left_imag=left_imag,
        gt_right_real=right_real,
        gt_right_imag=right_imag,
        sampling_probs=sampling_probs
    )

    print(f"\nScaled version comparison (should still be close to 1.0):")
    print(f"  Left similarity avg: {results_scaled['left_cosine_sim_avg']:.6f}")
    print(f"  Right similarity avg: {results_scaled['right_cosine_sim_avg']:.6f}")

    for i in range(num_eigenvectors):
        left_sim = results_scaled[f'left_cosine_sim_{i}']
        right_sim = results_scaled[f'right_cosine_sim_{i}']
        print(f"  Eigenvector {i}: left={left_sim:.6f}, right={right_sim:.6f}")

    print("\n✓ Test completed successfully!")

    # Check that similarities are high (> 0.99 for self-comparison)
    assert results['left_cosine_sim_avg'] > 0.99, "Self-comparison should give ~1.0"
    assert results['right_cosine_sim_avg'] > 0.99, "Self-comparison should give ~1.0"

    # Scaled versions should also match well after normalization
    assert results_scaled['left_cosine_sim_avg'] > 0.99, "Scaled comparison should still give ~1.0"
    assert results_scaled['right_cosine_sim_avg'] > 0.99, "Scaled comparison should still give ~1.0"

    print("✓ All assertions passed!")


if __name__ == "__main__":
    test_normalization()
