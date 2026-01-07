"""
Detailed diagnostic script to understand what the replay buffer sampling actually recovers.

This will help us understand if:
1. The sampling recovers P (1-step transitions)
2. The sampling recovers P*SR_γ (as intended)
3. The sampling recovers something else

We'll compare the empirical transition matrix with:
- Ground truth P
- Ground truth P*SR_γ
- Ground truth (1-γ)*P*SR_γ
"""

import sys
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import matplotlib.pyplot as plt

# Load the saved results
results_dir = Path("./sampling_test_results")

print("="*80)
print("DETAILED SAMPLING DIAGNOSTICS")
print("="*80)

# Load matrices
transition_counts = np.load(results_dir / "transition_counts.npy")
empirical_transition_matrix = np.load(results_dir / "empirical_transition_matrix.npy")
empirical_laplacian = np.load(results_dir / "empirical_laplacian.npy")
ground_truth_laplacian = np.load(results_dir / "ground_truth_laplacian.npy")

num_states = empirical_transition_matrix.shape[0]
gamma = 0.2  # From the test

print(f"\nNumber of states: {num_states}")
print(f"Discount factor γ: {gamma}")

# Reconstruct ground truth P from Laplacian
# L = I - (1-γ)P*SR_γ
# So: P*SR_γ = (I - L) / (1-γ)
ground_truth_P_SR = (np.eye(num_states) - ground_truth_laplacian) / (1 - gamma)

print("\n" + "="*80)
print("1. CHECKING ROW SUMS")
print("="*80)

emp_row_sums = empirical_transition_matrix.sum(axis=1)
gt_row_sums = ground_truth_P_SR.sum(axis=1)

print(f"\nEmpirical transition matrix:")
print(f"  Row sums - min: {emp_row_sums.min():.6f}, max: {emp_row_sums.max():.6f}, mean: {emp_row_sums.mean():.6f}")
print(f"  Expected: all rows should sum to 1.0 if it represents a stochastic matrix")

print(f"\nGround truth P*SR_γ:")
print(f"  Row sums - min: {gt_row_sums.min():.6f}, max: {gt_row_sums.max():.6f}, mean: {gt_row_sums.mean():.6f}")
print(f"  Expected: rows sum to 1/(1-γ) = {1/(1-gamma):.6f} if SR_γ is the successor representation")

# Scale ground truth to match empirical (for comparison)
gt_P_SR_normalized = ground_truth_P_SR / gt_row_sums[:, None]

print("\n" + "="*80)
print("2. COMPARING EMPIRICAL vs GROUND TRUTH")
print("="*80)

# Compute element-wise differences
diff_unnormalized = np.abs(empirical_transition_matrix - ground_truth_P_SR)
diff_normalized = np.abs(empirical_transition_matrix - gt_P_SR_normalized)

print(f"\nElement-wise L1 difference:")
print(f"  Empirical vs P*SR_γ (unnormalized): {diff_unnormalized.sum():.6f}")
print(f"  Empirical vs P*SR_γ (normalized):   {diff_normalized.sum():.6f}")

# Frobenius norm
print(f"\nFrobenius norm difference:")
print(f"  Empirical vs P*SR_γ (unnormalized): {np.linalg.norm(diff_unnormalized, 'fro'):.6f}")
print(f"  Empirical vs P*SR_γ (normalized):   {np.linalg.norm(diff_normalized, 'fro'):.6f}")

# Check a few specific states
print("\n" + "="*80)
print("3. EXAMINING SPECIFIC STATE TRANSITIONS")
print("="*80)

# Find states with most transitions
state_visit_counts = transition_counts.sum(axis=1)
most_visited_states = np.argsort(-state_visit_counts)[:5]

print(f"\nTop 5 most visited states (as source):")
for idx, state in enumerate(most_visited_states):
    visit_count = state_visit_counts[state]
    emp_dist = empirical_transition_matrix[state]
    gt_dist = ground_truth_P_SR[state]
    gt_dist_norm = gt_P_SR_normalized[state]

    # Find non-zero transitions
    nonzero_idx = np.where(emp_dist > 0.001)[0]

    print(f"\nState {state} (visited {int(visit_count)} times as source):")
    print(f"  Number of non-zero transitions: {len(nonzero_idx)}")
    print(f"  Row sum - empirical: {emp_dist.sum():.6f}, ground truth: {gt_dist.sum():.6f}")

    if len(nonzero_idx) > 0:
        # Show top 3 most likely next states
        top_next_states = np.argsort(-emp_dist)[:3]
        print(f"  Top 3 next states:")
        for next_state in top_next_states:
            print(f"    -> {next_state}: emp={emp_dist[next_state]:.6f}, gt={gt_dist[next_state]:.6f}, gt_norm={gt_dist_norm[next_state]:.6f}")

print("\n" + "="*80)
print("4. CHECKING WHAT EMPIRICAL MATRIX REPRESENTS")
print("="*80)

# Hypothesis 1: Empirical ≈ P (1-step transitions)
# We need to reconstruct P from the ground truth
# From L = I - (1-γ)P*SR_γ, and SR_γ = (I - γP)^{-1}, we can't directly extract P
# But we can check if empirical looks like a 1-step transition matrix

# Hypothesis 2: Empirical ≈ (1-γ)*P*SR_γ
scaled_gt = (1 - gamma) * ground_truth_P_SR
diff_scaled = np.abs(empirical_transition_matrix - scaled_gt)

print(f"\nHypothesis: Empirical ≈ (1-γ)*P*SR_γ")
print(f"  (1-γ)*P*SR_γ row sums - min: {scaled_gt.sum(axis=1).min():.6f}, max: {scaled_gt.sum(axis=1).max():.6f}")
print(f"  L1 difference: {diff_scaled.sum():.6f}")
print(f"  Frobenius norm: {np.linalg.norm(diff_scaled, 'fro'):.6f}")
print(f"  This hypothesis is: {'LIKELY CORRECT' if np.linalg.norm(diff_scaled, 'fro') < 1.0 else 'UNLIKELY'}")

# Hypothesis 3: Check if normalization is the issue
# Maybe we should not normalize the counts
print(f"\n" + "="*80)
print("5. CHECKING IF NORMALIZATION IS THE ISSUE")
print("="*80)

# Reconstruct empirical without normalization (just scale by total samples)
total_samples = transition_counts.sum()
empirical_unnormalized = transition_counts / total_samples

print(f"\nTotal samples: {int(total_samples)}")
print(f"Unnormalized empirical matrix:")
print(f"  Row sums - min: {empirical_unnormalized.sum(axis=1).min():.6f}, max: {empirical_unnormalized.sum(axis=1).max():.6f}")
print(f"  Total sum: {empirical_unnormalized.sum():.6f} (should be 1.0)")

# Compare with stationary distribution weighted version
# The true distribution we're sampling from is:
# P(s, s') = π(s) * P(s'|s)
# where π is the stationary distribution and P(s'|s) is what we want

# Create a plot comparing matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Show empirical matrix
im0 = axes[0, 0].imshow(empirical_transition_matrix, cmap='viridis', aspect='auto', vmin=0)
axes[0, 0].set_title('Empirical (normalized per row)')
axes[0, 0].set_xlabel('Next State')
axes[0, 0].set_ylabel('Current State')
plt.colorbar(im0, ax=axes[0, 0])

# Show P*SR_γ
im1 = axes[0, 1].imshow(ground_truth_P_SR, cmap='viridis', aspect='auto', vmin=0)
axes[0, 1].set_title('Ground Truth P*SR_γ')
axes[0, 1].set_xlabel('Next State')
axes[0, 1].set_ylabel('Current State')
plt.colorbar(im1, ax=axes[0, 1])

# Show (1-γ)*P*SR_γ
im2 = axes[0, 2].imshow(scaled_gt, cmap='viridis', aspect='auto', vmin=0)
axes[0, 2].set_title(f'(1-γ)*P*SR_γ = {1-gamma:.1f}*P*SR_γ')
axes[0, 2].set_xlabel('Next State')
axes[0, 2].set_ylabel('Current State')
plt.colorbar(im2, ax=axes[0, 2])

# Show differences
im3 = axes[1, 0].imshow(diff_unnormalized, cmap='hot', aspect='auto')
axes[1, 0].set_title('|Empirical - P*SR_γ|')
axes[1, 0].set_xlabel('Next State')
axes[1, 0].set_ylabel('Current State')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(diff_normalized, cmap='hot', aspect='auto')
axes[1, 1].set_title('|Empirical - P*SR_γ (normalized)|')
axes[1, 1].set_xlabel('Next State')
axes[1, 1].set_ylabel('Current State')
plt.colorbar(im4, ax=axes[1, 1])

im5 = axes[1, 2].imshow(diff_scaled, cmap='hot', aspect='auto')
axes[1, 2].set_title(f'|Empirical - (1-γ)*P*SR_γ|')
axes[1, 2].set_xlabel('Next State')
axes[1, 2].set_ylabel('Current State')
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.savefig(results_dir / "detailed_matrix_comparison.png", dpi=150, bbox_inches='tight')
print(f"\nSaved detailed comparison to {results_dir / 'detailed_matrix_comparison.png'}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
