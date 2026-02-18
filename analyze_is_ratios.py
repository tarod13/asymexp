#!/usr/bin/env python3
"""
Analyze importance sampling ratios from your training data.
Helps determine optimal clipping thresholds.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def analyze_is_ratios(results_dir):
    """Analyze IS ratios from saved sampling distribution."""
    results_path = Path(results_dir)

    # Load sampling distribution
    sampling_dist_path = results_path / "sampling_distribution.npy"
    if not sampling_dist_path.exists():
        print(f"Error: {sampling_dist_path} not found")
        return

    sampling_probs = np.load(sampling_dist_path)
    num_states = len(sampling_probs)

    print(f"\nSampling Distribution Analysis")
    print(f"=" * 50)
    print(f"Number of states: {num_states}")
    print(f"Sampling prob range: [{sampling_probs.min():.6f}, {sampling_probs.max():.6f}]")
    print(f"Sampling prob mean: {sampling_probs.mean():.6f}")
    print(f"Sampling prob std: {sampling_probs.std():.6f}")

    # Compute IS ratios (unnormalized)
    is_ratio_unnorm = 1.0 / (sampling_probs * num_states)

    print(f"\nIS Ratio (Unnormalized)")
    print(f"-" * 50)
    print(f"Range: [{is_ratio_unnorm.min():.3f}, {is_ratio_unnorm.max():.3f}]")
    print(f"Mean: {is_ratio_unnorm.mean():.3f}")
    print(f"Std: {is_ratio_unnorm.std():.3f}")
    print(f"Ratio max/min: {is_ratio_unnorm.max() / is_ratio_unnorm.min():.1f}x")

    # Compute IS ratios (normalized)
    is_ratio_norm = is_ratio_unnorm / is_ratio_unnorm.mean()

    print(f"\nIS Ratio (Normalized to mean=1)")
    print(f"-" * 50)
    print(f"Range: [{is_ratio_norm.min():.3f}, {is_ratio_norm.max():.3f}]")
    print(f"Mean: {is_ratio_norm.mean():.3f}")
    print(f"Std: {is_ratio_norm.std():.3f}")

    # Effective sample size
    ess = 1.0 / np.sum(sampling_probs ** 2)
    print(f"\nEffective Sample Size: {ess:.1f} (out of {num_states})")
    print(f"ESS ratio: {ess / num_states:.2%}")

    # Percentiles
    print(f"\nIS Ratio Percentiles (normalized):")
    print(f"-" * 50)
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(is_ratio_norm, p)
        print(f"  {p:2d}th percentile: {val:.3f}")

    # Clipping analysis
    print(f"\n\nClipping Threshold Analysis")
    print(f"=" * 50)

    for max_clip in [3.0, 5.0, 10.0, 20.0]:
        min_clip = 1.0 / max_clip
        clipped = np.clip(is_ratio_norm, min_clip, max_clip)

        n_clipped_high = np.sum(is_ratio_norm > max_clip)
        n_clipped_low = np.sum(is_ratio_norm < min_clip)
        pct_clipped = 100 * (n_clipped_high + n_clipped_low) / num_states

        print(f"\nClip range: [{min_clip:.3f}, {max_clip:.1f}]")
        print(f"  States clipped (high): {n_clipped_high} ({100*n_clipped_high/num_states:.1f}%)")
        print(f"  States clipped (low): {n_clipped_low} ({100*n_clipped_low/num_states:.1f}%)")
        print(f"  Total clipped: {pct_clipped:.1f}%")
        print(f"  Std after clipping: {clipped.std():.3f} (before: {is_ratio_norm.std():.3f})")

    # Recommendations
    print(f"\n\nRecommendations")
    print(f"=" * 50)

    if is_ratio_norm.max() / is_ratio_norm.min() > 50:
        print("⚠️  SEVERE variance (>50x range)")
        print("   → Use aggressive clipping: is_ratio_max=5.0")
    elif is_ratio_norm.max() / is_ratio_norm.min() > 20:
        print("⚠️  HIGH variance (>20x range)")
        print("   → Use moderate clipping: is_ratio_max=10.0")
    else:
        print("✓  Moderate variance")
        print("   → Current settings (max=10.0) should work well")

    if ess / num_states < 0.3:
        print("\n⚠️  Low effective sample size (<30% of states)")
        print("   → Consider more uniform sampling during data collection")

    # Create visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Sampling probability histogram
        axes[0, 0].hist(sampling_probs, bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Sampling Probability')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Sampling Probabilities')
        axes[0, 0].axvline(sampling_probs.mean(), color='r', linestyle='--', label=f'Mean: {sampling_probs.mean():.4f}')
        axes[0, 0].legend()

        # 2. IS ratio histogram (normalized)
        axes[0, 1].hist(is_ratio_norm, bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('IS Ratio (normalized)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of IS Ratios')
        axes[0, 1].axvline(1.0, color='r', linestyle='--', label='Mean: 1.0')
        axes[0, 1].legend()

        # 3. IS ratio vs state index (sorted)
        sorted_idx = np.argsort(is_ratio_norm)
        axes[1, 0].plot(is_ratio_norm[sorted_idx], linewidth=2)
        axes[1, 0].axhline(10.0, color='r', linestyle='--', label='Default max clip (10.0)')
        axes[1, 0].axhline(0.1, color='r', linestyle='--', label='Default min clip (0.1)')
        axes[1, 0].set_xlabel('State (sorted by IS ratio)')
        axes[1, 0].set_ylabel('IS Ratio (normalized)')
        axes[1, 0].set_title('IS Ratios Sorted by Magnitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Log-scale histogram
        axes[1, 1].hist(np.log10(is_ratio_norm), bins=50, edgecolor='black')
        axes[1, 1].set_xlabel('log10(IS Ratio)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Log-scale Distribution of IS Ratios')
        axes[1, 1].axvline(0, color='r', linestyle='--', label='log10(1.0) = 0')
        axes[1, 1].legend()

        plt.tight_layout()

        output_path = results_path / "is_ratio_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to: {output_path}")

    except Exception as e:
        print(f"\nWarning: Could not create visualization: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_is_ratios.py <results_dir>")
        print("Example: python analyze_is_ratios.py ./results/room4/room4__allo__0__42__1234567890")
        sys.exit(1)

    analyze_is_ratios(sys.argv[1])
