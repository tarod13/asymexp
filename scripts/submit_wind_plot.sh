#!/bin/bash
# Run the wind-sweep eigenvector visualisation.
#
# Usage
# -----
#   bash scripts/submit_wind_plot.sh [options forwarded to plot_wind_sweep.py]
#
# Examples
# --------
#   bash scripts/submit_wind_plot.sh
#   bash scripts/submit_wind_plot.sh --results_dir ./results/wind_sweep \
#                                    --output figures/wind_eigvecs.png \
#                                    --num_eigvecs 4 \
#                                    --component real \
#                                    --heatmap_winds -0.99 -0.5 0 0.5 0.99

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python scripts/plot_wind_sweep.py "$@"
