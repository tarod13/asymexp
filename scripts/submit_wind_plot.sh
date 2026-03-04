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
#                                    --output_dir  ./figures \
#                                    --num_eigvecs 4
#
# Produces (all in --output_dir):
#   gt_right_real.png   gt_left_real.png   allo_right_real.png
#   gt_right_imag.png   gt_left_imag.png
#   gt_right_abs.png    gt_left_abs.png

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python scripts/plot_wind_sweep.py "$@"
