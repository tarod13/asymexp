#!/bin/bash
#
# run_analysis.sh - Run analysis for asymexp experiments
#
# Usage:
#   ./run_analysis.sh [experiment_type] [options...]
#
# Experiment types:
#   portal    - Eigendecomposition analysis on portal environments (experiment 01)
#   door      - Eigendecomposition analysis on door environments (experiment 02)
#   sweep     - Batch size × learning rate sweep analysis
#
# Examples:
#   ./run_analysis.sh portal --num-envs 20 --k 30
#   ./run_analysis.sh door --num-doors 10 --num-envs 50
#   ./run_analysis.sh sweep --exp_name batch_lr_sweep

set -e  # Exit on error

# Default values
EXPERIMENT_TYPE=""
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo -e "${GREEN}Usage:${NC}"
    echo "  $0 [experiment_type] [options...]"
    echo ""
    echo -e "${GREEN}Experiment types:${NC}"
    echo "  portal    - Eigendecomposition analysis on portal environments (experiment 01)"
    echo "  door      - Eigendecomposition analysis on door environments (experiment 02)"
    echo "  sweep     - Batch size × learning rate sweep analysis"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  $0 portal --num-envs 20 --k 30 --output-dir ./results/portal_20envs"
    echo "  $0 door --num-doors 10 --num-envs 50"
    echo "  $0 sweep --exp_name batch_lr_sweep --results_dir ./results/sweeps"
    echo ""
    echo -e "${GREEN}Common options for portal/door:${NC}"
    echo "  --num-envs N       Number of environments (default: 10)"
    echo "  --k N              Number of eigenvectors to compute (default: 20)"
    echo "  --output-dir PATH  Output directory for results"
    echo "  --seed N           Random seed (default: 42)"
    echo ""
    echo -e "${GREEN}Portal-specific options:${NC}"
    echo "  --num-portals N    Number of portals per environment (default: 10)"
    echo "  --num-rollouts N   Number of rollouts per environment (default: 100)"
    echo "  --num-steps N      Number of steps per rollout (default: 100)"
    echo "  --base-env NAME    Base environment name (default: GridRoom-4)"
    echo ""
    echo -e "${GREEN}Door-specific options:${NC}"
    echo "  --num-doors N      Number of doors per environment (default: 5)"
    echo "  --num-rollouts N   Number of rollouts per environment (default: 100)"
    echo "  --num-steps N      Number of steps per rollout (default: 100)"
    echo "  --log-scale        Use logarithmic scale for hitting times"
    echo ""
    echo -e "${GREEN}Sweep-specific options:${NC}"
    echo "  --results_dir PATH Directory containing sweep results (default: ./results/sweeps)"
    echo "  --exp_name NAME    Experiment name to analyze (default: batch_lr_sweep)"
    echo "  --env_type TYPE    Environment type (default: file)"
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

# Get experiment type
EXPERIMENT_TYPE=$1
shift

# Validate experiment type
case "$EXPERIMENT_TYPE" in
    portal)
        SCRIPT_PATH="$SCRIPT_DIR/../experiments/01/run_analysis.py"
        ;;
    door)
        SCRIPT_PATH="$SCRIPT_DIR/../experiments/02/run_analysis.py"
        ;;
    sweep)
        SCRIPT_PATH="$SCRIPT_DIR/analyze_sweep.py"
        ;;
    -h|--help|help)
        print_usage
        exit 0
        ;;
    *)
        echo -e "${RED}Error: Invalid experiment type '${EXPERIMENT_TYPE}'${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: Analysis script not found at ${SCRIPT_PATH}${NC}"
    exit 1
fi

# Display what we're running
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running ${EXPERIMENT_TYPE} analysis${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Script: $SCRIPT_PATH"
echo "Arguments: $@"
echo ""

# Run the analysis
echo -e "${YELLOW}Starting analysis...${NC}"
echo ""

python "$SCRIPT_PATH" "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Analysis completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Analysis failed with exit code ${EXIT_CODE}${NC}"
    echo -e "${RED}========================================${NC}"
    exit $EXIT_CODE
fi
