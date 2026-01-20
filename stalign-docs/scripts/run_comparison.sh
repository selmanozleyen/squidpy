#!/usr/bin/env zsh
# =============================================================================
# Run all alignment comparisons between STalign and Squidpy
# =============================================================================
#
# This script runs all alignment examples with both methods and generates
# comparison plots.
#
# Requires two mamba environments:
#   - stalign: with STalign installed
#   - squidpy: with squidpy installed
#
# Usage:
#   ./run_comparison.sh           # Run all examples
#   ./run_comparison.sh visium    # Run only visium-visium affine
#   ./run_comparison.sh merfish   # Run only merfish-merfish (both affine and LDDMM)
#   ./run_comparison.sh visium_to_image  # Run only merfish-visium
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

# Initialize mamba for zsh
eval "$(mamba shell hook --shell zsh)"

# Parse arguments
RUN_ALL=true
RUN_VISIUM=false
RUN_MERFISH=false
RUN_VISIUM_TO_IMAGE=false

if [ $# -gt 0 ]; then
    RUN_ALL=false
    case "$1" in
        visium)
            RUN_VISIUM=true
            ;;
        merfish)
            RUN_MERFISH=true
            ;;
        visium_to_image|merfish_visium)
            RUN_VISIUM_TO_IMAGE=true
            ;;
        all)
            RUN_ALL=true
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [visium|merfish|visium_to_image|all]"
            exit 1
            ;;
    esac
fi

echo "=============================================="
echo "STalign vs Squidpy Alignment Comparison"
echo "=============================================="
echo ""

# =============================================================================
# 1. Visium-Visium Affine Alignment
# =============================================================================
if [ "$RUN_ALL" = true ] || [ "$RUN_VISIUM" = true ]; then
    echo "====== VISIUM-VISIUM AFFINE ======"
    echo ""

    echo "[STalign] Running visium-visium affine alignment..."
    mamba run -n stalign python visium-visium-alignment-affine-only.py
    echo ""

    echo "[Squidpy] Running visium-visium affine alignment..."
    mamba run -n squidpy python squidpy_example_visium_visium_affine.py
    echo ""

    echo "[Compare] Generating comparison..."
    mamba run -n squidpy python compare_visium_visium_affine.py
    echo ""
fi

# =============================================================================
# 2. MERFISH-MERFISH Affine Alignment
# =============================================================================
if [ "$RUN_ALL" = true ] || [ "$RUN_MERFISH" = true ]; then
    echo "====== MERFISH-MERFISH AFFINE ======"
    echo ""

    echo "[STalign] Running merfish-merfish affine alignment..."
    mamba run -n stalign python merfish-merfish-alignment-affine-only.py
    echo ""

    echo "[Squidpy] Running merfish-merfish affine alignment..."
    mamba run -n squidpy python squidpy_example_merfish_merfish_affine.py
    echo ""

    echo "[Compare] Generating comparison..."
    mamba run -n squidpy python compare_merfish_merfish_affine.py
    echo ""
fi

# =============================================================================
# 3. MERFISH-MERFISH Full LDDMM Alignment
# =============================================================================
if [ "$RUN_ALL" = true ] || [ "$RUN_MERFISH" = true ]; then
    echo "====== MERFISH-MERFISH LDDMM ======"
    echo ""

    echo "[STalign] Running merfish-merfish LDDMM alignment..."
    echo "  (This may take a while - 10000 iterations)"
    mamba run -n stalign python merfish-merfish-alignment.py
    echo ""

    echo "[Squidpy] Running merfish-merfish LDDMM alignment..."
    mamba run -n squidpy python squidpy_example_merfish_merfish.py
    echo ""

    echo "[Compare] Generating comparison..."
    mamba run -n squidpy python compare_merfish_merfish_lddmm.py
    echo ""
fi

# =============================================================================
# 4. MERFISH-Visium (Points to Image) Alignment
# =============================================================================
if [ "$RUN_ALL" = true ] || [ "$RUN_VISIUM_TO_IMAGE" = true ]; then
    echo "====== MERFISH-VISIUM (Points to Image) ======"
    echo ""

    echo "[STalign] Running merfish-visium alignment..."
    mamba run -n stalign python merfish-visium-alignment.py
    echo ""

    echo "[Squidpy] Running merfish-visium alignment..."
    mamba run -n squidpy python squidpy_example_merfish_visium.py
    echo ""

    echo "[Compare] Generating comparison..."
    mamba run -n squidpy python compare_merfish_visium.py
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "All done!"
echo "=============================================="
echo ""
echo "Output directories:"
echo "  - STalign results: ../output_stalign/"
echo "  - Squidpy results: ../output/"
echo "  - Comparisons:     ../comparison/"
echo ""
echo "Comparison figures:"
if [ "$RUN_ALL" = true ] || [ "$RUN_VISIUM" = true ]; then
    echo "  - ../comparison/visium_visium_affine_comparison.png"
fi
if [ "$RUN_ALL" = true ] || [ "$RUN_MERFISH" = true ]; then
    echo "  - ../comparison/merfish_merfish_affine_comparison.png"
    echo "  - ../comparison/merfish_merfish_lddmm_comparison.png"
fi
if [ "$RUN_ALL" = true ] || [ "$RUN_VISIUM_TO_IMAGE" = true ]; then
    echo "  - ../comparison/merfish_visium_comparison.png"
fi
echo ""
