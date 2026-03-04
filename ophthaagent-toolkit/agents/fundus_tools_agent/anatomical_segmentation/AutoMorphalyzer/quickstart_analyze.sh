#!/bin/bash
# Quick start script for analyze_single.py

echo "========================================"
echo "  AutoMorphalyzer Single Image Analyzer"
echo "  Quick Start Examples"
echo "========================================"
echo ""

# Check if conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  Warning: No conda environment detected"
    echo "Please activate the environment first:"
    echo "  conda activate automorph-env"
    echo ""
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Example 1: Single image
echo "Example 1: Analyzing a single image"
echo "------------------------------------"
echo "Command:"
echo "  python analyze_single.py \\"
echo "    --input example_data/test_images/100.png \\"
echo "    --output results/single_image/"
echo ""
read -p "Run this example? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python analyze_single.py \
        --input example_data/test_images/100.png \
        --output results/single_image/

    echo ""
    echo "✓ Results saved to: results/single_image/"
    echo "  - CSV: results/single_image/M3/feature_measurements.csv"
    echo "  - JSON: results/single_image/measurements.json"
    echo ""
fi

# Example 2: Batch processing
echo "Example 2: Batch processing all images"
echo "---------------------------------------"
echo "Command:"
echo "  python analyze_single.py \\"
echo "    --input example_data/test_images/ \\"
echo "    --output results/batch_analysis/ \\"
echo "    --batch"
echo ""
read -p "Run this example? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python analyze_single.py \
        --input example_data/test_images/ \
        --output results/batch_analysis/ \
        --batch

    echo ""
    echo "✓ Results saved to: results/batch_analysis/"
    echo "  - JSON: results/batch_analysis/measurements.json"
    echo ""
fi

# Example 3: View JSON results
echo "Example 3: View JSON results"
echo "----------------------------"
read -p "View JSON results? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "results/single_image/measurements.json" ]; then
        echo "JSON Preview:"
        python -m json.tool results/single_image/measurements.json | head -50
        echo ""
        echo "Full file: results/single_image/measurements.json"
    else
        echo "No JSON file found. Run Example 1 first."
    fi
    echo ""
fi

echo "========================================"
echo "✓ Quick start complete!"
echo ""
echo "For more examples, see: ANALYZE_SINGLE_README.md"
echo "To run tests: python test_analyze_single.py"
echo "========================================"
