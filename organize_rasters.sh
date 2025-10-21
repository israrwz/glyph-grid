#!/bin/bash

# Script to organize raster images into label-based subdirectories
# Usage: ./organize_rasters.sh

set -e

DATASET_DIR="dataset"
RASTERS_DIR="data/rasters"
CSV_FILE="$DATASET_DIR/chars.csv"

# Check if CSV exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: $CSV_FILE not found"
    exit 1
fi

# Extract unique labels from CSV (last column, skip header)
# Use awk to properly handle CSV with quoted fields
echo "Reading labels from $CSV_FILE..."
labels=($(awk -F',' 'NR>1 {print $NF}' "$CSV_FILE" | sort -u))

echo "Found ${#labels[@]} unique labels"

# Process each split directory
for split in train val test; do
    split_dir="$RASTERS_DIR/$split"

    if [ ! -d "$split_dir" ]; then
        echo "Warning: $split_dir does not exist, skipping..."
        continue
    fi

    echo ""
    echo "Processing $split directory..."

    # Iterate through labels
    for label in "${labels[@]}"; do
        # Check if any files exist for this label
        matching_files=("$split_dir/${label}_"*.png)

        # Check if glob matched any files
        if [ -e "${matching_files[0]}" ]; then
            # Create label directory
            label_dir="$split_dir/$label"
            mkdir -p "$label_dir"

            # Move all matching files
            mv "$split_dir/${label}_"*.png "$label_dir/"
            echo "  ✓ $label: moved ${#matching_files[@]} files"
        fi
    done
done

echo ""
echo "Done! Rasters organized by label."
