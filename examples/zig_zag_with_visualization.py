import pandas as pd
import numpy as np
import time

from pricelevels.cluster import ZigZagClusterLevels
from pricelevels.visualization.levels_with_zigzag import plot_with_pivots

# Start timing
start_time = time.time()

# Load data with all necessary columns for ZigZag analysis
df = pd.read_csv('sample.txt')

# Convert to numpy array for better performance
close_values = df['Close'].values

# Parameters - Significantly increased to reduce pivot points
zig_zag_percent = 5.0  # Higher percentage means fewer pivot points
merge_percent = 1.0    # Higher percentage means fewer levels
min_bars_between = 80  # Higher value means fewer pivot points

# Create the ZigZag level finder with more restrictive parameters
zl = ZigZagClusterLevels(
    peak_percent_delta=zig_zag_percent,  # Increased to reduce sensitivity
    merge_distance=None,
    merge_percent=merge_percent,        # Increased to merge more levels
    min_bars_between_peaks=min_bars_between,  # Increased to require more space between peaks
    peaks='Low',                       # Use both high and low pivots
    level_selector='median'            # Use median for level calculation
)

# Fit the model
zl.fit(close_values)

# Print timing information
processing_time = time.time() - start_time
print(f"Processing completed in {processing_time:.4f} seconds")
print(f"Found {len(zl.levels) if zl.levels else 0} support/resistance levels")
print(f"ZigZag percent: {zig_zag_percent}%, Min bars between peaks: {min_bars_between}")

# Display chart
plot_with_pivots(close_values, zl.levels, zig_zag_percent)

# Uncomment to save chart to image
# plot_with_pivots(close_values, zl.levels, zig_zag_percent, path='support_resistance.png')
